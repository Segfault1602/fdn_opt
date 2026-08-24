#include "optimizer.h"

#include "audio_loss.h"
#include "block_spsa_optimizer.h"
#include "model.h"
#include "optimizer_internal.h"
#include "random_searcher.h"
#include <audio_utils/audio_analysis.h>

#include <armadillo>
#include <ensmallen.hpp>

#include <cmath>
#include <iostream>
#include <omp.h>
#include <thread>

template <typename T>
concept HasStepSize = requires(T a) { a.StepSize(); };

template <class T, class U>
struct is_same_template : std::false_type
{
};

template <template <class...> class C, class... R1s, class... R2s>
struct is_same_template<C<R1s...>, C<R2s...>> : std::true_type
{
};

template <class T, class U>
inline constexpr bool is_same_template_v = is_same_template<T, U>::value;

namespace fdn_optimization
{

class OptimCallback
{
  public:
    OptimCallback(std::stop_token stop_token, const FDNModel* model,
                  std::chrono::steady_clock::time_point optimization_start, double max_time_seconds,
                  uint64_t max_objective_evaluations, bool record_trajectory, double decay_rate = 0.99)
        : stop_token_(stop_token)
        , evaluation_count_(0)
        , decay_rate_(decay_rate)
        , model_(model)
        , optimization_start_(optimization_start)
        , max_time_seconds_(max_time_seconds)
        , max_objective_evaluations_(max_objective_evaluations)
        , record_trajectory_(record_trajectory)
    {
    }

    bool ShouldTerminate()
    {
        if (stop_token_.stop_requested())
            SetTerminationCode(1);
        else if (max_objective_evaluations_ > 0 && model_->GetEvaluationCount() >= max_objective_evaluations_)
            SetTerminationCode(3);
        else if (max_time_seconds_ > 0.0 &&
                 std::chrono::duration<double>(std::chrono::steady_clock::now() - optimization_start_).count() >=
                     max_time_seconds_)
            SetTerminationCode(2);

        return termination_code_.load() != 0;
    }

    std::string TerminationReason() const
    {
        if (explicit_termination_reason_)
        {
            return *explicit_termination_reason_;
        }
        switch (termination_code_.load())
        {
        case 1:
            return "canceled";
        case 2:
            return "time_budget";
        case 3:
            return "evaluation_budget";
        default:
            return "converged";
        }
    }

    void SetTerminationReason(std::string_view reason)
    {
        explicit_termination_reason_ = reason;
        if (reason == "canceled")
        {
            SetTerminationCode(1);
        }
        else if (reason == "time_budget")
        {
            SetTerminationCode(2);
        }
        else if (reason == "evaluation_budget")
        {
            SetTerminationCode(3);
        }
    }

    void InitializeLosses(const FDNModel& model)
    {
        const auto loss_functions = model.GetLossFunctions();
        std::scoped_lock lock(mutex_);
        individual_losses_.resize(loss_functions.size());
    }

    void RecordBlockSPSAStep(const BlockSPSAStep& step)
    {
        ++evaluation_count_;
        {
            std::scoped_lock lock(mutex_);
            // Updates that skip the accepted-point evaluation carry a stale evaluation, so they
            // must not add duplicate rows to the loss history.
            if (step.evaluated)
            {
                loss_history_.push_back(step.evaluation.total);
                for (size_t index = 0; index < step.evaluation.components.size(); ++index)
                {
                    individual_losses_[index].push_back(step.evaluation.components[index]);
                }
            }
            if (record_trajectory_)
            {
                std::vector<BlockSPSAProbeInfo> probes;
                probes.reserve(step.block_probes.size());
                for (const auto& probe : step.block_probes)
                {
                    probes.push_back({.block = probe.block,
                                      .block_size = probe.block_size,
                                      .visit = probe.visit,
                                      .learning_rate = probe.learning_rate,
                                      .perturbation = probe.perturbation,
                                      .probe_plus = probe.probe_plus,
                                      .probe_minus = probe.probe_minus,
                                      .paired_difference = probe.paired_difference,
                                      .absolute_paired_difference = probe.absolute_paired_difference,
                                      .gradient_norm = probe.gradient_norm,
                                      .step_norm = probe.step_norm});
                }
                trajectory_.push_back({.step = step.step,
                                       .total_loss = step.evaluation.total,
                                       .component_losses = step.evaluation.components,
                                       .best_loss = step.best_loss,
                                       .learning_rate = step.learning_rate,
                                       .gradient_norm = step.gradient_norm,
                                       .objective_evaluations = step.objective_evaluations,
                                       .elapsed_time = step.elapsed_time,
                                       .active_block = step.active_block,
                                       .block_visit = step.block_visit,
                                       .perturbation = step.perturbation,
                                       .directions_averaged = step.directions_averaged,
                                       .evaluated = step.evaluated,
                                       .improved_best = step.improved_best,
                                       .step_norm = step.step_norm,
                                       .block_probes = std::move(probes)});
            }
            ++completed_steps_;
        }
    }

    template <typename OptimizerType, typename FunctionType, typename MatType>
    void BeginOptimization(OptimizerType& optimizer, FunctionType& function, MatType&)
    {
        if constexpr (HasStepSize<OptimizerType>)
        {
            starting_step_size_ = optimizer.StepSize();
        }

        auto loss_functions = function.GetLossFunctions();
        {
            std::scoped_lock lock(mutex_);
            individual_losses_.resize(loss_functions.size());
        }
    }

    template <typename OptimizerType, typename FunctionType, typename MatType>
    bool Evaluate(OptimizerType&, FunctionType& function, const MatType& iterate, const double objective)
    {
        ++evaluation_count_;

        if constexpr (std::is_same_v<OptimizerType, ens::SPSA>)
        {
            if (step_was_taken_)
            {
                step_was_taken_ = false;
                SaveLossHistory(function, objective);
            }
        }

        if constexpr (std::is_same_v<OptimizerType, ens::CNE>)
        {
            if (de_pop_size_ > 0)
            {
                if (objective < de_best_objective_)
                {
                    de_best_objective_ = objective;
                    de_best_params_ = iterate;
                }
                // For DE, there is no easy way to get the best objective per generation so we have to keep track of
                // it manually.
                ++de_pop_evals_;
                if (de_pop_evals_ == de_pop_size_) // Each generation evaluates 2 * population size
                {
                    de_pop_evals_ = 0;
                    function.Evaluate(de_best_params_);
                    SaveLossHistory(function, de_best_objective_);
                    de_best_objective_ = std::numeric_limits<double>::max();
                    de_best_params_.zeros();
                }
            }
        }

        return ShouldTerminate();
    }

    template <typename OptimizerType, typename FunctionType, typename MatType, typename GradType>
    bool Gradient(OptimizerType&, FunctionType&, const MatType&, GradType& gradient)
    {
        last_gradient_norm_ = arma::norm(gradient, 2);
        return ShouldTerminate();
    }

    template <typename OptimizerType, typename FunctionType, typename MatType>
    bool EndEpoch(OptimizerType& optimizer, FunctionType& function, const MatType&, const size_t epoch,
                  const double objective)
    {
        double learning_rate = 0.0;
        if constexpr (HasStepSize<OptimizerType>)
            learning_rate = optimizer.StepSize();
        SaveLossHistory(function, objective, learning_rate, last_gradient_norm_);

        if constexpr (is_same_template_v<OptimizerType, ens::SGD<>>)
        {
            if (decay_step_size_ > 0 && epoch % decay_step_size_ == 0)
            {
                optimizer.StepSize() = optimizer.StepSize() * decay_rate_;
            }

            if (epoch_restarts_ > 0 && epoch % epoch_restarts_ == 0 && epoch != 0 && restart_count_ < max_restarts_)
            {
                optimizer.StepSize() = starting_step_size_;
                ++restart_count_;
            }
        }

        return ShouldTerminate();
    }

    template <typename OptimizerType, typename FunctionType, typename MatType>
    bool StepTaken(OptimizerType& optimizer, FunctionType& function, MatType& iterate)
    {
        step_was_taken_ = true;

        if constexpr (std::is_same_v<OptimizerType, ens::SA<ens::ExponentialSchedule>> ||
                      std::is_same_v<OptimizerType, ens::DE> || std::is_same_v<OptimizerType, ens::LBestPSO> ||
                      std::is_same_v<OptimizerType, ens::L_BFGS> ||
                      std::is_same_v<OptimizerType, ens::GradientDescent> ||
                      is_same_template_v<OptimizerType, ens::GradientDescentType<>> ||
                      is_same_template_v<OptimizerType, ens::CMAES<>> ||
                      is_same_template_v<OptimizerType, ens::ActiveCMAES<>>)
        {
            double learning_rate = 0.0;
            if constexpr (HasStepSize<OptimizerType>)
                learning_rate = optimizer.StepSize();
            SaveLossHistory(function, function.Evaluate(iterate), learning_rate, last_gradient_norm_);
        }

        return ShouldTerminate();
    }

    template <typename FunctionType>
    void SaveLossHistory(FunctionType&, double objective, double learning_rate = 0.0, double gradient_norm = 0.0)
    {
        auto individual_losses = LossRegistry::Instance().GetLosses();
        assert(individual_losses_.size() == individual_losses.size());
        std::scoped_lock lock(mutex_);
        loss_history_.push_back(objective);
        for (size_t i = 0; i < individual_losses.size(); ++i)
            individual_losses_[i].push_back(individual_losses[i]);

        if (record_trajectory_)
        {
            best_recorded_loss_ = std::min(best_recorded_loss_, objective);
            trajectory_.push_back({.step = completed_steps_,
                                   .total_loss = objective,
                                   .component_losses = individual_losses,
                                   .best_loss = best_recorded_loss_,
                                   .learning_rate = learning_rate,
                                   .gradient_norm = gradient_norm,
                                   .objective_evaluations = model_->GetEvaluationCount(),
                                   .elapsed_time = std::chrono::steady_clock::now() - optimization_start_,
                                   .active_block = std::nullopt,
                                   .block_visit = std::nullopt,
                                   .perturbation = std::nullopt,
                                   .directions_averaged = std::nullopt,
                                   .evaluated = std::nullopt,
                                   .improved_best = std::nullopt,
                                   .step_norm = std::nullopt,
                                   .block_probes = {}});
        }
        ++completed_steps_;
    }

    std::vector<std::vector<double>> GetLossHistory()
    {
        std::scoped_lock lock(mutex_);
        std::vector<std::vector<double>> all_losses;
        all_losses.push_back(loss_history_);
        for (const auto& losses : individual_losses_)
        {
            all_losses.push_back(losses);
        }
        return all_losses;
    }

    std::vector<OptimizationStepInfo> GetTrajectory()
    {
        std::scoped_lock lock(mutex_);
        return trajectory_;
    }

    std::stop_token stop_token_;
    std::atomic<uint32_t> evaluation_count_;
    double decay_rate_ = 0.99;
    size_t decay_step_size_ = 1;
    size_t epoch_restarts_ = 0;
    size_t restart_count_ = 0;
    size_t max_restarts_ = 0;

    int de_pop_size_ = 0;
    int de_pop_evals_ = 0;
    double de_best_objective_ = std::numeric_limits<double>::max();
    arma::mat de_best_params_;

  private:
    void SetTerminationCode(int code)
    {
        int expected = 0;
        termination_code_.compare_exchange_strong(expected, code);
    }

    std::mutex mutex_;
    std::vector<double> loss_history_;
    bool step_was_taken_ = false;
    double starting_step_size_ = 0.01;
    const FDNModel* model_;
    std::chrono::steady_clock::time_point optimization_start_;
    double max_time_seconds_;
    uint64_t max_objective_evaluations_;
    std::atomic<int> termination_code_{0};
    std::optional<std::string> explicit_termination_reason_;
    bool record_trajectory_;
    size_t completed_steps_ = 0;
    double last_gradient_norm_ = 0.0;
    double best_recorded_loss_ = std::numeric_limits<double>::infinity();

    std::vector<std::vector<double>> individual_losses_;
    std::vector<OptimizationStepInfo> trajectory_;
};

struct OptimizationVisitor
{
    arma::mat& params;
    FDNModel& model;
    OptimCallback* optim_callback;
    const OptimizationInfo& info;
    quill::Logger* logger;
    bool verbose;
    ObjectiveEvaluation initial_evaluation;
    std::optional<BlockSPSAResult> block_spsa_result;

    template <typename OptimizerType, typename FunctionType>
    void DoOptimizeFunction(OptimizerType& optimizer, FunctionType& function, arma::mat& coordinates)
    {
        ens::StoreBestCoordinates<arma::mat> store_best;

        if (verbose)
        {
            optimizer.Optimize(function, coordinates, store_best, ens::Report(), *optim_callback);
        }
        else
        {
            optimizer.Optimize(function, coordinates, store_best, *optim_callback);
        }
        const double final_objective = function.Evaluate(coordinates);
        if (detail::ShouldUseStoredBest(store_best.BestCoordinates(), store_best.BestObjective(), final_objective))
            coordinates = store_best.BestCoordinates();
    }

    template <typename OptimizerType>
    void DoOptimize(OptimizerType& optimizer)
    {
        DoOptimizeFunction(optimizer, model, params);
    }

    void operator()(AdamParameters& adam_params)
    {
        optim_callback->decay_rate_ = adam_params.learning_rate_decay;
        optim_callback->epoch_restarts_ = adam_params.epoch_restarts;
        optim_callback->decay_step_size_ = adam_params.decay_step_size;
        optim_callback->max_restarts_ = adam_params.max_restarts;

        LOG_INFO(logger, "Starting Adam optimization with step size: {}, beta1: {}, beta2: {}, tolerance: {}",
                 adam_params.step_size, adam_params.beta1, adam_params.beta2, adam_params.tolerance);

        ens::Adam optimizer(adam_params.step_size, 1, adam_params.beta1, adam_params.beta2, 1e-8,
                            adam_params.max_iterations, adam_params.tolerance, false, true, true);
        DoOptimize(optimizer);
    }

    void operator()(SPSAParameters& spsa_params)
    {
        LOG_INFO(logger,
                 "Starting SPSA optimization with alpha: {}, gamma: {}, step size: {}, evaluation step size: {}, "
                 "max iterations: {}, tolerance: {}",
                 spsa_params.alpha, spsa_params.gamma, spsa_params.step_size, spsa_params.evaluationStepSize,
                 spsa_params.max_iterations, spsa_params.tolerance);
        ens::SPSA optimizer(spsa_params.alpha, spsa_params.gamma, spsa_params.step_size, spsa_params.evaluationStepSize,
                            spsa_params.max_iterations, spsa_params.tolerance);
        DoOptimize(optimizer);
    }

    void operator()(BlockSPSAParameters& parameters)
    {
        LOG_INFO(logger,
                 "Starting BlockSPSA optimization with mode: {}, block strategy: {}, directions per block: {}, "
                 "alpha: {}, gamma: {}, step size: {}, evaluation step size: {}, max iterations: {}, tolerance: {}",
                 parameters.mode == BlockSPSAMode::SnapshotSweepAll ? "snapshot" : "random-one",
                 parameters.block_strategy == ParameterBlockStrategy::Semantic ? "semantic" : "fixed",
                 parameters.directions_per_block, parameters.alpha, parameters.gamma, parameters.step_size,
                 parameters.evaluation_step_size, parameters.max_iterations, parameters.tolerance);

        const auto layout = BuildParameterLayout(info.initial_fdn_config.fdn_size, info.parameters_to_optimize);
        const auto blocks =
            parameters.block_strategy == ParameterBlockStrategy::Semantic
                ? BuildSemanticParameterBlocks(layout, info.initial_fdn_config.fdn_size, parameters.three_band_grouping)
                : BuildContiguousParameterBlocks(layout.total_size, parameters.contiguous_block_size);

        optim_callback->InitializeLosses(model);
        BlockSPSAOptimizer optimizer(parameters);
        block_spsa_result =
            optimizer.Optimize(model, params, initial_evaluation, blocks, info.seed, info.gradient_threads,
                               info.max_objective_evaluations, info.max_time_seconds, optim_callback->stop_token_,
                               [this](const BlockSPSAStep& step) { optim_callback->RecordBlockSPSAStep(step); });
        params = block_spsa_result->parameters;
        LossRegistry::Instance().RegisterLoss(block_spsa_result->evaluation.components);
        optim_callback->SetTerminationReason(block_spsa_result->termination_reason);
    }

    void operator()(SimulatedAnnealingParameters& p)
    {
        LOG_INFO(logger,
                 "Starting Simulated Annealing optimization with initial temperature: {}, max iterations: {}, init "
                 "moves: {}, move control sweep: {}, max tolerance sweep: {}, max move coefficient: {}, init move "
                 "coefficient: {}, gain: {}, tolerance: {}",
                 p.initial_temperature, p.max_iterations, p.init_moves, p.move_ctrl_sweep, p.max_tolerance_sweep,
                 p.max_move_coef, p.init_move_coef, p.gain, p.tolerance);
        ens::SA optimizer(ens::ExponentialSchedule(), p.max_iterations, p.initial_temperature, p.init_moves,
                          p.move_ctrl_sweep, p.tolerance, p.max_tolerance_sweep, p.max_move_coef, p.init_move_coef,
                          p.gain);

        DoOptimize(optimizer);
    }

    void operator()(CNEParameters& p)
    {
        LOG_INFO(logger,
                 "Starting CNE optimization with population size: {}, max generations: {}, mutation probability: {}, "
                 "mutation size: {}, select percent: {}, tolerance: {}",
                 p.population_size, p.max_generations, p.mutation_probability, p.mutation_size, p.select_percent,
                 p.tolerance);

        optim_callback->de_pop_size_ = static_cast<int>(p.population_size);
        ens::CNE optimizer(p.population_size, p.max_generations, p.mutation_probability, p.mutation_size,
                           p.select_percent, p.tolerance);

        DoOptimize(optimizer);
    }

    void operator()(DifferentialEvolutionParameters& p)
    {
        LOG_INFO(logger,
                 "Starting Differential Evolution optimization with population size: {}, max generation: {}, crossover "
                 "rate: {}, differential weight: {}, tolerance: {}",
                 p.population_size, p.max_generation, p.crossover_rate, p.differential_weight, p.tolerance);
        optim_callback->de_pop_size_ = static_cast<int>(p.population_size);
        ens::DE optimizer(p.population_size, p.max_generation, p.crossover_rate, p.differential_weight, p.tolerance);

        DoOptimize(optimizer);
    }

    void operator()(PSOParameters& p)
    {
        LOG_INFO(logger,
                 "Starting PSO optimization with num particles: {}, max iterations: {}, horizon size: {}, "
                 "exploitation factor: {}, exploration factor: {}, tolerance: {}",
                 p.num_particles, p.max_iterations, p.horizon_size, p.exploitation_factor, p.exploration_factor,
                 p.tolerance);

        ens::LBestPSO optimizer(p.num_particles, -1.0, 1.0, p.max_iterations, p.horizon_size, p.tolerance,
                                p.exploitation_factor, p.exploration_factor);
        optimizer.NumThreads() = info.optimizer_threads;

        DoOptimize(optimizer);
    }

    void operator()(RandomSearchParameters& p)
    {
        RandomSearcher optimizer;
        optim_callback->BeginOptimization(optimizer, model, params);
        std::stop_token stop_token = optim_callback->stop_token_;
        params = model.GetInitialParams();
        optimizer.StartSearch(model, stop_token, p.time_limit_seconds, optim_callback);

        params = optimizer.GetBestParams();
        optim_callback->evaluation_count_ = optimizer.GetEvaluationCount();
        const double evaluated_objective = model.Evaluate(params);
        const double best_objective = optimizer.GetBestObjective();
        optim_callback->SaveLossHistory(model, std::isfinite(best_objective) ? best_objective : evaluated_objective);
    }

    void operator()(L_BFGSParameters& p)
    {
        LOG_INFO(logger,
                 "Starting L-BFGS optimization with num basis: {}, max iterations: {}, wolfe: {}, min gradient norm: "
                 "{}, factor: {}, max line search trials: {}, min step: {}, max step: {}",
                 p.num_basis, p.max_iterations, p.wolfe, p.min_gradient_norm, p.factor, p.max_line_search_trials,
                 p.min_step, p.max_step);
        constexpr double kArmijoConstant = 1e-4;
        ens::L_BFGS optimizer(p.num_basis, p.max_iterations, kArmijoConstant, p.wolfe, p.min_gradient_norm, p.factor,
                              p.max_line_search_trials, p.min_step, p.max_step);

        DoOptimize(optimizer);
    }

    void operator()(GradientDescentParameters& p)
    {
        LOG_INFO(logger,
                 "Starting Gradient Descent optimization with step size: {}, max iterations: {}, tolerance: {}, kappa: "
                 "{}, phi: {}, momentum: {}, min gain: {}, max step norm: {}",
                 p.step_size, p.max_iterations, p.tolerance, p.kappa, p.phi, p.momentum, p.min_gain, p.max_step_norm);
        if (p.max_step_norm > 0.0)
        {
            detail::CappedMomentumDeltaBarDeltaUpdate update_policy(p.kappa, p.phi, p.momentum, p.min_gain,
                                                                    p.max_step_norm);
            ens::GradientDescentType<detail::CappedMomentumDeltaBarDeltaUpdate, ens::NoDecay> optimizer(
                p.step_size, p.max_iterations, p.tolerance, update_policy, ens::NoDecay(), false);
            DoOptimize(optimizer);
        }
        else
        {
            // Preserve the exact historical Ensmallen implementation when the cap is disabled.
            ens::MomentumDeltaBarDelta optimizer(p.step_size, p.max_iterations, p.tolerance, p.kappa, p.phi, p.momentum,
                                                 p.min_gain, false);
            DoOptimize(optimizer);
        }
    }

    void operator()(CMAESParameters& p)
    {
        LOG_INFO(logger,
                 "Starting CMA-ES optimization with population size: {}, max iterations: {}, tolerance: {}, step size: "
                 "{}",
                 p.population_size, p.max_iterations, p.tolerance, p.step_size);
        const bool has_matching_coordinates =
            std::ranges::any_of(info.parameters_to_optimize, [](OptimizationParamType type) {
                return type == OptimizationParamType::AttenuationFilters ||
                       type == OptimizationParamType::AttenuationFilters_3Band ||
                       type == OptimizationParamType::TonecorrectionFilters ||
                       type == OptimizationParamType::OverallGain;
            });

        if (has_matching_coordinates)
        {
            // Matching uses heterogeneous coordinates.  Valid seeded T60 logits can be near +/-17,
            // tone coordinates map +/-1 to +/-12 dB, and overall gain is represented in log space.
            // One global [-1, 1] box changes the physical initial T60s; no box permits extreme tone
            // and gain candidates that overflow the objective.  Use parameter-aware bounds instead.
            const auto layout = BuildParameterLayout(info.initial_fdn_config.fdn_size, info.parameters_to_optimize);
            const auto coordinate_bounds = detail::BuildMatchingCMAESBounds(layout, info.matching_parameters);
            ens::BoundaryBoxConstraint bounds(coordinate_bounds.lower, coordinate_bounds.upper);
            ens::ActiveCMAES<ens::FullSelection, ens::BoundaryBoxConstraint<>> optimizer(
                p.population_size, bounds, 1, p.max_iterations, p.tolerance, ens::FullSelection(), p.step_size);
            DoOptimize(optimizer);
        }
        else
        {
            ens::BoundaryBoxConstraint b(-1.0, 1.0);
            ens::ActiveCMAES<ens::FullSelection, ens::BoundaryBoxConstraint<>> optimizer(
                p.population_size, b, 1, p.max_iterations, p.tolerance, ens::FullSelection(), p.step_size);
            DoOptimize(optimizer);
        }
    }
};

FDNOptimizer::FDNOptimizer(quill::Logger* logger, bool verbose)
    : logger_(logger)
    , verbose_(verbose)
    , status_(OptimizationStatus::Ready)
{
}

FDNOptimizer::~FDNOptimizer()
{
    thread_.request_stop();
    if (thread_.joinable())
    {
        thread_.join();
    }
}

void FDNOptimizer::SetStatus(OptimizationStatus status)
{
    {
        std::scoped_lock lock(mutex_);
        status_.store(status);
    }
    status_cv_.notify_all();
}

void FDNOptimizer::SetLossFunctions(std::span<std::shared_ptr<AudioLoss>> loss_functions)
{
    std::scoped_lock lock(mutex_);
    loss_functions_ = std::vector<std::shared_ptr<AudioLoss>>(loss_functions.begin(), loss_functions.end());
}

void FDNOptimizer::StartOptimization(OptimizationInfo& info)
{
    LOG_INFO(logger_, "Optimizing for: ");
    for (auto p : info.parameters_to_optimize)
    {
        LOG_INFO(logger_, "  - {}", OptimizationParamTypeToString(p));
    }

    OptimizationInfo info_copy = info;

    LOG_INFO(logger_, "Starting optimization.");
    {
        std::scoped_lock lock(mutex_);
        const auto current_status = status_.load();
        if (current_status == OptimizationStatus::Running || current_status == OptimizationStatus::StartRequested ||
            current_status == OptimizationStatus::CancelRequested)
        {
            LOG_WARNING(logger_, "Optimization is already running.");
            return;
        }

        status_.store(OptimizationStatus::StartRequested);
        start_time_ = std::chrono::steady_clock::now();
        thread_ = std::jthread([this, info_copy](std::stop_token st) {
            try
            {
                ThreadProc(st, info_copy);
            }
            catch (const std::exception& error)
            {
                LOG_ERROR(logger_, "Optimization failed: {}", error.what());
                SetStatus(OptimizationStatus::Failed);
            }
            catch (...)
            {
                LOG_ERROR(logger_, "Optimization failed with an unknown exception.");
                SetStatus(OptimizationStatus::Failed);
            }
        });
    }
    status_cv_.notify_all();
}

void FDNOptimizer::CancelOptimization()
{
    {
        std::scoped_lock lock(mutex_);
        const auto current_status = status_.load();
        if (current_status != OptimizationStatus::Running && current_status != OptimizationStatus::StartRequested)
        {
            LOG_WARNING(logger_, "Optimization is not running.");
            return;
        }
        status_.store(OptimizationStatus::CancelRequested);
    }

    LOG_INFO(logger_, "Requesting optimization cancellation.");
    status_cv_.notify_all();
    thread_.request_stop();
}

void FDNOptimizer::ResetStatus()
{
    bool cancel_active_optimization = false;
    {
        std::scoped_lock lock(mutex_);
        const auto current_status = status_.load();
        cancel_active_optimization = current_status == OptimizationStatus::Running ||
                                     current_status == OptimizationStatus::StartRequested ||
                                     current_status == OptimizationStatus::CancelRequested;
        if (cancel_active_optimization)
            status_.store(OptimizationStatus::CancelRequested);
    }

    if (cancel_active_optimization)
    {
        LOG_WARNING(logger_, "Cannot reset status while optimization is running. Cancelling first.");
        status_cv_.notify_all();
        thread_.request_stop();
        if (thread_.joinable())
            thread_.join();
    }
    SetStatus(OptimizationStatus::Ready);
}

OptimizationStatus FDNOptimizer::GetStatus() const
{
    return status_.load();
}

OptimizationStatus FDNOptimizer::WaitForCompletion()
{
    std::unique_lock lock(mutex_);
    status_cv_.wait(lock, [this] {
        const auto status = status_.load();
        return status == OptimizationStatus::Completed || status == OptimizationStatus::Canceled ||
               status == OptimizationStatus::Failed;
    });
    return status_.load();
}

OptimizationProgressInfo FDNOptimizer::GetProgress()
{
    std::scoped_lock lock(mutex_);

    OptimizationProgressInfo progress;
    progress.elapsed_time = std::chrono::steady_clock::now() - start_time_;

    if (optim_callback_)
    {
        progress.evaluation_count = optim_callback_->evaluation_count_.load();
        progress.loss_history = optim_callback_->GetLossHistory();
    }
    else
    {
        progress.evaluation_count = 0;
    }

    return progress;
}

OptimizationResult FDNOptimizer::GetResult()
{
    std::scoped_lock lock(mutex_);
    return optimization_result_;
}

void FDNOptimizer::ThreadProc(std::stop_token stop_token, OptimizationInfo info)
{
    LOG_INFO(logger_, "Optimization thread started.");
    bool canceled_before_start = false;
    {
        std::scoped_lock lock(mutex_);
        canceled_before_start = status_.load() == OptimizationStatus::CancelRequested;
        status_.store(canceled_before_start ? OptimizationStatus::Canceled : OptimizationStatus::Running);
    }
    status_cv_.notify_all();
    if (canceled_before_start)
    {
        LOG_WARNING(logger_, "Optimization was canceled before the worker started.");
        return;
    }

    arma::arma_rng::set_seed(info.seed);
    omp_set_max_active_levels(1);
    if (info.gradient_threads == 0)
        info.gradient_threads = static_cast<uint32_t>(std::max(1, omp_get_max_threads()));
    if (info.optimizer_threads == 0)
        info.optimizer_threads = static_cast<uint32_t>(std::max(1, omp_get_max_threads()));

    bool optimizing_filters =
        std::ranges::find(info.parameters_to_optimize, fdn_optimization::OptimizationParamType::AttenuationFilters) !=
        info.parameters_to_optimize.end();

    optimizing_filters |= (std::ranges::find(info.parameters_to_optimize,
                                             fdn_optimization::OptimizationParamType::TonecorrectionFilters) !=
                           info.parameters_to_optimize.end());
    optimizing_filters |= (std::ranges::find(info.parameters_to_optimize,
                                             fdn_optimization::OptimizationParamType::AttenuationFilters_3Band) !=
                           info.parameters_to_optimize.end());

    if (optimizing_filters && info.target_rir.empty())
    {
        LOG_ERROR(logger_, "Target RIR must be provided when optimizing filters. Cancelling optimization.");
        SetStatus(OptimizationStatus::Failed);
        return;
    }

    if (optimizing_filters && info.ir_size != info.target_rir.size())
    {
        LOG_WARNING(logger_,
                    "IR size ({}) does not match target RIR size ({}). Adjusting IR size to match target RIR size.",
                    info.ir_size, info.target_rir.size());
        info.ir_size = static_cast<uint32_t>(info.target_rir.size());
    }

    const auto setup_start = std::chrono::steady_clock::now();
    FDNModel model(info.initial_fdn_config, info.ir_size, info.parameters_to_optimize, info.gradient_method,
                   info.matching_parameters);
    model.SetGradientThreads(info.gradient_threads);

    double gradient_delta = std::visit(
        [](auto&& params) -> double {
            using T = std::decay_t<decltype(params)>;
            if constexpr (std::is_same_v<T, AdamParameters> || std::is_same_v<T, L_BFGSParameters> ||
                          std::is_same_v<T, GradientDescentParameters>)
            {
                return params.gradient_delta;
            }
            else
            {
                return 1e-4; // Default gradient delta for other optimizers when optimizing filters
            }
        },
        info.optimizer_params);

    model.SetGradientDelta(gradient_delta);
    if (!info.t60_estimates.empty())
        model.SetT60Estimates(info.t60_estimates);

    if (!info.early_fir.empty())
    {
        LOG_INFO(logger_, "Setting early FIR with size {}", info.early_fir.size());
        model.SetEarlyFir(info.early_fir, info.early_fir_mode);
    }

    LOG_INFO(logger_, "Gradient method: {}, Gradient delta: {}",
             info.gradient_method == GradientMethod::CentralDifferences ? "Central Differences" : "Forward Differences",
             gradient_delta);

    model.SetLossFunctions(loss_functions_);

    arma::mat params = model.GetInitialParams();
    if (verbose_)
    {
        LOG_INFO(logger_, "Initial config: {}", model.PrintFDNConfig(params));
        std::stringstream param_stream;
        param_stream << params;
        LOG_INFO(logger_, "Initial parameters: {}", param_stream.str());
    }

    const auto initial_evaluation_start = std::chrono::steady_clock::now();
    auto initial_evaluation = model.EvaluateDetailed(params);
    const auto initial_loss = initial_evaluation.total;
    const auto initial_evaluation_end = std::chrono::steady_clock::now();
    if (!std::isfinite(initial_loss))
        throw std::runtime_error("Initial objective is not finite.");
    LOG_INFO(logger_, "Initial loss: {}", initial_loss);

    sfFDN::FDNConfig initial_config = model.GetFDNConfig(params);

    const auto optimization_start = std::chrono::steady_clock::now();
    {
        std::scoped_lock lock(mutex_);
        optim_callback_ = std::make_unique<OptimCallback>(stop_token, &model, optimization_start, info.max_time_seconds,
                                                          info.max_objective_evaluations, info.record_trajectory);
    }

    OptimizationVisitor visitor{.params = params,
                                .model = model,
                                .optim_callback = optim_callback_.get(),
                                .info = info,
                                .logger = logger_,
                                .verbose = verbose_,
                                .initial_evaluation = initial_evaluation,
                                .block_spsa_result = std::nullopt};
    std::visit(visitor, info.optimizer_params);
    const auto optimization_end = std::chrono::steady_clock::now();

    const auto final_evaluation_start = std::chrono::steady_clock::now();
    const bool used_block_spsa = visitor.block_spsa_result.has_value();
    const ObjectiveEvaluation final_evaluation =
        used_block_spsa ? visitor.block_spsa_result->evaluation : model.EvaluateDetailed(params);
    const double final_loss = final_evaluation.total;
    const auto final_evaluation_end = std::chrono::steady_clock::now();
    if (!std::isfinite(final_loss))
        throw std::runtime_error("Final objective is not finite.");
    LOG_INFO(logger_, "Final loss: {}", final_loss);
    if (verbose_)
    {
        LOG_INFO(logger_, "Final config:\n{}", model.PrintFDNConfig(params));
        std::stringstream param_stream;
        param_stream << params;
        LOG_INFO(logger_, "Optimization finished. Final parameters: {}", param_stream.str());
    }

    OptimizationStatus final_status;
    {
        std::scoped_lock lock(mutex_);

        optimized_config_ = model.GetFDNConfig(params);

        if (!optimizing_filters)
        {
            // If we are not optimizing filters, we can copy the initial filter configs to the optimized config
            optimized_config_.attenuation_filter_bank_config = info.initial_fdn_config.attenuation_filter_bank_config;
        }

        optimization_result_.initial_fdn_config = initial_config;
        optimization_result_.optimized_fdn_config = optimized_config_;
        optimization_result_.total_time = std::chrono::steady_clock::now() - start_time_;
        optimization_result_.setup_time = initial_evaluation_start - setup_start;
        optimization_result_.initial_evaluation_time = initial_evaluation_end - initial_evaluation_start;
        optimization_result_.optimizer_time = optimization_end - optimization_start;
        optimization_result_.final_evaluation_time =
            used_block_spsa ? std::chrono::duration<double>::zero() : final_evaluation_end - final_evaluation_start;
        optimization_result_.total_evaluations = optim_callback_->evaluation_count_.load();
        optimization_result_.objective_evaluations = model.GetEvaluationCount();
        optimization_result_.gradient_threads = info.gradient_threads;
        optimization_result_.optimizer_threads = info.optimizer_threads;
        optimization_result_.loss_history = optim_callback_->GetLossHistory();
        optimization_result_.best_loss = final_loss;
        optimization_result_.final_losses = final_evaluation.components;
        optimization_result_.trajectory = optim_callback_->GetTrajectory();
        optimization_result_.termination_reason = optim_callback_->TerminationReason();
        if (visitor.block_spsa_result)
        {
            optimization_result_.parameter_block_count = visitor.block_spsa_result->block_count;
            optimization_result_.optimizer_stability_constant = visitor.block_spsa_result->stability_constant;
            optimization_result_.optimizer_stall_window = visitor.block_spsa_result->stall_window;
        }
        else
        {
            optimization_result_.parameter_block_count.reset();
            optimization_result_.optimizer_stability_constant.reset();
            optimization_result_.optimizer_stall_window.reset();
        }

        optimization_result_.loss_names.clear();
        auto loss_functions = model.GetLossFunctions();
        for (const auto& lf : loss_functions)
        {
            optimization_result_.loss_names.push_back(lf->GetName());
        }

        final_status = status_.load() == OptimizationStatus::CancelRequested ? OptimizationStatus::Canceled
                                                                             : OptimizationStatus::Completed;
        if (final_status == OptimizationStatus::Canceled)
            optimization_result_.termination_reason = "canceled";
        status_.store(final_status);
    }
    status_cv_.notify_all();

    if (final_status == OptimizationStatus::Canceled)
        LOG_WARNING(logger_, "Optimization was canceled.");
    else
        LOG_INFO(logger_, "Optimization thread completed.");
}

} // namespace fdn_optimization