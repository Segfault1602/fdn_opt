#include "block_spsa_optimizer.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <exception>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

#include <omp.h>

namespace
{

constexpr std::uint64_t kSignStream = 0x5f4d3c2b1a098765ULL;
constexpr std::uint64_t kScheduleStream = 0x84a19c37d52e6bf0ULL;

// Signals that the trajectory itself diverged, as opposed to a configuration or overflow error.
// Only this type ends a run early while keeping the best point found so far.
class NonFiniteObjectiveError : public std::runtime_error
{
  public:
    using std::runtime_error::runtime_error;
};

std::uint64_t Mix(std::uint64_t value)
{
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31U);
}

std::uint64_t RandomValue(std::uint32_t seed, std::uint64_t stream, std::size_t update, std::size_t block,
                          std::size_t replication, std::size_t coordinate)
{
    std::uint64_t value = Mix(seed);
    value = Mix(value ^ stream);
    value = Mix(value ^ update);
    value = Mix(value ^ block);
    value = Mix(value ^ replication);
    return Mix(value ^ coordinate);
}

double Rademacher(std::uint32_t seed, std::size_t update, std::size_t block, std::size_t replication,
                  std::size_t coordinate)
{
    return (RandomValue(seed, kSignStream, update, block, replication, coordinate) & 1ULL) == 0 ? -1.0 : 1.0;
}

std::size_t UniformIndex(std::uint32_t seed, std::size_t update, std::size_t item, std::size_t upper_bound)
{
    if (upper_bound == 0)
    {
        throw std::invalid_argument("Random selection requires a nonempty range.");
    }

    const std::uint64_t bound = static_cast<std::uint64_t>(upper_bound);
    const std::uint64_t threshold = static_cast<std::uint64_t>(-bound) % bound;
    for (std::size_t attempt = 0;; ++attempt)
    {
        const auto value = RandomValue(seed, kScheduleStream, update, item, attempt, 0);
        if (value >= threshold)
        {
            return static_cast<std::size_t>(value % bound);
        }
    }
}

std::vector<std::size_t> ShuffledBlocks(std::uint32_t seed, std::size_t epoch, std::size_t block_count)
{
    std::vector<std::size_t> blocks(block_count);
    std::iota(blocks.begin(), blocks.end(), 0);
    for (std::size_t remaining = block_count; remaining > 1; --remaining)
    {
        const auto selected = UniformIndex(seed, epoch, remaining, remaining);
        std::swap(blocks[remaining - 1], blocks[selected]);
    }
    return blocks;
}

void ValidateParameters(const fdn_optimization::BlockSPSAParameters& parameters,
                        std::span<const fdn_optimization::ParameterBlock> blocks, const arma::mat& coordinates)
{
    if (coordinates.n_rows != 1 || coordinates.n_cols == 0)
    {
        throw std::invalid_argument("BlockSPSA requires a nonempty one-row coordinate matrix.");
    }
    if (blocks.empty())
    {
        throw std::invalid_argument("BlockSPSA requires at least one parameter block.");
    }
    if (parameters.directions_per_block == 0)
    {
        throw std::invalid_argument("BlockSPSA directions_per_block must be positive.");
    }
    if (parameters.accepted_evaluation_interval == 0)
    {
        throw std::invalid_argument("BlockSPSA accepted_evaluation_interval must be positive.");
    }
    if (!std::isfinite(parameters.max_step_norm) || parameters.max_step_norm < 0.0)
    {
        throw std::invalid_argument("BlockSPSA max_step_norm must be finite and nonnegative.");
    }
    if (parameters.max_iterations == 0)
    {
        throw std::invalid_argument("BlockSPSA max_iterations must be positive.");
    }
    for (const auto& scale : parameters.block_scales)
    {
        if (!std::isfinite(scale.a_scale) || scale.a_scale <= 0.0 || !std::isfinite(scale.c_scale) ||
            scale.c_scale <= 0.0)
        {
            throw std::invalid_argument("BlockSPSA block gain scales must be finite and positive.");
        }
    }
    if (!std::isfinite(parameters.alpha) || parameters.alpha <= 0.0 || !std::isfinite(parameters.gamma) ||
        parameters.gamma <= 0.0 || !std::isfinite(parameters.step_size) || parameters.step_size <= 0.0 ||
        !std::isfinite(parameters.evaluation_step_size) || parameters.evaluation_step_size <= 0.0 ||
        !std::isfinite(parameters.tolerance) || parameters.tolerance < 0.0)
    {
        throw std::invalid_argument("BlockSPSA gains and tolerance must be finite and valid.");
    }

    std::vector<unsigned int> coverage(coordinates.n_cols, 0);
    for (const auto& block : blocks)
    {
        if (block.coordinates.empty())
        {
            throw std::invalid_argument("BlockSPSA parameter blocks must be nonempty.");
        }
        for (const auto coordinate : block.coordinates)
        {
            if (coordinate >= coordinates.n_cols)
            {
                throw std::invalid_argument("BlockSPSA block coordinate is out of range.");
            }
            ++coverage[coordinate];
        }
    }
    if (std::ranges::any_of(coverage, [](unsigned int count) { return count != 1; }))
    {
        throw std::invalid_argument("BlockSPSA parameter blocks must form an exact partition.");
    }
}

void ValidateOptimizationInputs(const fdn_optimization::BlockSPSAParameters& parameters,
                                std::span<const fdn_optimization::ParameterBlock> blocks, const arma::mat& coordinates,
                                const fdn_optimization::BlockSPSAOptimizer::ObjectiveFunction& objective,
                                const fdn_optimization::ObjectiveEvaluation& initial_evaluation)
{
    ValidateParameters(parameters, blocks, coordinates);
    if (!objective)
    {
        throw std::invalid_argument("BlockSPSA requires an objective function.");
    }
    if (!std::isfinite(initial_evaluation.total))
    {
        throw std::invalid_argument("BlockSPSA initial objective is not finite.");
    }
}

bool TimeExpired(std::chrono::steady_clock::time_point start, double max_time_seconds)
{
    return max_time_seconds > 0.0 &&
           std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() >= max_time_seconds;
}

double Gain(double scale, double exponent, double stability, std::size_t clock)
{
    return scale / std::pow(stability + static_cast<double>(clock) + 1.0, exponent);
}

// Maps a parameter block onto the gain-scale class that controls its schedule.
fdn_optimization::BlockScaleClass ScaleClassOf(const fdn_optimization::ParameterBlock& block)
{
    using fdn_optimization::BlockScaleClass;
    using fdn_optimization::OptimizationParamType;

    if (!block.semantic_type)
    {
        return BlockScaleClass::Default;
    }
    switch (*block.semantic_type)
    {
    case OptimizationParamType::Gains:
        return block.semantic_index == 0 ? BlockScaleClass::GainsInput : BlockScaleClass::GainsOutput;
    case OptimizationParamType::Matrix:
    case OptimizationParamType::Matrix_Householder:
    case OptimizationParamType::Matrix_Circulant:
        return BlockScaleClass::Matrix;
    case OptimizationParamType::AttenuationFilters:
    case OptimizationParamType::AttenuationFilters_3Band:
        return BlockScaleClass::Attenuation;
    case OptimizationParamType::TonecorrectionFilters:
        return BlockScaleClass::Tone;
    case OptimizationParamType::OverallGain:
        return BlockScaleClass::OverallGain;
    default:
        return BlockScaleClass::Default;
    }
}

struct ResolvedBlockScale
{
    double a_scale = 1.0;
    double c_scale = 1.0;
};

// Resolves the per-block multiplicative scales once, before the optimization loop.
std::vector<ResolvedBlockScale> ResolveBlockScales(const fdn_optimization::BlockSPSAParameters& parameters,
                                                   std::span<const fdn_optimization::ParameterBlock> blocks)
{
    ResolvedBlockScale fallback;
    for (const auto& scale : parameters.block_scales)
    {
        if (scale.scale_class == fdn_optimization::BlockScaleClass::Default)
        {
            fallback = {.a_scale = scale.a_scale, .c_scale = scale.c_scale};
        }
    }

    std::vector<ResolvedBlockScale> resolved(blocks.size(), fallback);
    for (std::size_t index = 0; index < blocks.size(); ++index)
    {
        const auto block_class = ScaleClassOf(blocks[index]);
        for (const auto& scale : parameters.block_scales)
        {
            if (scale.scale_class == block_class && scale.scale_class != fdn_optimization::BlockScaleClass::Default)
            {
                resolved[index] = {.a_scale = scale.a_scale, .c_scale = scale.c_scale};
            }
        }
        if (parameters.probe_radius_normalization == fdn_optimization::ProbeRadiusNormalization::SqrtDimension)
        {
            resolved[index].c_scale /= std::sqrt(static_cast<double>(blocks[index].coordinates.size()));
        }
    }
    return resolved;
}

// Stops a run once the best loss fails to improve by more than a relative tolerance across a
// window of evaluated accepted points. A per-update absolute test is unusable for block-coordinate
// updates, whose individual objective changes are legitimately tiny.
class StallMonitor
{
  public:
    StallMonitor(std::size_t window, double tolerance, double initial_best)
        : window_(window)
        , tolerance_(tolerance)
        , history_(window + 1, initial_best)
    {
    }

    bool Enabled() const
    {
        return window_ > 0 && tolerance_ > 0.0 && !history_.empty();
    }

    bool Stalled(double best_loss)
    {
        if (!Enabled())
        {
            return false;
        }
        const double previous = history_[position_];
        history_[position_] = best_loss;
        position_ = (position_ + 1) % history_.size();
        if (observations_ < history_.size())
        {
            ++observations_;
            return false;
        }
        constexpr double kRelativeFloor = 1e-12;
        const double threshold = tolerance_ * std::max(std::abs(previous), kRelativeFloor);
        return (previous - best_loss) < threshold;
    }

  private:
    std::size_t window_ = 0;
    double tolerance_ = 0.0;
    std::vector<double> history_;
    std::size_t position_ = 0;
    std::size_t observations_ = 0;
};

std::optional<std::string> InterruptionReason(std::stop_token stop_token, std::chrono::steady_clock::time_point start,
                                              double max_time_seconds)
{
    if (stop_token.stop_requested())
    {
        return "canceled";
    }
    if (TimeExpired(start, max_time_seconds))
    {
        return "time_budget";
    }
    return std::nullopt;
}

struct BlockSelection
{
    std::vector<std::size_t> active_blocks;
    std::optional<std::size_t> active_block;
    std::size_t clock = 0;
};

std::size_t CheckedProduct(std::size_t left, std::size_t right, const char* description);

BlockSelection SelectBlocks(const fdn_optimization::BlockSPSAParameters& parameters, std::size_t block_count,
                            std::uint32_t seed, std::size_t update, std::span<const std::size_t> block_visits,
                            std::vector<std::size_t>& shuffled_blocks, std::size_t& shuffled_position)
{
    BlockSelection selection;
    selection.clock = update;
    if (parameters.mode == fdn_optimization::BlockSPSAMode::SnapshotSweepAll)
    {
        selection.active_blocks.resize(block_count);
        std::iota(selection.active_blocks.begin(), selection.active_blocks.end(), 0);
        return selection;
    }

    std::size_t block_index = 0;
    if (parameters.random_schedule == fdn_optimization::RandomBlockSchedule::ShuffledSweep)
    {
        if (shuffled_position >= shuffled_blocks.size())
        {
            shuffled_blocks = ShuffledBlocks(seed, update / block_count, block_count);
            shuffled_position = 0;
        }
        block_index = shuffled_blocks[shuffled_position++];
    }
    else
    {
        block_index = UniformIndex(seed, update, 0, block_count);
    }

    selection.active_blocks.push_back(block_index);
    selection.active_block = block_index;
    selection.clock = block_visits[block_index];
    return selection;
}

std::uint64_t UpdateCost(const BlockSelection& selection, std::size_t directions_per_block, bool evaluates_candidate)
{
    const std::size_t pair_count =
        CheckedProduct(selection.active_blocks.size(), directions_per_block, "BlockSPSA pair count overflow.");
    const std::size_t probe_count = CheckedProduct(pair_count, 2, "BlockSPSA probe count overflow.");
    if (probe_count == std::numeric_limits<std::size_t>::max())
    {
        throw std::overflow_error("BlockSPSA update cost overflow.");
    }
    return static_cast<std::uint64_t>(probe_count) + (evaluates_candidate ? 1U : 0U);
}

bool FitsEvaluationBudget(std::uint64_t evaluation_count, std::uint64_t update_cost,
                          std::uint64_t max_objective_evaluations)
{
    if (max_objective_evaluations == 0)
    {
        return true;
    }
    if (evaluation_count >= max_objective_evaluations)
    {
        return false;
    }
    return update_cost <= max_objective_evaluations - evaluation_count;
}

void ValidateScheduledGains(double learning_rate, double perturbation)
{
    if (!std::isfinite(learning_rate) || learning_rate <= 0.0 || !std::isfinite(perturbation) || perturbation <= 0.0)
    {
        // A degenerate schedule is a configuration error, so it must not be treated as divergence.
        throw std::invalid_argument("BlockSPSA gain schedule produced a nonpositive or non-finite value.");
    }
}

struct AppliedUpdate
{
    arma::mat candidate;
    double scale = 1.0;
};

AppliedUpdate ApplyUpdate(const arma::mat& current_parameters, const arma::mat& gradient,
                          std::span<const double> block_learning_rates, std::span<const std::size_t> active_blocks,
                          std::span<const fdn_optimization::ParameterBlock> blocks, double max_step_norm)
{
    arma::mat step(1, current_parameters.n_cols, arma::fill::zeros);
    for (const auto block_index : active_blocks)
    {
        const double learning_rate = block_learning_rates[block_index];
        for (const auto coordinate : blocks[block_index].coordinates)
        {
            step(0, coordinate) = -learning_rate * gradient(0, coordinate);
        }
    }

    double scale = 1.0;
    if (max_step_norm > 0.0)
    {
        const double norm = arma::norm(step, 2);
        if (std::isfinite(norm) && norm > max_step_norm)
        {
            scale = max_step_norm / norm;
            step *= scale;
        }
    }

    arma::mat candidate = current_parameters + step;
    if (!candidate.is_finite())
    {
        throw NonFiniteObjectiveError("BlockSPSA update produced non-finite coordinates.");
    }
    return {.candidate = std::move(candidate), .scale = scale};
}

struct ProbeTask
{
    std::size_t block = 0;
    std::size_t replication = 0;
    std::vector<double> signs;
};

std::size_t CheckedProduct(std::size_t left, std::size_t right, const char* description)
{
    if (right != 0 && left > std::numeric_limits<std::size_t>::max() / right)
    {
        throw std::overflow_error(description);
    }
    return left * right;
}

std::vector<ProbeTask> BuildTasks(std::span<const fdn_optimization::ParameterBlock> blocks,
                                  std::span<const std::size_t> active_blocks, std::size_t directions_per_block,
                                  std::uint32_t seed, std::size_t update)
{
    std::vector<ProbeTask> tasks;
    tasks.reserve(CheckedProduct(active_blocks.size(), directions_per_block, "BlockSPSA task count overflow."));
    for (const auto block_index : active_blocks)
    {
        for (std::size_t replication = 0; replication < directions_per_block; ++replication)
        {
            std::vector<double> signs;
            signs.reserve(blocks[block_index].coordinates.size());
            for (const auto coordinate : blocks[block_index].coordinates)
            {
                signs.push_back(Rademacher(seed, update, block_index, replication, coordinate));
            }
            tasks.push_back({.block = block_index, .replication = replication, .signs = std::move(signs)});
        }
    }
    return tasks;
}

struct ProbeOutcome
{
    double plus_loss = 0.0;
    double minus_loss = 0.0;
    double difference = 0.0;
};

struct GradientEstimate
{
    arma::mat gradient;
    std::vector<ProbeOutcome> outcomes;
};

GradientEstimate EstimateGradient(const fdn_optimization::BlockSPSAOptimizer::ObjectiveFunction& objective,
                                  const arma::mat& base, std::span<const ProbeTask> tasks,
                                  std::span<const fdn_optimization::ParameterBlock> blocks,
                                  std::span<const double> block_perturbations, std::size_t directions_per_block,
                                  std::uint32_t probe_threads)
{
    std::vector<ProbeOutcome> outcomes(tasks.size());
    std::vector<std::exception_ptr> exceptions(tasks.size());
    const int thread_count = static_cast<int>(
        std::max<std::uint32_t>(1, std::min<std::uint32_t>(probe_threads, static_cast<std::uint32_t>(tasks.size()))));

#pragma omp parallel for schedule(static) num_threads(thread_count)
    for (std::ptrdiff_t task_index = 0; task_index < static_cast<std::ptrdiff_t>(tasks.size()); ++task_index)
    {
        try
        {
            const auto& task = tasks[static_cast<std::size_t>(task_index)];
            arma::mat plus = base;
            arma::mat minus = base;
            const auto& block = blocks[task.block];
            const double perturbation = block_perturbations[task.block];
            for (std::size_t local_index = 0; local_index < block.coordinates.size(); ++local_index)
            {
                const auto coordinate = block.coordinates[local_index];
                const double offset = perturbation * task.signs[local_index];
                plus(0, coordinate) += offset;
                minus(0, coordinate) -= offset;
            }
            const auto plus_evaluation = objective(plus, false);
            const auto minus_evaluation = objective(minus, false);
            if (!std::isfinite(plus_evaluation.total) || !std::isfinite(minus_evaluation.total))
            {
                throw NonFiniteObjectiveError("BlockSPSA probe objective is not finite.");
            }
            outcomes[static_cast<std::size_t>(task_index)] = {
                .plus_loss = plus_evaluation.total,
                .minus_loss = minus_evaluation.total,
                .difference = (plus_evaluation.total - minus_evaluation.total) / (2.0 * perturbation),
            };
        }
        catch (...)
        {
            exceptions[static_cast<std::size_t>(task_index)] = std::current_exception();
        }
    }

    for (const auto& exception : exceptions)
    {
        if (exception)
        {
            std::rethrow_exception(exception);
        }
    }

    arma::mat gradient(1, base.n_cols, arma::fill::zeros);
    for (std::size_t task_index = 0; task_index < tasks.size(); ++task_index)
    {
        const auto& task = tasks[task_index];
        const auto& block = blocks[task.block];
        for (std::size_t local_index = 0; local_index < block.coordinates.size(); ++local_index)
        {
            gradient(0, block.coordinates[local_index]) +=
                outcomes[task_index].difference * task.signs[local_index] / static_cast<double>(directions_per_block);
        }
    }
    return {.gradient = std::move(gradient), .outcomes = std::move(outcomes)};
}

// Derives a stall window large enough that ordinary block-coordinate progress is not mistaken for
// convergence. A per-update absolute change test terminated random-one runs after a few updates,
// and a short window still cost random-one roughly 4% median loss at FDN order 8.
std::size_t DefaultStallWindow(std::size_t block_count)
{
    return std::max<std::size_t>(256, 32 * block_count);
}

// Aggregates the raw probe results into one diagnostic record per active block.
std::vector<fdn_optimization::BlockSPSAProbe> SummarizeProbes(
    std::span<const ProbeTask> tasks, std::span<const ProbeOutcome> outcomes,
    std::span<const fdn_optimization::ParameterBlock> blocks, const BlockSelection& selection,
    const arma::mat& gradient, std::span<const double> block_learning_rates,
    std::span<const double> block_perturbations, std::size_t directions_per_block, double update_scale)
{
    std::vector<fdn_optimization::BlockSPSAProbe> probes;
    probes.reserve(selection.active_blocks.size());
    const double replications = static_cast<double>(directions_per_block);

    for (const auto block_index : selection.active_blocks)
    {
        fdn_optimization::BlockSPSAProbe probe;
        probe.block = block_index;
        probe.block_size = blocks[block_index].coordinates.size();
        // Reports the clock the gains above were computed from, not the post-update visit count.
        probe.visit = selection.clock;
        probe.learning_rate = block_learning_rates[block_index];
        probe.perturbation = block_perturbations[block_index];

        for (std::size_t task_index = 0; task_index < tasks.size(); ++task_index)
        {
            if (tasks[task_index].block != block_index)
            {
                continue;
            }
            probe.probe_plus += outcomes[task_index].plus_loss / replications;
            probe.probe_minus += outcomes[task_index].minus_loss / replications;
            const double paired = outcomes[task_index].plus_loss - outcomes[task_index].minus_loss;
            probe.paired_difference += paired / replications;
            probe.absolute_paired_difference += std::abs(paired) / replications;
        }

        double squared_norm = 0.0;
        for (const auto coordinate : blocks[block_index].coordinates)
        {
            squared_norm += gradient(0, coordinate) * gradient(0, coordinate);
        }
        probe.gradient_norm = std::sqrt(squared_norm);
        probe.step_norm = update_scale * probe.learning_rate * probe.gradient_norm;
        probes.push_back(probe);
    }
    return probes;
}

} // namespace

namespace fdn_optimization
{

BlockSPSAOptimizer::BlockSPSAOptimizer(BlockSPSAParameters parameters)
    : parameters_(std::move(parameters))
{
}

BlockSPSAResult BlockSPSAOptimizer::Optimize(ObjectiveFunction objective, arma::mat initial_parameters,
                                             ObjectiveEvaluation initial_evaluation,
                                             std::span<const ParameterBlock> blocks, std::uint32_t seed,
                                             std::uint32_t probe_threads, std::uint64_t max_objective_evaluations,
                                             double max_time_seconds, std::stop_token stop_token,
                                             StepCallback step_callback) const
{
    ValidateOptimizationInputs(parameters_, blocks, initial_parameters, objective, initial_evaluation);

    std::atomic<std::uint64_t> evaluation_count = 1;
    const ObjectiveFunction counted_objective = [&objective, &evaluation_count](const arma::mat& coordinates,
                                                                                bool publish_components) {
        auto evaluation = objective(coordinates, publish_components);
        evaluation_count.fetch_add(1, std::memory_order_relaxed);
        return evaluation;
    };
    const auto start = std::chrono::steady_clock::now();
    // The historical fallback of 0.1 * max_iterations produced A = 10000, which is unrelated to
    // the few hundred accepted updates a finite evaluation budget actually allows.
    constexpr double kDefaultStabilityConstant = 40.0;
    const double stability = parameters_.stability_constant.value_or(kDefaultStabilityConstant);
    if (!std::isfinite(stability) || stability < 0.0)
    {
        throw std::invalid_argument("BlockSPSA stability constant must be finite and nonnegative.");
    }

    arma::mat current_parameters = std::move(initial_parameters);
    ObjectiveEvaluation current_evaluation = std::move(initial_evaluation);
    arma::mat best_parameters = current_parameters;
    ObjectiveEvaluation best_evaluation = current_evaluation;
    std::vector<std::size_t> block_visits(blocks.size(), 0);
    std::vector<std::size_t> shuffled_blocks;
    std::size_t shuffled_position = blocks.size();
    const auto block_scales = ResolveBlockScales(parameters_, blocks);
    std::vector<double> block_learning_rates(blocks.size(), 0.0);
    std::vector<double> block_perturbations(blocks.size(), 0.0);
    // Bounds the window so that a large or overflowing request cannot allocate wildly or wrap.
    const std::size_t stall_window =
        std::min(parameters_.stall_window.value_or(DefaultStallWindow(blocks.size())), parameters_.max_iterations);
    StallMonitor stall_monitor(stall_window, parameters_.tolerance, best_evaluation.total);

    BlockSPSAResult result;
    result.block_count = blocks.size();
    result.stability_constant = stability;
    result.stall_window = stall_monitor.Enabled() ? stall_window : 0;
    result.termination_reason = "max_iterations";

    for (std::size_t update = 0; update < parameters_.max_iterations; ++update)
    {
        if (const auto reason = InterruptionReason(stop_token, start, max_time_seconds))
        {
            result.termination_reason = *reason;
            break;
        }

        // A non-finite probe, candidate, or update must not discard the best point found so far.
        // Configuration errors keep propagating so that callers still see them.
        try
        {
            const auto selection = SelectBlocks(parameters_, blocks.size(), seed, update, block_visits, shuffled_blocks,
                                                shuffled_position);
            const bool evaluates_candidate = (update + 1) % parameters_.accepted_evaluation_interval == 0;
            const auto update_cost = UpdateCost(selection, parameters_.directions_per_block, evaluates_candidate);
            if (!FitsEvaluationBudget(evaluation_count.load(std::memory_order_relaxed), update_cost,
                                      max_objective_evaluations))
            {
                result.termination_reason = "evaluation_budget";
                break;
            }

            for (const auto block_index : selection.active_blocks)
            {
                block_learning_rates[block_index] = Gain(parameters_.step_size * block_scales[block_index].a_scale,
                                                         parameters_.alpha, stability, selection.clock);
                block_perturbations[block_index] =
                    Gain(parameters_.evaluation_step_size * block_scales[block_index].c_scale, parameters_.gamma, 0.0,
                         selection.clock);
                ValidateScheduledGains(block_learning_rates[block_index], block_perturbations[block_index]);
            }

            const auto tasks =
                BuildTasks(blocks, selection.active_blocks, parameters_.directions_per_block, seed, update);
            const auto estimate =
                EstimateGradient(counted_objective, current_parameters, tasks, blocks, block_perturbations,
                                 parameters_.directions_per_block, probe_threads);
            const arma::mat& gradient = estimate.gradient;

            if (const auto reason = InterruptionReason(stop_token, start, max_time_seconds))
            {
                result.termination_reason = *reason;
                break;
            }

            auto applied_update = ApplyUpdate(current_parameters, gradient, block_learning_rates,
                                              selection.active_blocks, blocks, parameters_.max_step_norm);
            arma::mat candidate = std::move(applied_update.candidate);
            const double step_norm = arma::norm(candidate - current_parameters, 2);
            bool improved_best = false;
            std::optional<std::string> post_evaluation_interruption;
            if (evaluates_candidate)
            {
                auto candidate_evaluation = counted_objective(candidate, true);
                if (!std::isfinite(candidate_evaluation.total))
                {
                    throw NonFiniteObjectiveError("BlockSPSA candidate objective is not finite.");
                }
                post_evaluation_interruption = InterruptionReason(stop_token, start, max_time_seconds);
                current_evaluation = std::move(candidate_evaluation);
                if (current_evaluation.total < best_evaluation.total)
                {
                    best_evaluation = current_evaluation;
                    improved_best = true;
                }
            }
            current_parameters = std::move(candidate);
            if (improved_best)
            {
                best_parameters = current_parameters;
            }
            if (selection.active_block)
            {
                ++block_visits[*selection.active_block];
            }

            ++result.accepted_updates;
            if (step_callback)
            {
                step_callback({.step = update,
                               .evaluation = current_evaluation,
                               .best_loss = best_evaluation.total,
                               .learning_rate = block_learning_rates[selection.active_blocks.front()],
                               .gradient_norm = arma::norm(gradient, 2),
                               .objective_evaluations = evaluation_count.load(std::memory_order_relaxed),
                               .elapsed_time = std::chrono::steady_clock::now() - start,
                               .active_block = selection.active_block,
                               .block_visit = selection.active_block
                                                  ? std::optional<std::size_t>(block_visits[*selection.active_block])
                                                  : std::nullopt,
                               .perturbation = block_perturbations[selection.active_blocks.front()],
                               .directions_averaged = parameters_.directions_per_block,
                               .evaluated = evaluates_candidate,
                               .improved_best = improved_best,
                               .step_norm = step_norm,
                               .block_probes = SummarizeProbes(
                                   tasks, estimate.outcomes, blocks, selection, gradient, block_learning_rates,
                                   block_perturbations, parameters_.directions_per_block, applied_update.scale)});
            }

            if (post_evaluation_interruption)
            {
                result.termination_reason = *post_evaluation_interruption;
                break;
            }

            if (evaluates_candidate && stall_monitor.Stalled(best_evaluation.total))
            {
                result.termination_reason = "converged";
                break;
            }
        }
        catch (const NonFiniteObjectiveError&)
        {
            result.termination_reason = "non_finite_objective";
            break;
        }
    }

    result.parameters = std::move(best_parameters);
    result.evaluation = std::move(best_evaluation);
    return result;
}

BlockSPSAResult BlockSPSAOptimizer::Optimize(FDNModel& model, arma::mat initial_parameters,
                                             ObjectiveEvaluation initial_evaluation,
                                             std::span<const ParameterBlock> blocks, std::uint32_t seed,
                                             std::uint32_t probe_threads, std::uint64_t max_objective_evaluations,
                                             double max_time_seconds, std::stop_token stop_token,
                                             StepCallback step_callback) const
{
    const auto objective = [&model](const arma::mat& coordinates, bool publish_components) {
        return model.EvaluateDetailed(coordinates, publish_components);
    };
    auto result =
        Optimize(objective, std::move(initial_parameters), std::move(initial_evaluation), blocks, seed, probe_threads,
                 max_objective_evaluations, max_time_seconds, stop_token, std::move(step_callback));
    LossRegistry::Instance().RegisterLoss(result.evaluation.components);
    return result;
}

} // namespace fdn_optimization
