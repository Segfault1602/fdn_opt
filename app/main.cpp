#include "optimizer.h"

#include "utils.h"

#include <sffdn/sffdn.h>

#include <audio_utils/audio_analysis.h>
#include <audio_utils/audio_file_manager.h>

#include "quill/Logger.h"
#include "quill/sinks/ConsoleSink.h"
#include <CLI/CLI.hpp>
#include <armadillo>
#include <quill/Backend.h>
#include <quill/Frontend.h>
#include <quill/LogMacros.h>

#include <omp.h>

#include <chrono>
#include <filesystem>
#include <format>
#include <iostream>
#include <ostream>
#include <random>
#include <thread>
#include <vector>

template <typename T>
void SetOptimizerParams(const T& params, fdn_optimization::OptimizationAlgoParams& optim_params)
{
    optim_params = params;
}

const std::vector<float> kInitialInputGains = {0.021565, -0.10697,  0.271459, -0.507918,
                                               0.696453, -0.366612, 0.161309, -0.10464};
const std::vector<float> kInitialOutputGains = {0.467987, -0.403734, 0.303612,  -0.192053,
                                                0.166956, -0.442626, -0.503543, -0.107584};

const std::vector<float> kOptimizedMatrix = {
    0.700161,   -0.0379672, -0.351466,  0.546243,  0.248043,   -0.0527009, 0.0695207,  -0.13148,   0.359406,  0.841295,
    0.106511,   -0.182104,  -0.327287,  0.0915586, -0.0347661, -0.0428309, -0.115603,  -0.0647493, 0.500109,  0.627607,
    -0.389675,  0.0714991,  -0.332708,  -0.266066, -0.381827,  0.41174,    -0.0327591, 0.409852,   0.463725,  0.408213,
    0.0292857,  0.364815,   0.0419472,  0.0523951, 0.066017,   -0.268326,  0.522374,   0.146676,   -0.623391, -0.485936,
    0.14331,    -0.14488,   0.365672,   -0.113971, 0.111146,   0.571918,   0.588995,   -0.35413,   -0.370269, 0.304733,
    -0.0730987, 0.144298,   0.160557,   -0.517478, 0.376585,   -0.555424,  -0.249098,  -0.02145,   -0.68595,  0.027881,
    -0.391745,  0.447998,   -0.0711281, -0.327048};

sfFDN::FDNConfig2 CreateInitialFDNConfig(uint32_t fdn_order, bool randomize = false, bool random_delays = false)
{
    sfFDN::FDNConfig2 initial_fdn_config{};
    initial_fdn_config.fdn_size = fdn_order;
    initial_fdn_config.transposed = false;
    initial_fdn_config.direct_gain = 0.0f;
    initial_fdn_config.sample_rate = kSampleRate;
    initial_fdn_config.block_size = 128;
    initial_fdn_config.input_block_config.parallel_gains_config = {sfFDN::ParallelGainsMode::Split,
                                                                   std::vector<float>(fdn_order, 0.5f)};
    initial_fdn_config.output_block_config.parallel_gains_config = {sfFDN::ParallelGainsMode::Merge,
                                                                    std::vector<float>(fdn_order, 0.5f)};

    if (fdn_order == 4)
    {
        initial_fdn_config.delay_bank_config.delays = {1499, 1889, 2381, 2999};
    }
    else if (fdn_order == 6)
    {
        initial_fdn_config.delay_bank_config.delays = {997, 1153, 1327, 1559, 1801, 2099};
    }
    else if (fdn_order == 8)
    {
        initial_fdn_config.delay_bank_config.delays = {809, 877, 937, 1049, 1151, 1249, 1373, 1499};
    }
    else
    {
        initial_fdn_config.delay_bank_config.delays =
            sfFDN::GetDelayLengths(fdn_order, 512, 3000, sfFDN::DelayLengthType::Uniform, 42);
    }

    if (random_delays)
    {
        std::cout << "Using random delays..." << std::endl;
        initial_fdn_config.delay_bank_config.delays =
            sfFDN::GetDelayLengths(fdn_order, 512, 3000, sfFDN::DelayLengthType::Uniform);
    }

    initial_fdn_config.delay_bank_config.block_size = 128;
    initial_fdn_config.delay_bank_config.interpolation_type = sfFDN::DelayInterpolationType::None;

    // initial_fdn_config.loop_filter_configs = sfFDN::ProportionalAttenuationConfig{10.f};
    sfFDN::AttenuationFilterBankOptions loop_filters;
    for (uint32_t i = 0; i < fdn_order; ++i)
    {
        sfFDN::ProportionalAttenuationOptions c{
            .t60 = 1.f, .delay = initial_fdn_config.delay_bank_config.delays[i], .sample_rate = kSampleRate};
        loop_filters.filter_configs.push_back(c);
    }
    initial_fdn_config.loop_filter_configs = {loop_filters};

    if (randomize)
    {
        std::cout << "Using random initial parameters..." << std::endl;
        // arma::arma_rng::set_seed_random();
        arma::fvec input_gains(fdn_order, arma::fill::randn);
        arma::fvec output_gains(fdn_order, arma::fill::randn);

        input_gains /= arma::norm(input_gains, 2);
        output_gains /= arma::norm(output_gains, 2);

        for (uint32_t i = 0; i < fdn_order; ++i)
        {
            initial_fdn_config.input_block_config.parallel_gains_config.gains[i] = input_gains(i);
            initial_fdn_config.output_block_config.parallel_gains_config.gains[i] = output_gains(i);
        }

        initial_fdn_config.feedback_matrix_config =
            sfFDN::ScalarFeedbackMatrixOptions{.matrix_size = fdn_order, .type = sfFDN::ScalarMatrixType::Random};
    }
    else
    {
        if (fdn_order == 8 || fdn_order == 4)
        {
            initial_fdn_config.feedback_matrix_config =
                sfFDN::ScalarFeedbackMatrixOptions{.matrix_size = fdn_order, .type = sfFDN::ScalarMatrixType::Hadamard};
        }
        else
        {
            initial_fdn_config.feedback_matrix_config = sfFDN::ScalarFeedbackMatrixOptions{
                .matrix_size = fdn_order, .type = sfFDN::ScalarMatrixType::Householder};
        }
    }
    return initial_fdn_config;
}

fdn_optimization::OptimizationResult DoOptimization(quill::Logger* logger, fdn_optimization::OptimizationInfo& opt_info)
{
    fdn_optimization::FDNOptimizer optimizer(logger);

    optimizer.StartOptimization(opt_info);

    while (optimizer.GetStatus() != fdn_optimization::OptimizationStatus::Running)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    while (optimizer.GetStatus() != fdn_optimization::OptimizationStatus::Completed)
    {
        std::this_thread::sleep_for(std::chrono::seconds(1));

        auto progress = optimizer.GetProgress();
        float last_loss = 0.0f;
        if (!progress.loss_history.empty() && !progress.loss_history[0].empty())
        {
            last_loss = static_cast<float>(progress.loss_history[0].back());
        }
        LOG_INFO(logger, "Elapsed Time: {:.2f} s, Evaluations: {}, Last Loss: {:.6f}", progress.elapsed_time.count(),
                 progress.evaluation_count, last_loss);
    }

    auto result = optimizer.GetResult();
    return result;
}

fdn_optimization::OptimizationResult OptimizeColorless(quill::Logger* logger,
                                                       const sfFDN::FDNConfig2& initial_fdn_config,
                                                       const fdn_optimization::OptimizationAlgoParams& optimizer_params,
                                                       const std::tuple<double, double, double>& loss_weights,
                                                       bool verbose)
{

    std::vector params_to_optimize = {fdn_optimization::OptimizationParamType::Gains,
                                      fdn_optimization::OptimizationParamType::Matrix};

    fdn_optimization::OptimizationInfo opt_info{.parameters_to_optimize = params_to_optimize,
                                                .initial_fdn_config = initial_fdn_config,
                                                .ir_size = kSampleRate,
                                                .gradient_method = fdn_optimization::GradientMethod::CentralDifferences,
                                                .spectral_flatness_weight = std::get<0>(loss_weights),
                                                .sparsity_weight = std::get<1>(loss_weights),
                                                .power_envelope_weight = std::get<2>(loss_weights),
                                                .target_rir = {},
                                                .early_fir = {},
                                                .optimizer_params = optimizer_params};

    fdn_optimization::FDNOptimizer optimizer(logger, verbose);

    optimizer.StartOptimization(opt_info);

    while (optimizer.GetStatus() != fdn_optimization::OptimizationStatus::Running)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    while (optimizer.GetStatus() != fdn_optimization::OptimizationStatus::Completed)
    {
        std::this_thread::sleep_for(std::chrono::seconds(1));

        auto progress = optimizer.GetProgress();
        float last_loss = 0.0f;
        if (!progress.loss_history.empty() && !progress.loss_history[0].empty())
        {
            last_loss = static_cast<float>(progress.loss_history[0].back());
        }
        LOG_DEBUG(logger, "Elapsed Time: {:.2f} s, Evaluations: {}, Last Loss: {:.6f}", progress.elapsed_time.count(),
                  progress.evaluation_count, last_loss);
    }

    auto result = optimizer.GetResult();
    return result;
}

fdn_optimization::OptimizationResult OptimizeSpectrum(
    quill::Logger* logger, const sfFDN::FDNConfig2& initial_fdn_config, const fdn_optimization::OptimizationAlgoParams&,
    const std::vector<float>& target_rir, const std::vector<float>& early_fir,
    const std::tuple<double, double, double>& loss_weights, bool verbose)
{
    fdn_optimization::AdamParameters opt_params{.step_size = 0.1,
                                                .learning_rate_decay = 1.0,
                                                .decay_step_size = 1,
                                                .epoch_restarts = 180,
                                                .max_restarts = 0,
                                                .tolerance = 1e-3,
                                                .gradient_delta = 1e-1};

    std::vector params_to_optimize = {fdn_optimization::OptimizationParamType::AttenuationFilters,
                                      fdn_optimization::OptimizationParamType::TonecorrectionFilters,
                                      fdn_optimization::OptimizationParamType::OverallGain};

    fdn_optimization::OptimizationInfo opt_info{.parameters_to_optimize = params_to_optimize,
                                                .initial_fdn_config = initial_fdn_config,
                                                .ir_size = static_cast<uint32_t>(target_rir.size()),
                                                .gradient_method = fdn_optimization::GradientMethod::CentralDifferences,
                                                .edc_weight = std::get<0>(loss_weights),
                                                .mel_edr_weight = std::get<1>(loss_weights),
                                                .weighted_edr_weight = std::get<2>(loss_weights),
                                                .target_rir = target_rir,
                                                .early_fir = early_fir,
                                                .optimizer_params = opt_params};

    fdn_optimization::FDNOptimizer optimizer(logger, verbose);

    optimizer.StartOptimization(opt_info);

    while (optimizer.GetStatus() != fdn_optimization::OptimizationStatus::Running)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    while (optimizer.GetStatus() != fdn_optimization::OptimizationStatus::Completed)
    {
        std::this_thread::sleep_for(std::chrono::seconds(1));

        auto progress = optimizer.GetProgress();
        float last_loss = 0.0f;
        if (!progress.loss_history.empty() && !progress.loss_history[0].empty())
        {
            last_loss = static_cast<float>(progress.loss_history[0].back());
        }
        LOG_DEBUG(logger, "Elapsed Time: {:.2f} s, Evaluations: {}, Last Loss: {:.6f}", progress.elapsed_time.count(),
                  progress.evaluation_count, last_loss);
    }

    auto result = optimizer.GetResult();
    return result;
}

void RenderAudio(const sfFDN::FDNConfig2& fdn_config, const std::string& input_filename,
                 const std::filesystem::path& output_dir, quill::Logger* logger);

int main(int argc, char** argv)
{
    omp_set_num_threads(1);
    quill::Backend::start();
    quill::Logger* logger = quill::Frontend::create_or_get_logger(
        "root", quill::Frontend::create_or_get_sink<quill::ConsoleSink>("sink_id_1"));

    CLI::App app{"FDN Optimization Tool"};

    std::string ir_filename;
    app.add_option("-i,--ir", ir_filename, "Path to target RIR WAV file")->check(CLI::ExistingFile);

    std::string early_fir_path;
    app.add_option("--early_fir_path", early_fir_path, "Path to early reflection FIR WAV file")
        ->check(CLI::ExistingFile);

    uint32_t fdn_order = 6;
    app.add_option("-n,--num_channels", fdn_order, "FDN order (number of channels), e.g., 4, 6, 8")->default_val(6);

    bool colorless_only = false;
    app.add_flag("-c,--colorless_only", colorless_only, "Only perform colorless optimization");

    bool save_output = true;
    app.add_flag("-s,--save_output", save_output, "Save optimization results to output directory");

    bool verbose = false;
    app.add_flag("-v,--verbose", verbose, "Enable verbose logging");

    double spectral_flatness_weight = 1.0;
    app.add_option("--spectral_flatness_weight", spectral_flatness_weight, "Weight for spectral flatness loss term")
        ->default_val(1.0);
    double sparsity_weight = 1.0;
    app.add_option("--sparsity_weight", sparsity_weight, "Weight for sparsity loss term")->default_val(1.0);
    double power_envelope_weight = 0.0;
    app.add_option("--power_envelope_weight", power_envelope_weight, "Weight for power envelope loss term")
        ->default_val(0.0);

    double edc_weight = 1.0;
    app.add_option("--edc_weight", edc_weight, "Weight for EDC loss term")->default_val(1.0);
    double mel_edr_weight = 1.0;
    app.add_option("--mel_edr_weight", mel_edr_weight, "Weight for Mel EDR loss term")->default_val(1.0);
    double weighted_edr_weight = 0.0;
    app.add_option("--weighted_edr_weight", weighted_edr_weight, "Weight for Weighted EDR loss term")->default_val(0.0);

    bool randomize_initial = false;
    app.add_flag("--randomize_initial_params", randomize_initial,
                 "Randomize initial FDN configuration instead of using Householder matrix");

    bool random_delays = false;
    app.add_flag("--random_delays", random_delays, "Use random delay lengths instead of predefined sets");

    std::string output_dir = "optim_output";
    app.add_option("-o,--output_dir", output_dir, "Output directory for optimization results    ")
        ->capture_default_str();

    fdn_optimization::OptimizationAlgoParams optimizer_params;

    // ADAM
    CLI::App* adam_sub = app.add_subcommand("Adam", "Use Adam optimization algorithm");
    fdn_optimization::AdamParameters adam_params;
    adam_sub->add_option("--step_size", adam_params.step_size, "Step size for Adam optimizer");
    adam_sub->add_option("--beta1", adam_params.beta1, "Beta1 parameter for Adam optimizer");
    adam_sub->add_option("--beta2", adam_params.beta2, "Beta2 parameter for Adam optimizer");
    adam_sub->add_option("--tolerance", adam_params.tolerance, "Tolerance for Adam optimizer");
    adam_sub->add_option("--gradient_delta", adam_params.gradient_delta, "Gradient delta for Adam optimizer");
    adam_sub->callback([&]() { SetOptimizerParams(adam_params, optimizer_params); });

    // SPSA
    CLI::App* spsa_sub = app.add_subcommand("SPSA", "Use SPSA optimization algorithm");
    fdn_optimization::SPSAParameters spsa_params;
    spsa_sub->add_option("--alpha", spsa_params.alpha, "Alpha parameter for SPSA optimizer");
    spsa_sub->add_option("--gamma", spsa_params.gamma, "Gamma parameter for SPSA optimizer");
    spsa_sub->add_option("--step_size", spsa_params.step_size, "Step size for SPSA optimizer");
    spsa_sub->add_option("--evaluation_step_size", spsa_params.evaluationStepSize,
                         "Evaluation step size for SPSA optimizer");
    spsa_sub->add_option("--max_iterations", spsa_params.max_iterations, "Maximum iterations for SPSA optimizer");
    spsa_sub->add_option("--tolerance", spsa_params.tolerance, "Tolerance for SPSA optimizer");
    spsa_sub->callback([&]() { SetOptimizerParams(spsa_params, optimizer_params); });

    // Simulated Annealing
    CLI::App* sa_sub = app.add_subcommand("SimulatedAnnealing", "Use Simulated Annealing optimization algorithm");
    fdn_optimization::SimulatedAnnealingParameters sa_params;
    sa_sub->add_option("--max_iterations", sa_params.max_iterations, "Maximum iterations for Simulated Annealing");
    sa_sub->add_option("--initial_temperature", sa_params.initial_temperature,
                       "Initial temperature for Simulated Annealing");
    sa_sub->add_option("--init_moves", sa_params.init_moves, "Initial moves for Simulated Annealing");
    sa_sub->add_option("--move_ctrl_sweep", sa_params.move_ctrl_sweep, "Move control sweep for Simulated Annealing");
    sa_sub->add_option("--max_tolerance_sweep", sa_params.max_tolerance_sweep,
                       "Max tolerance sweep for Simulated Annealing");
    sa_sub->add_option("--max_move_coeff", sa_params.max_move_coef, "Max move coefficient for Simulated Annealing");
    sa_sub->add_option("--init_move_coeff", sa_params.init_move_coef,
                       "Initial move coefficient for Simulated Annealing");
    sa_sub->add_option("--gain", sa_params.gain, "Gain for Simulated Annealing");
    sa_sub->add_option("--tolerance", sa_params.tolerance, "Tolerance for Simulated Annealing");
    sa_sub->callback([&]() { SetOptimizerParams(sa_params, optimizer_params); });

    // CNE
    CLI::App* cne_sub = app.add_subcommand("CNE", "Use CNE optimization algorithm");
    fdn_optimization::CNEParameters cne_params;
    cne_sub->add_option("--population_size", cne_params.population_size, "Population size for CNE optimizer");
    cne_sub->add_option("--max_generations", cne_params.max_generations, "Maximum generations for CNE optimizer");
    cne_sub->add_option("--mutation_probability", cne_params.mutation_probability,
                        "Mutation probability for CNE optimizer");
    cne_sub->add_option("--mutation_size", cne_params.mutation_size, "Mutation size for CNE optimizer");
    cne_sub->add_option("--select_percent", cne_params.select_percent, "Selection percentage for CNE optimizer");
    cne_sub->add_option("--tolerance", cne_params.tolerance, "Tolerance for CNE optimizer");
    cne_sub->callback([&]() { SetOptimizerParams(cne_params, optimizer_params); });

    // Differential Evolution
    CLI::App* de_sub = app.add_subcommand("DifferentialEvolution", "Use Differential Evolution optimization algorithm");
    fdn_optimization::DifferentialEvolutionParameters de_params;
    de_sub->add_option("--population_size", de_params.population_size,
                       "Population size for Differential Evolution optimizer");
    de_sub->add_option("--max_generations", de_params.max_generation,
                       "Maximum generations for Differential Evolution optimizer");
    de_sub->add_option("--crossover_rate", de_params.crossover_rate,
                       "Crossover rate for Differential Evolution optimizer");
    de_sub->add_option("--differential_weight", de_params.differential_weight,
                       "Differential weight for Differential Evolution optimizer");
    de_sub->add_option("--tolerance", de_params.tolerance, "Tolerance for Differential Evolution optimizer");
    de_sub->callback([&]() { SetOptimizerParams(de_params, optimizer_params); });

    // PSO
    CLI::App* pso_sub = app.add_subcommand("PSO", "Use Particle Swarm Optimization algorithm");
    fdn_optimization::PSOParameters pso_params;
    pso_sub->add_option("--num_particles", pso_params.num_particles, "Number of particles for PSO optimizer");
    pso_sub->add_option("--max_iterations", pso_params.max_iterations, "Maximum iterations for PSO optimizer");
    pso_sub->add_option("--horizon_size", pso_params.horizon_size, "Horizon size for PSO optimizer");
    pso_sub->add_option("--exploitation_factor", pso_params.exploitation_factor,
                        "Exploitation factor for PSO optimizer");
    pso_sub->add_option("--exploration_factor", pso_params.exploration_factor, "Exploration factor for PSO optimizer");
    pso_sub->add_option("--tolerance", pso_params.tolerance, "Tolerance for PSO optimizer");
    pso_sub->callback([&]() { SetOptimizerParams(pso_params, optimizer_params); });

    // L-BFGS
    CLI::App* lbfgs_sub = app.add_subcommand("L-BFGS", "Use L-BFGS optimization algorithm");
    fdn_optimization::L_BFGSParameters lbfgs_params;
    lbfgs_sub->add_option("--num_basis", lbfgs_params.num_basis, "Number of basis vectors for L-BFGS optimizer");
    lbfgs_sub->add_option("--max_iterations", lbfgs_params.max_iterations, "Maximum iterations for L-BFGS optimizer");
    lbfgs_sub->add_option("--wolfe", lbfgs_params.wolfe, "Wolfe condition parameter for L-BFGS optimizer");
    lbfgs_sub->add_option("--min_gradient_norm", lbfgs_params.min_gradient_norm,
                          "Minimum gradient norm for L-BFGS optimizer");
    lbfgs_sub->add_option("--factor", lbfgs_params.factor, "Factor for L-BFGS optimizer");
    lbfgs_sub->add_option("--max_line_search", lbfgs_params.max_line_search_trials,
                          "Maximum line search trials for L-BFGS optimizer");
    lbfgs_sub->add_option("--min_step", lbfgs_params.min_step, "Minimum step size for L-BFGS optimizer");
    lbfgs_sub->add_option("--max_step", lbfgs_params.max_step, "Maximum step size for L-BFGS optimizer");
    lbfgs_sub->add_option("--gradient_delta", lbfgs_params.gradient_delta,
                          "Gradient delta for L-BFGS optimizer when optimizing filters");
    lbfgs_sub->callback([&]() { SetOptimizerParams(lbfgs_params, optimizer_params); });

    // Gradient Descent
    CLI::App* gd_sub = app.add_subcommand("GradientDescent", "Use Gradient Descent optimization algorithm");
    fdn_optimization::GradientDescentParameters gd_params;
    gd_sub->add_option("--step_size", gd_params.step_size, "Step size for Gradient Descent optimizer");
    gd_sub->add_option("--max_iterations", gd_params.max_iterations,
                       "Maximum iterations for Gradient Descent optimizer");
    gd_sub->add_option("--tolerance", gd_params.tolerance, "Tolerance for Gradient Descent optimizer");
    gd_sub->add_option("--kappa", gd_params.kappa, "Kappa");
    gd_sub->add_option("--phi", gd_params.phi, "Phi");
    gd_sub->add_option("--momentum", gd_params.momentum, "Momemtum");
    gd_sub->add_option("--min_gain", gd_params.min_gain, "Minimum gain for Gradient Descent optimizer");
    gd_sub
        ->add_option("--gradient_delta", gd_params.gradient_delta,
                     "Gradient delta for Gradient Descent optimizer when optimizing filters")
        ->default_val(1e-2);
    gd_sub->callback([&]() { SetOptimizerParams(gd_params, optimizer_params); });

    // CMAES
    CLI::App* cmaes_sub = app.add_subcommand("CMAES", "Use CMA-ES optimization algorithm");
    fdn_optimization::CMAESParameters cmaes_params;
    cmaes_sub->add_option("--population_size", cmaes_params.population_size, "Population size for CMA-ES optimizer");
    cmaes_sub->add_option("--max_iterations", cmaes_params.max_iterations, "Maximum iterations for CMA-ES optimizer");
    cmaes_sub->add_option("--tolerance", cmaes_params.tolerance, "Tolerance for CMA-ES optimizer");
    cmaes_sub->add_option("--step_size", cmaes_params.step_size, "Step size for CMA-ES optimizer");
    cmaes_sub->callback([&]() { SetOptimizerParams(cmaes_params, optimizer_params); });

    // Random Search
    CLI::App* random_search_sub = app.add_subcommand("RandomSearch", "Use Random Search optimization algorithm");
    fdn_optimization::RandomSearchParameters random_search_params;
    random_search_sub->add_option("--time_limit", random_search_params.time_limit_seconds,
                                  "Time limit in seconds for Random Search optimizer");
    random_search_sub->callback([&]() { SetOptimizerParams(random_search_params, optimizer_params); });

    app.set_config("--config");
    app.allow_config_extras(CLI::config_extras_mode::error);

    app.require_subcommand(1);
    CLI11_PARSE(app, argc, argv);

    if (verbose)
    {
        logger->set_log_level(quill::LogLevel::Debug);
    }

    auto omp_num_threads = omp_get_max_threads();
#ifdef __APPLE__
    omp_set_num_threads(std::min(4, omp_num_threads));
    omp_num_threads = omp_get_max_threads();
#endif
    LOG_INFO(logger, "Using up to {} threads for optimization.", omp_num_threads);

    auto config_filename = app.get_config_ptr()->as<std::string>();
    LOG_INFO(logger, "Using configuration file: {}", config_filename);

    std::string selected_optimizer = app.get_subcommands()[0]->get_name();
    LOG_INFO(logger, "Selected optimization algorithm: {}", selected_optimizer);

    auto now = std::chrono::system_clock::now();
#ifndef __APPLE__
    auto local_now = std::chrono::current_zone()->to_local(std::chrono::floor<std::chrono::seconds>(now));
#else
    auto local_now = std::chrono::floor<std::chrono::seconds>(now);
#endif

    std::string timestamp = std::format("{:%Y%m%d_%H%M%S}", local_now);
    LOG_INFO(logger, "Optimization timestamp: {}", timestamp);

    std::string output_dir_name = timestamp + "_" + selected_optimizer;

    std::filesystem::path optim_subdir;

    if (save_output)
    {
        optim_subdir = std::filesystem::path(output_dir) / output_dir_name;
        std::filesystem::create_directory(optim_subdir);
    }

    auto initial_fdn_config = CreateInitialFDNConfig(fdn_order, randomize_initial, random_delays);

    if (save_output)
    {
        // initial_fdn_config.attenuation_filter_config = sfFDN::ProportionalAttenuationConfig{2.f};
        SaveImpulseResponse(initial_fdn_config, kSampleRate * 2.f, optim_subdir / "initial_ir.wav", logger);
        WriteConfigToFile(initial_fdn_config, optim_subdir / "initial_fdn_config.txt", logger);
    }

    LOG_INFO(logger, "Starting colorless optimization...");
    fdn_optimization::OptimizationResult result;

    result =
        OptimizeColorless(logger, initial_fdn_config, optimizer_params,
                          std::make_tuple(spectral_flatness_weight, sparsity_weight, power_envelope_weight), verbose);

    LOG_INFO(logger, "[Colorless] Final loss: {:.6f}", result.best_loss);
    LOG_INFO(logger, "[Colorless] Elapsed time: {:.4f} s", result.total_time.count());
    LOG_INFO(logger, "[Colorless] Total evaluations: {}", result.total_evaluations);

    if (save_output)
    {
        WriteConfigToFile(result.optimized_fdn_config, optim_subdir / "colorless_fdn_config.txt", logger);
        WriteInfoToFile(result, optimizer_params, optim_subdir / "colorless_fdn_info.txt", logger);

        // result.optimized_fdn_config.attenuation_filter_config = sfFDN::ProportionalAttenuationConfig{2.f};
        SaveImpulseResponse(result.optimized_fdn_config, kSampleRate * 3.f, optim_subdir / "colorless_ir.wav", logger);
        WriteLossHistoryToFile(result.loss_history, result.loss_names, optim_subdir / "colorless_loss_history.txt",
                               logger);
    }

    initial_fdn_config = result.optimized_fdn_config;

    if (colorless_only)
    {
        LOG_INFO(logger, "Colorless-only optimization flag set. Exiting.");
        return 0;
    }

    std::vector<float> target_rir;
    if (!ir_filename.empty())
    {
        int num_channels = 0;
        int sample_rate = kSampleRate;
        if (audio_utils::audio_file::ReadWavFile(ir_filename, target_rir, sample_rate, num_channels))
        {
            LOG_INFO(logger, "Loaded {} with {} samples at {} Hz.", ir_filename, target_rir.size(), sample_rate);
        }
        else
        {
            LOG_ERROR(logger, "Failed to load target RIR.");
            return -1;
        }

        // if (target_rir.size() < kSampleRate)
        // {
        //     std::vector<float> padded_rir(kSampleRate, 0.0f);
        //     std::copy(target_rir.begin(), target_rir.end(), padded_rir.begin());
        //     target_rir = std::move(padded_rir);
        //     LOG_INFO(logger, "Padded target RIR to {} samples.", target_rir.size());
        // }
    }

    std::vector<float> early_fir;
    if (!early_fir_path.empty())
    {
        int num_channels = 0;
        int sample_rate = kSampleRate;
        if (audio_utils::audio_file::ReadWavFile(early_fir_path, early_fir, sample_rate, num_channels))
        {
            LOG_INFO(logger, "Loaded {} with {} samples at {} Hz.", early_fir_path, early_fir.size(), sample_rate);
        }
        else
        {
            LOG_ERROR(logger, "Failed to load early reflection FIR.");
            return -1;
        }
    }

    result = OptimizeSpectrum(logger, initial_fdn_config, optimizer_params, target_rir, early_fir,
                              std::make_tuple(edc_weight, mel_edr_weight, weighted_edr_weight), verbose);
    LOG_INFO(logger, "[Spectrum] Final loss: {:.6f}", result.best_loss);
    LOG_INFO(logger, "[Spectrum] Elapsed time: {:.4f} s", result.total_time.count());
    LOG_INFO(logger, "[Spectrum] Total evaluations: {}", result.total_evaluations);

    if (save_output)
    {
        WriteConfigToFile(result.optimized_fdn_config, optim_subdir / "optimized_fdn_config.txt", logger);
        WriteFilterConfigToFile(result.optimized_fdn_config, optim_subdir / "optimized_filter_config.txt", logger);
        SaveImpulseResponse(result.initial_fdn_config, target_rir.size(), optim_subdir / "spectrum_initial_ir.wav",
                            logger, early_fir);
        SaveImpulseResponse(result.optimized_fdn_config, target_rir.size(), optim_subdir / "spectrum_optimized_ir.wav",
                            logger, early_fir);
        WriteLossHistoryToFile(result.loss_history, result.loss_names, optim_subdir / "spectrum_loss_history.txt",
                               logger);

        std::filesystem::path target_rir_name_path = optim_subdir / "target_rir_name.txt";
        std::ofstream file(target_rir_name_path, std::ios::out);
        if (!file.is_open())
        {
            LOG_ERROR(logger, "Failed to open file {} for writing target RIR name.", target_rir_name_path.string());
            return -1;
        }
        file << ir_filename << std::endl;

        RenderAudio(result.optimized_fdn_config, "./audio/drumloop.wav", optim_subdir, logger);
        RenderAudio(result.optimized_fdn_config, "./audio/saxophone.wav", optim_subdir, logger);
        RenderAudio(result.optimized_fdn_config, "./audio/bleepsandbloops.wav", optim_subdir, logger);
    }

    return 0;
}

void RenderAudio(const sfFDN::FDNConfig2& fdn_config, const std::string& input_filename,
                 const std::filesystem::path& output_dir, quill::Logger* logger)
{
    std::vector<float> audio_file;
    int sample_rate = 0;
    int num_channels = 0;
    audio_utils::audio_file::ReadWavFile(input_filename, audio_file, sample_rate, num_channels);
    if (sample_rate != kSampleRate)
    {
        LOG_ERROR(logger, "Input audio sample rate {} does not match expected sample rate {}.", sample_rate,
                  kSampleRate);
        return;
    }
    if (num_channels != 1)
    {
        LOG_ERROR(logger, "Input audio has {} channels, but only mono audio is supported.", num_channels);
        return;
    }

    auto fdn = sfFDN::CreateFDNFromConfig2(fdn_config);
    fdn->SetDirectGain(0.0f);

    std::vector<float> output_audio(audio_file.size(), 0.0f);
    sfFDN::AudioBuffer input_buffer(audio_file);
    sfFDN::AudioBuffer output_buffer(output_audio);

    fdn->Process(input_buffer, output_buffer);

    std::filesystem::path output_path =
        output_dir / (std::filesystem::path(input_filename).stem().string() + "_wet.wav");
    audio_utils::audio_file::WriteWavFile(output_path.string(), output_audio, kSampleRate);
}