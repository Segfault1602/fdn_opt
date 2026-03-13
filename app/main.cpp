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

#include <chrono>
#include <filesystem>
#include <format>
#include <iostream>
#include <ostream>
#include <random>
#include <thread>
#include <vector>

constexpr uint32_t kFDNOrder = 8;

sfFDN::FDNConfig CreateInitialFDNConfig(uint32_t fdn_order, bool randomize = false)
{
    sfFDN::FDNConfig initial_fdn_config{};
    initial_fdn_config.N = fdn_order;
    initial_fdn_config.transposed = false;
    initial_fdn_config.input_gains = std::vector<float>(fdn_order, 0.5f);
    initial_fdn_config.output_gains = std::vector<float>(fdn_order, 0.5f);
    initial_fdn_config.attenuation_t60s = {10.f};
    initial_fdn_config.tc_frequencies = {31.25, 62.5, 125, 250, 500, 1000, 2000, 4000, 8000, 16000};

    if (fdn_order == 4)
    {
        initial_fdn_config.delays = {1499, 1889, 2381, 2999};
    }
    else if (fdn_order == 6)
    {
        initial_fdn_config.delays = {997, 1153, 1327, 1559, 1801, 2099};
    }
    else if (fdn_order == 8)
    {
        initial_fdn_config.delays = {809, 877, 937, 1049, 1151, 1249, 1373, 1499};
    }
    else
    {
        initial_fdn_config.delays = sfFDN::GetDelayLengths(fdn_order, 512, 3000, sfFDN::DelayLengthType::Random, 42);
    }

    if (randomize)
    {
        std::mt19937 rng(42);
        std::normal_distribution<float> gain_dist(-1.f, 1.f);
        for (uint32_t i = 0; i < fdn_order; ++i)
        {
            initial_fdn_config.input_gains[i] = gain_dist(rng);
            initial_fdn_config.output_gains[i] = gain_dist(rng);
        }
        initial_fdn_config.matrix_info = sfFDN::GenerateMatrix(fdn_order, sfFDN::ScalarMatrixType::Random);
    }
    else
    {
        initial_fdn_config.matrix_info = sfFDN::GenerateMatrix(fdn_order, sfFDN::ScalarMatrixType::Householder, 42);
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
                                                       const sfFDN::FDNConfig& initial_fdn_config,
                                                       const fdn_optimization::OptimizationAlgoParams& optimizer_params,
                                                       const std::tuple<double, double, double>& loss_weights)
{

    std::vector params_to_optimize = {fdn_optimization::OptimizationParamType::Gains,
                                      fdn_optimization::OptimizationParamType::Matrix};

    fdn_optimization::OptimizationInfo opt_info{.parameters_to_optimize = params_to_optimize,
                                                .initial_fdn_config = initial_fdn_config,
                                                .ir_size = kSampleRate,
                                                .gradient_method = fdn_optimization::GradientMethod::CentralDifferences,
                                                .gradient_delta = 1e-4,
                                                .spectral_flatness_weight = std::get<0>(loss_weights),
                                                .sparsity_weight = std::get<1>(loss_weights),
                                                .power_envelope_weight = std::get<2>(loss_weights),
                                                .target_rir = {},
                                                .optimizer_params = optimizer_params};

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

fdn_optimization::OptimizationResult OptimizeSpectrum(quill::Logger* logger, const sfFDN::FDNConfig& initial_fdn_config,
                                                      const std::vector<float>& target_rir)
{
    fdn_optimization::AdamParameters opt_params{.step_size = 0.5,
                                                .learning_rate_decay = 0.99,
                                                .decay_step_size = 1,
                                                .epoch_restarts = 180,
                                                .max_restarts = 3,
                                                .tolerance = 1e-4};

    std::vector params_to_optimize = {fdn_optimization::OptimizationParamType::AttenuationFilters,
                                      fdn_optimization::OptimizationParamType::TonecorrectionFilters,
                                      fdn_optimization::OptimizationParamType::OverallGain};

    fdn_optimization::OptimizationInfo opt_info{.parameters_to_optimize = params_to_optimize,
                                                .initial_fdn_config = initial_fdn_config,
                                                .ir_size = static_cast<uint32_t>(target_rir.size()),
                                                .gradient_method = fdn_optimization::GradientMethod::CentralDifferences,
                                                .gradient_delta = 1e-2,
                                                .target_rir = target_rir,
                                                .optimizer_params = opt_params};

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

int main(int argc, char** argv)
{
    quill::Backend::start();
    quill::Logger* logger = quill::Frontend::create_or_get_logger(
        "root", quill::Frontend::create_or_get_sink<quill::ConsoleSink>("sink_id_1"));

    CLI::App app{"FDN Optimization Tool"};

    std::string ir_filename;
    app.add_option("-i,--ir", ir_filename, "Path to target RIR WAV file")->check(CLI::ExistingFile);

    uint32_t fdn_order = kFDNOrder;
    app.add_option("-n,--num_channels", fdn_order, "FDN order (number of channels), e.g., 4, 6, 8")
        ->default_val(kFDNOrder);

    bool colorless_only = false;
    app.add_flag("-c,--colorless_only", colorless_only, "Only perform colorless optimization");

    double spectral_flatness_weight = 1.0;
    app.add_option("--spectral_flatness_weight", spectral_flatness_weight, "Weight for spectral flatness loss term")
        ->default_val(1.0);
    double sparsity_weight = 1.0;
    app.add_option("--sparsity_weight", sparsity_weight, "Weight for sparsity loss term")->default_val(1.0);
    double power_envelope_weight = 1.0;
    app.add_option("--power_envelope_weight", power_envelope_weight, "Weight for power envelope loss term")
        ->default_val(1.0);

    bool randomize_initial = false;
    app.add_flag("--randomize_initial_params", randomize_initial,
                 "Randomize initial FDN configuration instead of using Householder matrix");

    std::string output_dir = "optim_output";
    app.add_option("-o,--output_dir", output_dir, "Output directory for optimization results    ")
        ->capture_default_str();

    // ADAM
    CLI::App* adam_sub = app.add_subcommand("Adam", "Use Adam optimization algorithm");
    fdn_optimization::AdamParameters adam_params;
    adam_sub->add_option("--step_size", adam_params.step_size, "Step size for Adam optimizer");
    adam_sub->add_option("--learning_rate_decay", adam_params.learning_rate_decay,
                         "Learning rate decay for Adam optimizer");
    adam_sub->add_option("--decay_step_size", adam_params.decay_step_size, "Decay step size for Adam optimizer");
    adam_sub->add_option("--epoch_restarts", adam_params.epoch_restarts, "Number of epoch restarts for Adam optimizer");
    adam_sub->add_option("--max_restarts", adam_params.max_restarts, "Maximum number of restarts for Adam optimizer");
    adam_sub->add_option("--tolerance", adam_params.tolerance, "Tolerance for Adam optimizer");

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
    sa_sub->add_option("--max_move_coef", sa_params.max_move_coef, "Max move coefficient for Simulated Annealing");
    sa_sub->add_option("--init_move_coef", sa_params.init_move_coef,
                       "Initial move coefficient for Simulated Annealing");
    sa_sub->add_option("--gain", sa_params.gain, "Gain for Simulated Annealing");

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

    // Differential Evolution
    CLI::App* de_sub = app.add_subcommand("DifferentialEvolution", "Use Differential Evolution optimization algorithm");
    fdn_optimization::DifferentialEvolutionParameters de_params;
    de_sub->add_option("--population_size", de_params.population_size,
                       "Population size for Differential Evolution optimizer");
    de_sub->add_option("--max_generation", de_params.max_generation,
                       "Maximum generations for Differential Evolution optimizer");
    de_sub->add_option("--crossover_rate", de_params.crossover_rate,
                       "Crossover rate for Differential Evolution optimizer");
    de_sub->add_option("--differential_weight", de_params.differential_weight,
                       "Differential weight for Differential Evolution optimizer");
    de_sub->add_option("--tolerance", de_params.tolerance, "Tolerance for Differential Evolution optimizer");

    // PSO
    CLI::App* pso_sub = app.add_subcommand("PSO", "Use Particle Swarm Optimization algorithm");
    fdn_optimization::PSOParameters pso_params;
    pso_sub->add_option("--num_particles", pso_params.num_particles, "Number of particles for PSO optimizer");
    pso_sub->add_option("--max_iterations", pso_params.max_iterations, "Maximum iterations for PSO optimizer");
    pso_sub->add_option("--horizon_size", pso_params.horizon_size, "Horizon size for PSO optimizer");
    pso_sub->add_option("--inertia_weight", pso_params.exploitation_factor, "Exploitation factor for PSO optimizer");
    pso_sub->add_option("--exploration_factor", pso_params.exploration_factor, "Exploration factor for PSO optimizer");

    // L-BFGS
    CLI::App* lbfgs_sub = app.add_subcommand("L-BFGS", "Use L-BFGS optimization algorithm");
    fdn_optimization::L_BFGSParameters lbfgs_params;
    lbfgs_sub->add_option("--num_basis", lbfgs_params.num_basis, "Number of basis vectors for L-BFGS optimizer");
    lbfgs_sub->add_option("--max_iterations", lbfgs_params.max_iterations, "Maximum iterations for L-BFGS optimizer");
    lbfgs_sub->add_option("--wolfe", lbfgs_params.wolfe, "Wolfe condition parameter for L-BFGS optimizer");
    lbfgs_sub->add_option("--min_gradient_norm", lbfgs_params.min_gradient_norm,
                          "Minimum gradient norm for L-BFGS optimizer");
    lbfgs_sub->add_option("--factor", lbfgs_params.factor, "Factor for L-BFGS optimizer");
    lbfgs_sub->add_option("--max_line_search_trials", lbfgs_params.max_line_search_trials,
                          "Maximum line search trials for L-BFGS optimizer");
    lbfgs_sub->add_option("--min_step", lbfgs_params.min_step, "Minimum step size for L-BFGS optimizer");
    lbfgs_sub->add_option("--max_step", lbfgs_params.max_step, "Maximum step size for L-BFGS optimizer");

    // Gradient Descent
    CLI::App* gd_sub = app.add_subcommand("GradientDescent", "Use Gradient Descent optimization algorithm");
    fdn_optimization::GradientDescentParameters gd_params;
    gd_sub->add_option("--step_size", gd_params.step_size, "Step size for Gradient Descent optimizer");
    gd_sub->add_option("--max_iterations", gd_params.max_iterations,
                       "Maximum iterations for Gradient Descent optimizer");
    gd_sub->add_option("--tolerance", gd_params.tolerance, "Tolerance for Gradient Descent optimizer");

    // CMAES
    CLI::App* cmaes_sub = app.add_subcommand("CMAES", "Use CMA-ES optimization algorithm");
    fdn_optimization::CMAESParameters cmaes_params;
    cmaes_sub->add_option("--population_size", cmaes_params.population_size, "Population size for CMA-ES optimizer");
    cmaes_sub->add_option("--max_iterations", cmaes_params.max_iterations, "Maximum iterations for CMA-ES optimizer");
    cmaes_sub->add_option("--tolerance", cmaes_params.tolerance, "Tolerance for CMA-ES optimizer");
    cmaes_sub->add_option("--step_size", cmaes_params.step_size, "Step size for CMA-ES optimizer");

    // Random Search
    CLI::App* random_search_sub = app.add_subcommand("RandomSearch", "Use Random Search optimization algorithm");
    fdn_optimization::RandomSearchParameters random_search_params;
    random_search_sub->add_option("--time_limit", random_search_params.time_limit_seconds,
                                  "Time limit in seconds for Random Search optimizer");

    app.set_config("--config");
    app.allow_config_extras(CLI::config_extras_mode::ignore);

    app.require_subcommand(1);
    CLI11_PARSE(app, argc, argv);

    auto config_filename = app.get_config_ptr()->as<std::string>();
    LOG_INFO(logger, "Using configuration file: {}", config_filename);

    fdn_optimization::OptimizationAlgoParams optimizer_params = adam_params;

    std::string selected_optimizer;

    if (app.got_subcommand("adam"))
    {
        LOG_INFO(logger, "Using Adam optimization algorithm.");
        optimizer_params = adam_params;
        selected_optimizer = "adam";
    }
    else if (app.got_subcommand("SPSA"))
    {
        LOG_INFO(logger, "Using SPSA optimization algorithm.");
        optimizer_params = spsa_params;
        selected_optimizer = "SPSA";
    }
    else if (app.got_subcommand("SimulatedAnnealing"))
    {
        LOG_INFO(logger, "Using Simulated Annealing optimization algorithm.");
        optimizer_params = sa_params;
        selected_optimizer = "SimulatedAnnealing";
    }
    else if (app.got_subcommand("CNE"))
    {
        LOG_INFO(logger, "Using CNE optimization algorithm.");
        optimizer_params = cne_params;
        selected_optimizer = "CNE";
    }
    else if (app.got_subcommand("DifferentialEvolution"))
    {
        LOG_INFO(logger, "Using Differential Evolution optimization algorithm.");
        optimizer_params = de_params;
        selected_optimizer = "DifferentialEvolution";
    }
    else if (app.got_subcommand("PSO"))
    {
        LOG_INFO(logger, "Using Particle Swarm Optimization algorithm.");
        optimizer_params = pso_params;
        selected_optimizer = "PSO";
    }
    else if (app.got_subcommand("L-BFGS"))
    {
        LOG_INFO(logger, "Using L-BFGS optimization algorithm.");
        optimizer_params = lbfgs_params;
        selected_optimizer = "L-BFGS";
    }
    else if (app.got_subcommand("GradientDescent"))
    {
        LOG_INFO(logger, "Using Gradient Descent optimization algorithm.");
        optimizer_params = gd_params;
        selected_optimizer = "GradientDescent";
    }
    else if (app.got_subcommand("CMAES"))
    {
        LOG_INFO(logger, "Using CMA-ES optimization algorithm.");
        optimizer_params = cmaes_params;
        selected_optimizer = "CMAES";
    }
    else if (app.got_subcommand("RandomSearch"))
    {
        LOG_INFO(logger, "Using Random Search optimization algorithm.");
        optimizer_params = random_search_params;
        selected_optimizer = "RandomSearch";
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
    }

    auto now = std::chrono::system_clock::now();
    auto local_now = std::chrono::current_zone()->to_local(std::chrono::floor<std::chrono::seconds>(now));

    std::string timestamp = std::format("{:%Y%m%d_%H%M%S}", local_now);
    LOG_INFO(logger, "Optimization timestamp: {}", timestamp);

    std::string out_dir = timestamp + "_" + selected_optimizer;

    std::filesystem::create_directory(output_dir);

    std::filesystem::path optim_subdir = std::filesystem::path(output_dir) / out_dir;
    std::filesystem::create_directory(optim_subdir);

    auto initial_fdn_config = CreateInitialFDNConfig(fdn_order, randomize_initial);
    auto result = OptimizeColorless(logger, initial_fdn_config, optimizer_params,
                                    std::make_tuple(spectral_flatness_weight, sparsity_weight, power_envelope_weight));
    LOG_INFO(logger, "Colorless optimization completed in {:.2f} s with {} evaluations.", result.total_time.count(),
             result.total_evaluations);

    WriteConfigToFile(result.initial_fdn_config, optim_subdir / "initial_fdn_config.txt", logger);
    WriteConfigToFile(result.optimized_fdn_config, optim_subdir / "colorless_fdn_config.txt", logger);
    SaveImpulseResponse(result.initial_fdn_config, kSampleRate * 3.f, optim_subdir / "colorless_initial_ir.wav",
                        logger);
    SaveImpulseResponse(result.optimized_fdn_config, kSampleRate * 3.f, optim_subdir / "colorless_ir.wav", logger);
    WriteLossHistoryToFile(result.loss_history, result.loss_names, optim_subdir / "colorless_loss_history.txt", logger);
    if (colorless_only)
    {
        LOG_INFO(logger, "Colorless-only optimization flag set. Exiting.");
        return 0;
    }

    result = OptimizeSpectrum(logger, result.optimized_fdn_config, target_rir);
    LOG_INFO(logger, "Filter optimization completed in {:.2f} s with {} evaluations.", result.total_time.count(),
             result.total_evaluations);

    WriteConfigToFile(result.optimized_fdn_config, "optim_output/optimized_fdn_config.txt", logger);
    WriteFilterConfigToFile(result.optimized_fdn_config, "optim_output/optimized_filter_config.txt", logger);
    SaveImpulseResponse(result.initial_fdn_config, target_rir.size(), "optim_output/initial_ir.wav", logger);
    SaveImpulseResponse(result.optimized_fdn_config, target_rir.size(), "optim_output/optimized_ir.wav", logger);
    WriteLossHistoryToFile(result.loss_history, result.loss_names, "optim_output/loss_history.txt", logger);

    return 0;
}