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
#include <thread>
#include <vector>

namespace
{

template <typename T>
void SetOptimizerParams(const T& params, fdn_optimization::OptimizationAlgoParams& optim_params)
{
    optim_params = params;
}

struct ExecutionOptions
{
    fdn_optimization::GradientMethod gradient_method = fdn_optimization::GradientMethod::CentralDifferences;
    uint32_t seed = 0;
    uint32_t gradient_threads = fdn_optimization::DefaultGradientThreadCount();
    uint32_t optimizer_threads = 1;
    double max_time_seconds = 0.0;
    uint64_t max_objective_evaluations = 0;
    size_t sparsity_window_samples = 4096;
    uint32_t spectral_fft_size = 65536;
    bool record_trajectory = false;
};

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

sfFDN::FDNConfig CreateInitialFDNConfig(uint32_t fdn_order, bool randomize = false, bool random_delays = false,
                                        uint32_t seed = 0)
{
    const uint32_t sf_seed = std::max(seed, 1u);
    sfFDN::FDNConfig initial_fdn_config{};
    initial_fdn_config.fdn_size = fdn_order;
    initial_fdn_config.transposed = false;
    initial_fdn_config.direct_gain = 0.0f;
    initial_fdn_config.sample_rate = kSampleRate;
    initial_fdn_config.block_size = 128;
    initial_fdn_config.input_block_config.parallel_gains_config = {.mode = sfFDN::ParallelGainsMode::Split,
                                                                   .gains = std::vector<float>(fdn_order, 0.5f),
                                                                   .time_varying_config = {}};
    initial_fdn_config.output_block_config.parallel_gains_config = {.mode = sfFDN::ParallelGainsMode::Merge,
                                                                    .gains = std::vector<float>(fdn_order, 0.5f),
                                                                    .time_varying_config = {}};

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
        std::cout << "Using random delays..." << "\n";
        initial_fdn_config.delay_bank_config.delays =
            sfFDN::GetDelayLengths(fdn_order, 512, 3000, sfFDN::DelayLengthType::Uniform, sf_seed);
    }

    initial_fdn_config.delay_bank_config.block_size = 128;
    initial_fdn_config.delay_bank_config.interpolation_type = sfFDN::DelayInterpolationType::None;

    // initial_fdn_config.loop_filter_configs = sfFDN::ProportionalAttenuationConfig{10.f};
    sfFDN::AttenuationFilterBankOptions loop_filters;
    for (uint32_t i = 0; i < fdn_order; ++i)
    {
        sfFDN::HomogenousFilterOptions c{
            .t60 = 1.f, .delay = initial_fdn_config.delay_bank_config.delays[i], .sample_rate = kSampleRate};
        loop_filters.filter_configs.push_back(c);
    }
    initial_fdn_config.attenuation_filter_bank_config = std::move(loop_filters);

    if (randomize)
    {
        std::cout << "Using random initial parameters..." << "\n";
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

        initial_fdn_config.feedback_matrix_config = sfFDN::ScalarFeedbackMatrixOptions{
            .matrix_size = fdn_order,
            .type = sfFDN::ScalarMatrixType::Random,
            .custom_matrix = sfFDN::GenerateMatrix(fdn_order, sfFDN::ScalarMatrixType::Random, sf_seed),
            .rng_seed = sf_seed};
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

fdn_optimization::OptimizationStatus WaitForOptimization(fdn_optimization::FDNOptimizer& optimizer,
                                                         quill::Logger* logger)
{
    using namespace std::chrono_literals;

    auto next_progress_log = std::chrono::steady_clock::now() + 1s;
    while (true)
    {
        const auto status = optimizer.GetStatus();
        if (status == fdn_optimization::OptimizationStatus::Completed ||
            status == fdn_optimization::OptimizationStatus::Canceled ||
            status == fdn_optimization::OptimizationStatus::Failed)
        {
            return status;
        }

        const auto now = std::chrono::steady_clock::now();
        if (now >= next_progress_log)
        {
            const auto progress = optimizer.GetProgress();
            float last_loss = 0.0f;
            if (!progress.loss_history.empty() && !progress.loss_history.front().empty())
                last_loss = static_cast<float>(progress.loss_history.front().back());

            LOG_DEBUG(logger, "Elapsed Time: {:.2f} s, Evaluations: {}, Last Loss: {:.6f}",
                      progress.elapsed_time.count(), progress.evaluation_count, last_loss);
            next_progress_log = now + 1s;
        }

        std::this_thread::sleep_for(100ms);
    }
}

fdn_optimization::OptimizationResult OptimizeColorless(quill::Logger* logger,
                                                       const sfFDN::FDNConfig& initial_fdn_config,
                                                       const fdn_optimization::OptimizationAlgoParams& optimizer_params,
                                                       const std::tuple<double, double, double>& loss_weights,
                                                       const ExecutionOptions& execution_options, bool verbose)
{

    std::vector params_to_optimize = {fdn_optimization::OptimizationParamType::Gains,
                                      fdn_optimization::OptimizationParamType::Matrix};

    std::vector<std::shared_ptr<fdn_optimization::AudioLoss>> loss_functions;

    // Spectral Flatness Loss
    const double spectral_flatness_weight = std::get<0>(loss_weights);
    if (spectral_flatness_weight > 0.0)
    {
        constexpr float kTargetSpectralFlatness = 0.5575f;
        loss_functions.push_back(std::make_shared<fdn_optimization::SpectralFlatnessLoss>(
            kTargetSpectralFlatness, spectral_flatness_weight, execution_options.spectral_fft_size));
    }

    // Time Domain Sparsity Loss
    const double sparsity_weight = std::get<1>(loss_weights);
    if (sparsity_weight > 0.0)
    {
        loss_functions.push_back(std::make_shared<fdn_optimization::TimeDomainSparsityLoss>(
            sparsity_weight, execution_options.sparsity_window_samples));
    }

    // Power Envelope Loss
    // TODO

    fdn_optimization::OptimizationInfo opt_info{.parameters_to_optimize = params_to_optimize,
                                                .initial_fdn_config = initial_fdn_config,
                                                .ir_size = kSampleRate,
                                                .gradient_method = execution_options.gradient_method,
                                                .target_rir = {},
                                                .early_fir = {},
                                                .optimizer_params = optimizer_params,
                                                .seed = execution_options.seed,
                                                .gradient_threads = execution_options.gradient_threads,
                                                .optimizer_threads = execution_options.optimizer_threads,
                                                .max_time_seconds = execution_options.max_time_seconds,
                                                .max_objective_evaluations =
                                                    execution_options.max_objective_evaluations,
                                                .record_trajectory = execution_options.record_trajectory};

    fdn_optimization::FDNOptimizer optimizer(logger, verbose);

    optimizer.SetLossFunctions(loss_functions);
    optimizer.StartOptimization(opt_info);

    if (WaitForOptimization(optimizer, logger) != fdn_optimization::OptimizationStatus::Completed)
        throw std::runtime_error("Colorless optimization did not complete successfully.");

    auto result = optimizer.GetResult();
    return result;
}

fdn_optimization::OptimizationResult OptimizeSpectrum(quill::Logger* logger, const sfFDN::FDNConfig& initial_fdn_config,
                                                      const fdn_optimization::OptimizationAlgoParams& optimizer_params,
                                                      const std::vector<float>& target_rir,
                                                      const std::vector<float>& early_fir,
                                                      const std::tuple<double, double, double>& loss_weights,
                                                      const ExecutionOptions& execution_options, bool verbose)
{
    std::vector params_to_optimize = {fdn_optimization::OptimizationParamType::AttenuationFilters,
                                      fdn_optimization::OptimizationParamType::TonecorrectionFilters,
                                      fdn_optimization::OptimizationParamType::OverallGain};

    fdn_optimization::OptimizationInfo opt_info{.parameters_to_optimize = params_to_optimize,
                                                .initial_fdn_config = initial_fdn_config,
                                                .ir_size = static_cast<uint32_t>(target_rir.size()),
                                                .gradient_method = execution_options.gradient_method,
                                                .target_rir = target_rir,
                                                .early_fir = early_fir,
                                                .optimizer_params = optimizer_params,
                                                .seed = execution_options.seed,
                                                .gradient_threads = execution_options.gradient_threads,
                                                .optimizer_threads = execution_options.optimizer_threads,
                                                .max_time_seconds = execution_options.max_time_seconds,
                                                .max_objective_evaluations =
                                                    execution_options.max_objective_evaluations,
                                                .record_trajectory = execution_options.record_trajectory};

    fdn_optimization::FDNOptimizer optimizer(logger, verbose);

    std::vector<std::shared_ptr<fdn_optimization::AudioLoss>> loss_functions;

    const double edc_weight = std::get<0>(loss_weights);
    if (edc_weight > 0.0)
    {
        loss_functions.push_back(std::make_shared<fdn_optimization::EnergyDecayCurveLoss>(target_rir, edc_weight));
    }

    constexpr uint32_t kMelEdrFftLength = 4096;
    constexpr uint32_t kMelEdrHopSize = 128;
    constexpr uint32_t kMelEdrWindowSize = 1024;
    constexpr uint32_t kMelEdrNumBands = 32;

    const double mel_edr_weight = std::get<1>(loss_weights);
    if (mel_edr_weight > 0.0)
    {
        audio_utils::analysis::EnergyDecayReliefOptions edr_options{.fft_length = kMelEdrFftLength,
                                                                    .hop_size = kMelEdrHopSize,
                                                                    .window_size = kMelEdrWindowSize,
                                                                    .window_type = audio_utils::FFTWindowType::Hann,
                                                                    .n_mels = kMelEdrNumBands,
                                                                    .to_db = true};
        loss_functions.push_back(
            std::make_shared<fdn_optimization::EnergyDecayReliefLoss>(target_rir, edr_options, mel_edr_weight));
    }

    const double weighted_edr_weight = std::get<2>(loss_weights);
    if (weighted_edr_weight > 0.0)
    {
        audio_utils::analysis::EnergyDecayReliefOptions edr_options{.fft_length = kMelEdrFftLength,
                                                                    .hop_size = kMelEdrHopSize,
                                                                    .window_size = kMelEdrWindowSize,
                                                                    .window_type = audio_utils::FFTWindowType::Hann,
                                                                    .n_mels = kMelEdrNumBands,
                                                                    .to_db = true};
        loss_functions.push_back(
            std::make_shared<fdn_optimization::WeightedEDRLoss>(target_rir, edr_options, -20.0f, weighted_edr_weight));
    }

    optimizer.SetLossFunctions(loss_functions);
    optimizer.StartOptimization(opt_info);

    if (WaitForOptimization(optimizer, logger) != fdn_optimization::OptimizationStatus::Completed)
        throw std::runtime_error("Spectrum optimization did not complete successfully.");

    auto result = optimizer.GetResult();
    return result;
}

void RenderAudio(const sfFDN::FDNConfig& fdn_config, const std::string& input_filename,
                 const std::filesystem::path& output_dir, quill::Logger* logger, std::span<const float> early_fir)
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

    auto fdn = sfFDN::CreateFDNFromConfig(fdn_config);
    fdn->SetDirectGain(0.0f);

    std::vector<float> output_audio(audio_file.size(), 0.0f);
    sfFDN::AudioBuffer input_buffer(audio_file);
    sfFDN::AudioBuffer output_buffer(output_audio);

    fdn->Process(input_buffer, output_buffer);

    if (!early_fir.empty())
    {
        auto direct_audio = audio_utils::analysis::Convolve(audio_file, early_fir);
        direct_audio.resize(output_audio.size());
        for (size_t index = 0; index < output_audio.size(); ++index)
            output_audio[index] += direct_audio[index];
    }

    std::filesystem::path output_path =
        output_dir / (std::filesystem::path(input_filename).stem().string() + "_wet.wav");
    audio_utils::audio_file::WriteWavFile(output_path.string(), output_audio, kSampleRate);
}

} // namespace

int main(int argc, char** argv)
{
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
    bool no_save_output = false;
    app.add_flag("--no-save-output", no_save_output, "Disable writing WAV and configuration output");

    bool verbose = false;
    app.add_flag("-v,--verbose", verbose, "Enable verbose logging");

    std::string gradient_method_name = "central";
    app.add_option("--gradient_method", gradient_method_name, "Finite-difference method")
        ->check(CLI::IsMember({"central", "forward"}));

    ExecutionOptions execution_options;
    app.add_option("--seed", execution_options.seed, "Random seed")->capture_default_str();
    app.add_option("--gradient_threads", execution_options.gradient_threads,
                   "Threads used for finite-difference gradients; 0 uses the OpenMP maximum")
        ->capture_default_str();
    app.add_option("--optimizer_threads", execution_options.optimizer_threads,
                   "Threads used by supported population optimizers; 0 uses the OpenMP maximum")
        ->capture_default_str();
    app.add_option("--max_time_seconds", execution_options.max_time_seconds,
                   "In-process optimizer time budget in seconds; 0 disables the budget")
        ->capture_default_str();
    app.add_option("--max_objective_evaluations", execution_options.max_objective_evaluations,
                   "Objective-evaluation budget; 0 disables the budget")
        ->capture_default_str();
    app.add_option("--sparsity_window_samples", execution_options.sparsity_window_samples,
                   "Number of initial IR samples used by the sparsity loss; thesis runs used 4096")
        ->capture_default_str();
    app.add_option("--spectral_fft_size", execution_options.spectral_fft_size,
                   "FFT size used by the spectral-flatness loss; thesis IPP runs used 65536")
        ->capture_default_str();

    double spectral_flatness_weight = 1.0;
    app.add_option("--spectral_flatness_weight", spectral_flatness_weight, "Weight for spectral flatness loss term")
        ->default_val(1.0);
    double sparsity_weight = 0.5;
    app.add_option("--sparsity_weight", sparsity_weight, "Weight for sparsity loss term")->default_val(0.5);
    double power_envelope_weight = 0.0;
    app.add_option("--power_envelope_weight", power_envelope_weight, "Weight for power envelope loss term")
        ->default_val(0.0);

    double edc_weight = 0.1;
    app.add_option("--edc_weight", edc_weight, "Weight for EDC loss term")->default_val(0.1);
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

    std::string result_json_path;
    app.add_option("--result_json", result_json_path, "Write machine-readable colorless results to this JSON file");
    std::string spectrum_result_json_path;
    app.add_option("--spectrum_result_json", spectrum_result_json_path,
                   "Write machine-readable RIR-matching results to this JSON file");
    std::string trajectory_jsonl_path;
    app.add_option("--trajectory_jsonl", trajectory_jsonl_path, "Write colorless accepted-step trajectory as JSONL");
    std::string spectrum_trajectory_jsonl_path;
    app.add_option("--spectrum_trajectory_jsonl", spectrum_trajectory_jsonl_path,
                   "Write RIR-matching accepted-step trajectory as JSONL");

    fdn_optimization::OptimizationAlgoParams optimizer_params;

    // ADAM
    CLI::App* adam_sub = app.add_subcommand("Adam", "Use Adam optimization algorithm");
    fdn_optimization::AdamParameters adam_params;
    adam_sub->add_option("--step_size", adam_params.step_size, "Step size for Adam optimizer");
    adam_sub->add_option("--beta1", adam_params.beta1, "Beta1 parameter for Adam optimizer");
    adam_sub->add_option("--beta2", adam_params.beta2, "Beta2 parameter for Adam optimizer");
    adam_sub->add_option("--tolerance", adam_params.tolerance, "Tolerance for Adam optimizer");
    adam_sub->add_option("--gradient_delta", adam_params.gradient_delta, "Gradient delta for Adam optimizer");
    adam_sub->add_option("--max_iterations", adam_params.max_iterations, "Maximum Adam iterations");
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

    if (no_save_output)
    {
        save_output = false;
    }

    execution_options.gradient_method = gradient_method_name == "forward"
                                            ? fdn_optimization::GradientMethod::ForwardDifferences
                                            : fdn_optimization::GradientMethod::CentralDifferences;
    execution_options.record_trajectory = !trajectory_jsonl_path.empty();
    ExecutionOptions matching_execution_options = execution_options;
    matching_execution_options.record_trajectory = !spectrum_trajectory_jsonl_path.empty();

    if (!colorless_only && ir_filename.empty())
    {
        std::cerr << "RIR matching requires --ir; use --colorless_only to skip matching.\n";
        return -1;
    }
    if (execution_options.max_time_seconds < 0.0)
    {
        std::cerr << "--max_time_seconds cannot be negative.\n";
        return -1;
    }
    if (execution_options.spectral_fft_size < kSampleRate)
    {
        std::cerr << "--spectral_fft_size must be at least " << kSampleRate << " samples.\n";
        return -1;
    }

    if (verbose)
    {
        logger->set_log_level(quill::LogLevel::Debug);
    }

    LOG_INFO(logger, "Requested gradient threads: {}, optimizer threads: {}.", execution_options.gradient_threads,
             execution_options.optimizer_threads);

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
        std::filesystem::create_directories(optim_subdir);
    }

    arma::arma_rng::set_seed(execution_options.seed);
    auto initial_fdn_config =
        CreateInitialFDNConfig(fdn_order, randomize_initial, random_delays, execution_options.seed);
    const auto shortest_delay = *std::ranges::min_element(initial_fdn_config.delay_bank_config.delays);
    if (execution_options.sparsity_window_samples <= shortest_delay)
    {
        std::cerr << "--sparsity_window_samples must be greater than the shortest delay (" << shortest_delay
                  << " samples).\n";
        return -1;
    }

    if (save_output)
    {
        // initial_fdn_config.attenuation_filter_config = sfFDN::ProportionalAttenuationConfig{2.f};
        SaveImpulseResponse(initial_fdn_config, kSampleRate * 2.f, optim_subdir / "initial_ir.wav", logger);
        WriteConfigToFile(initial_fdn_config, optim_subdir / "initial_fdn_config.txt", logger);
    }

    LOG_INFO(logger, "Starting colorless optimization...");
    fdn_optimization::OptimizationResult result;

    try
    {
        result =
            OptimizeColorless(logger, initial_fdn_config, optimizer_params,
                              std::make_tuple(spectral_flatness_weight, sparsity_weight, power_envelope_weight),
                              execution_options, verbose);
    }
    catch (const std::exception& error)
    {
        std::cerr << "Colorless optimization failed: " << error.what() << '\n';
        return -1;
    }

    LOG_INFO(logger, "[Colorless] Final loss: {:.6f}", result.best_loss);
    LOG_INFO(logger, "[Colorless] Elapsed time: {:.4f} s", result.total_time.count());
    LOG_INFO(logger, "[Colorless] Total evaluations: {}", result.total_evaluations);
    LOG_INFO(logger, "[Colorless] Objective evaluations: {}", result.objective_evaluations);
    LOG_INFO(logger, "[Colorless] Termination reason: {}", result.termination_reason);

    if (!trajectory_jsonl_path.empty() &&
        !WriteTrajectoryJsonl(result.trajectory, result.loss_names, trajectory_jsonl_path, logger))
    {
        return -1;
    }

    if (!result_json_path.empty())
    {
        const uint32_t actual_fft_size = audio_utils::FFT::NextSupportedFFTSize(execution_options.spectral_fft_size);
        const nlohmann::json metadata = {{"stage", "colorless"},
                                         {"optimizer", selected_optimizer},
                                         {"fdn_order", fdn_order},
                                         {"sample_rate", kSampleRate},
                                         {"ir_samples", kSampleRate},
                                         {"seed", execution_options.seed},
                                         {"gradient_method", gradient_method_name},
                                         {"gradient_threads", execution_options.gradient_threads},
                                         {"optimizer_threads", execution_options.optimizer_threads},
                                         {"max_time_seconds", execution_options.max_time_seconds},
                                         {"max_objective_evaluations",
                                          execution_options.max_objective_evaluations},
                                         {"spectral_flatness_target", 0.5575},
                                         {"spectral_flatness_weight", spectral_flatness_weight},
                                         {"sparsity_weight", sparsity_weight},
                                         {"sparsity_window_samples", execution_options.sparsity_window_samples},
                                         {"fft_backend", audio_utils::FFT::BackendName()},
                                         {"requested_fft_size", execution_options.spectral_fft_size},
                                         {"actual_fft_size", actual_fft_size}};
        if (!WriteJsonToFile(OptimizationResultToJson(result, optimizer_params, metadata), result_json_path, logger))
            return -1;
    }

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
            if (sample_rate != kSampleRate || num_channels != 1)
            {
                LOG_ERROR(logger, "Target RIR must be mono at {} Hz.", kSampleRate);
                return -1;
            }
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
            if (sample_rate != kSampleRate || num_channels != 1 || early_fir.size() > target_rir.size())
            {
                LOG_ERROR(logger, "Early FIR must be mono at {} Hz and no longer than the target RIR.", kSampleRate);
                return -1;
            }
        }
        else
        {
            LOG_ERROR(logger, "Failed to load early reflection FIR.");
            return -1;
        }
    }

    try
    {
        result = OptimizeSpectrum(logger, initial_fdn_config, optimizer_params, target_rir, early_fir,
                                  std::make_tuple(edc_weight, mel_edr_weight, weighted_edr_weight),
                                  matching_execution_options, verbose);
    }
    catch (const std::exception& error)
    {
        std::cerr << "RIR-matching optimization failed: " << error.what() << '\n';
        return -1;
    }
    LOG_INFO(logger, "[Spectrum] Final loss: {:.6f}", result.best_loss);
    LOG_INFO(logger, "[Spectrum] Elapsed time: {:.4f} s", result.total_time.count());
    LOG_INFO(logger, "[Spectrum] Total evaluations: {}", result.total_evaluations);
    LOG_INFO(logger, "[Spectrum] Objective evaluations: {}", result.objective_evaluations);
    LOG_INFO(logger, "[Spectrum] Termination reason: {}", result.termination_reason);

    if (!spectrum_trajectory_jsonl_path.empty() &&
        !WriteTrajectoryJsonl(result.trajectory, result.loss_names, spectrum_trajectory_jsonl_path, logger))
    {
        return -1;
    }

    if (!spectrum_result_json_path.empty())
    {
        const nlohmann::json metadata = {{"stage", "spectrum"},
                                         {"optimizer", selected_optimizer},
                                         {"fdn_order", fdn_order},
                                         {"sample_rate", kSampleRate},
                                         {"target_rir", ir_filename},
                                         {"target_rir_samples", target_rir.size()},
                                         {"early_fir", early_fir_path},
                                         {"seed", matching_execution_options.seed},
                                         {"gradient_method", gradient_method_name},
                                         {"gradient_threads", matching_execution_options.gradient_threads},
                                         {"optimizer_threads", matching_execution_options.optimizer_threads},
                                         {"max_time_seconds", matching_execution_options.max_time_seconds},
                                         {"max_objective_evaluations",
                                          matching_execution_options.max_objective_evaluations},
                                         {"fft_backend", audio_utils::FFT::BackendName()},
                                         {"edc_weight", edc_weight},
                                         {"mel_edr_weight", mel_edr_weight},
                                         {"weighted_edr_weight", weighted_edr_weight}};
        if (!WriteJsonToFile(OptimizationResultToJson(result, optimizer_params, metadata), spectrum_result_json_path,
                             logger))
        {
            return -1;
        }
    }

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
        file << ir_filename << "\n";

        RenderAudio(result.optimized_fdn_config, "./audio/drumloop.wav", optim_subdir, logger, early_fir);
        RenderAudio(result.optimized_fdn_config, "./audio/saxophone.wav", optim_subdir, logger, early_fir);
        RenderAudio(result.optimized_fdn_config, "./audio/bleepsandbloops.wav", optim_subdir, logger, early_fir);
    }

    return 0;
}