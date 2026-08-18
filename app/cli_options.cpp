#include "cli_options.h"

#include "optimizer_cli.h"

#include <map>
#include <utility>

namespace
{

constexpr uint32_t kSampleRate = 48000;

const std::map<std::string, fdn_optimization::GradientMethod> kGradientMethods = {
    {"central", fdn_optimization::GradientMethod::CentralDifferences},
    {"forward", fdn_optimization::GradientMethod::ForwardDifferences},
};

const std::map<std::string, fdn_opt_app::MatchingFilterType> kMatchingFilterTypes = {
    {"10band", fdn_opt_app::MatchingFilterType::TenBand},
    {"3band", fdn_opt_app::MatchingFilterType::ThreeBand},
};

const std::map<std::string, fdn_optimization::EarlyFirMode> kEarlyFirModes = {
    {"direct", fdn_optimization::EarlyFirMode::DirectPath},
    {"excitation", fdn_optimization::EarlyFirMode::Excitation},
};

const std::map<std::string, fdn_optimization::MatchingParameterization> kMatchingParameterizations = {
    {"raw", fdn_optimization::MatchingParameterization::RawClamped},
    {"scaled", fdn_optimization::MatchingParameterization::ScaledSmooth},
};

const std::map<std::string, fdn_optimization::MatchingInitialization> kMatchingInitializations = {
    {"random", fdn_optimization::MatchingInitialization::SeededRandom},
    {"neutral", fdn_optimization::MatchingInitialization::Neutral},
    {"target", fdn_optimization::MatchingInitialization::TargetDerived},
};

void RegisterInputOptions(CLI::App& app, fdn_opt_app::RawCliOptions& options)
{
    auto* group = app.add_option_group("Input and stages", "Input files and optimization stage selection");
    group->add_option("-i,--ir", options.input.ir_filename, "Path to target RIR WAV file")->check(CLI::ExistingFile);
    group->add_option("--early_fir_path", options.input.early_fir_path, "Path to early reflection FIR WAV file")
        ->check(CLI::ExistingFile);
    group->add_option("-n,--num_channels", options.input.fdn_order, "FDN order (number of channels), e.g., 4, 6, 8")
        ->default_val(6);
    auto* colorless_only =
        group->add_flag("-c,--colorless_only", options.input.colorless_only, "Only perform colorless optimization");
    auto* matching_only =
        group->add_flag("--matching_only", options.input.matching_only, "Only perform RIR-matching optimization");
    auto* colorless_config = group
                                 ->add_option("--colorless_config", options.input.colorless_config_path,
                                              "Path to a saved colorless FDN JSON config")
                                 ->check(CLI::ExistingFile);
    colorless_only->excludes(matching_only);
    matching_only->needs(colorless_config);
}

void RegisterExecutionOptions(CLI::App& app, fdn_opt_app::RawCliOptions& options)
{
    auto* group = app.add_option_group("Execution", "Randomness, threading, budgets, and colorless analysis");
    group->add_option("--gradient_method", options.execution.gradient_method, "Finite-difference method")
        ->transform(CLI::CheckedTransformer(kGradientMethods));
    group->add_option("--seed", options.execution.seed, "Random seed")->capture_default_str();
    group
        ->add_option("--gradient_threads", options.execution.gradient_threads,
                     "Threads used for finite-difference gradients; 0 uses the OpenMP maximum")
        ->capture_default_str();
    group
        ->add_option("--optimizer_threads", options.execution.optimizer_threads,
                     "Threads used by supported population optimizers; 0 uses the OpenMP maximum")
        ->capture_default_str();
    group
        ->add_option("--max_time_seconds", options.execution.max_time_seconds,
                     "In-process optimizer time budget in seconds; 0 disables the budget")
        ->check(CLI::NonNegativeNumber)
        ->capture_default_str();
    group
        ->add_option("--max_objective_evaluations", options.execution.max_objective_evaluations,
                     "Objective-evaluation budget; 0 disables the budget")
        ->capture_default_str();
    group
        ->add_option("--sparsity_window_samples", options.execution.sparsity_window_samples,
                     "Number of initial IR samples used by the sparsity loss; thesis runs used 4096")
        ->capture_default_str();
    group
        ->add_option("--spectral_fft_size", options.execution.spectral_fft_size,
                     "FFT size used by the spectral-flatness loss; thesis IPP runs used 65536")
        ->capture_default_str();
}

void RegisterLossOptions(CLI::App& app, fdn_opt_app::RawCliOptions& options)
{
    auto* colorless = app.add_option_group("Colorless losses", "Colorless optimization loss weights");
    colorless
        ->add_option("--spectral_flatness_weight", options.losses.spectral_flatness_weight,
                     "Weight for spectral flatness loss term")
        ->default_val(1.0);
    colorless->add_option("--sparsity_weight", options.losses.sparsity_weight, "Weight for sparsity loss term")
        ->default_val(0.5);
    colorless
        ->add_option("--power_envelope_weight", options.losses.power_envelope_weight,
                     "Weight for power envelope loss term")
        ->default_val(0.0);

    auto* matching = app.add_option_group("Matching losses", "RIR-matching loss weights");
    matching->add_option("--edc_weight", options.losses.edc_weight, "Weight for EDC loss term")->default_val(0.1);
    matching->add_option("--mel_edr_weight", options.losses.mel_edr_weight, "Weight for Mel EDR loss term")
        ->default_val(1.0);
    matching
        ->add_option("--weighted_edr_weight", options.losses.weighted_edr_weight, "Weight for Weighted EDR loss term")
        ->default_val(0.0);
}

void RegisterMatchingOptions(CLI::App& app, fdn_opt_app::RawCliOptions& options)
{
    auto* group = app.add_option_group("Matching analysis", "RIR-matching model and analysis controls");
    group->add_option("--matching_filter_type", options.matching.filter_type, "Matching attenuation filter type")
        ->transform(CLI::CheckedTransformer(kMatchingFilterTypes));
    group->add_option("--early_fir_mode", options.matching.early_fir_mode, "Early FIR modelling mode")
        ->transform(CLI::CheckedTransformer(kEarlyFirModes));
    group->add_option("--matching_fft_length", options.matching.fft_length, "Mel-EDR FFT length")
        ->check(CLI::PositiveNumber)
        ->capture_default_str();
    group->add_option("--matching_hop_size", options.matching.hop_size, "Mel-EDR hop size")
        ->check(CLI::PositiveNumber)
        ->capture_default_str();
    group->add_option("--matching_window_size", options.matching.window_size, "Mel-EDR window size")
        ->check(CLI::PositiveNumber)
        ->capture_default_str();
    group->add_option("--matching_mel_bands", options.matching.mel_bands, "Number of mel bands")
        ->check(CLI::PositiveNumber)
        ->capture_default_str();
    group
        ->add_option("--matching_parameterization", options.matching.parameter_config.parameterization,
                     "Matching parameter coordinates")
        ->transform(CLI::CheckedTransformer(kMatchingParameterizations));
    group
        ->add_option("--matching_initialization", options.matching.parameter_config.initialization,
                     "Matching parameter initialization")
        ->transform(CLI::CheckedTransformer(kMatchingInitializations));
    group->add_option("--matching_min_t60", options.matching.parameter_config.minimum_t60, "Minimum RT60")
        ->check(CLI::PositiveNumber)
        ->capture_default_str();
    group->add_option("--matching_max_t60", options.matching.parameter_config.maximum_t60, "Maximum RT60")
        ->check(CLI::PositiveNumber)
        ->capture_default_str();
    group
        ->add_option("--matching_tone_scale_db", options.matching.parameter_config.tone_gain_scale_db,
                     "Tone-correction coordinate scale in dB")
        ->check(CLI::PositiveNumber)
        ->capture_default_str();
    group
        ->add_option("--matching_zero_mean_tone_gains", options.matching.parameter_config.zero_mean_tone_gains,
                     "Constrain tone-correction gains to zero mean")
        ->capture_default_str();
}

void RegisterInitializationOptions(CLI::App& app, fdn_opt_app::RawCliOptions& options)
{
    auto* group = app.add_option_group("Initialization", "Initial FDN configuration controls");
    group->add_flag("--randomize_initial_params", options.initialization.randomize_initial,
                    "Randomize initial FDN configuration instead of using Householder matrix");
    group->add_flag("--random_delays", options.initialization.random_delays,
                    "Use random delay lengths instead of predefined sets");
}

void RegisterOutputOptions(CLI::App& app, fdn_opt_app::RawCliOptions& options)
{
    auto* group = app.add_option_group("Output", "Logging and output files");
    group->add_flag("-s,--save_output", options.output.save_output, "Save optimization results to output directory");
    group->add_flag("--no-save-output", options.output.no_save_output, "Disable writing WAV and configuration output");
    group->add_flag("-v,--verbose", options.verbose, "Enable verbose logging");
    group->add_option("-o,--output_dir", options.output.output_dir, "Output directory for optimization results    ")
        ->capture_default_str();
    group->add_option("--result_json", options.output.result_json_path,
                      "Write machine-readable colorless results to this JSON file");
    group->add_option("--spectrum_result_json", options.output.spectrum_result_json_path,
                      "Write machine-readable RIR-matching results to this JSON file");
    group->add_option("--trajectory_jsonl", options.output.trajectory_jsonl_path,
                      "Write colorless accepted-step trajectory as JSONL");
    group->add_option("--spectrum_trajectory_jsonl", options.output.spectrum_trajectory_jsonl_path,
                      "Write RIR-matching accepted-step trajectory as JSONL");
}

} // namespace

namespace fdn_opt_app
{

std::expected<ParsedCliOptions, std::string> ValidateAndNormalizeCliOptions(const RawCliOptions& options)
{
    if (options.input.colorless_only && options.input.matching_only)
    {
        return std::unexpected("--colorless_only and --matching_only are mutually exclusive.");
    }
    if (options.input.matching_only && options.input.colorless_config_path.empty())
    {
        return std::unexpected("--matching_only requires --colorless_config.");
    }
    if (!options.input.colorless_only && options.input.ir_filename.empty())
    {
        return std::unexpected("RIR matching requires --ir; use --colorless_only to skip matching.");
    }
    if (options.matching.window_size == 0 || options.matching.window_size > options.matching.fft_length ||
        options.matching.hop_size == 0 || options.matching.mel_bands == 0)
    {
        return std::unexpected("Invalid matching analysis options.");
    }
    if (options.matching.parameter_config.minimum_t60 <= 0.0 ||
        options.matching.parameter_config.maximum_t60 <= options.matching.parameter_config.minimum_t60 ||
        options.matching.parameter_config.tone_gain_scale_db <= 0.0)
    {
        return std::unexpected("Invalid matching parameterization options.");
    }
    if (options.execution.max_time_seconds < 0.0)
    {
        return std::unexpected("--max_time_seconds cannot be negative.");
    }
    if (!options.input.matching_only && options.execution.spectral_fft_size < kSampleRate)
    {
        return std::unexpected("--spectral_fft_size must be at least 48000 samples.");
    }
    if (options.optimizer.selected_name.empty())
    {
        return std::unexpected("An optimizer subcommand is required.");
    }

    ParsedCliOptions parsed;
    parsed.input = options.input;
    parsed.losses = options.losses;
    parsed.initialization = options.initialization;
    parsed.output = options.output;
    parsed.colorless_execution = options.execution;
    parsed.colorless_execution.record_trajectory = !options.output.trajectory_jsonl_path.empty();
    parsed.matching_execution = options.execution;
    parsed.matching_execution.record_trajectory = !options.output.spectrum_trajectory_jsonl_path.empty();
    parsed.matching = options.matching;
    parsed.optimizer_params = options.optimizer.selected_params;
    parsed.selected_optimizer = options.optimizer.selected_name;
    parsed.save_output = options.output.save_output && !options.output.no_save_output;
    parsed.verbose = options.verbose;
    return parsed;
}

void ConfigureCliApp(CLI::App& app, RawCliOptions& options)
{
    RegisterInputOptions(app, options);
    RegisterExecutionOptions(app, options);
    RegisterLossOptions(app, options);
    RegisterMatchingOptions(app, options);
    RegisterInitializationOptions(app, options);
    RegisterOutputOptions(app, options);
    RegisterOptimizerSubcommands(app, options.optimizer);

    app.set_config("--config");
    app.allow_config_extras(CLI::config_extras_mode::error);
    app.require_subcommand(1);
    app.callback([&app, &options]() {
        auto parsed = ValidateAndNormalizeCliOptions(options);
        if (!parsed)
        {
            throw CLI::ValidationError(parsed.error());
        }
        parsed->config_filename = app.get_config_ptr()->as<std::string>();
        options.parsed = std::move(*parsed);
    });
}

std::string_view GradientMethodName(fdn_optimization::GradientMethod method)
{
    return method == fdn_optimization::GradientMethod::ForwardDifferences ? "forward" : "central";
}

std::string_view MatchingFilterTypeName(MatchingFilterType type)
{
    return type == MatchingFilterType::ThreeBand ? "3band" : "10band";
}

std::string_view EarlyFirModeName(fdn_optimization::EarlyFirMode mode)
{
    return mode == fdn_optimization::EarlyFirMode::Excitation ? "excitation" : "direct";
}

std::string_view MatchingParameterizationName(fdn_optimization::MatchingParameterization parameterization)
{
    return parameterization == fdn_optimization::MatchingParameterization::RawClamped ? "raw" : "scaled";
}

std::string_view MatchingInitializationName(fdn_optimization::MatchingInitialization initialization)
{
    switch (initialization)
    {
    case fdn_optimization::MatchingInitialization::SeededRandom:
        return "random";
    case fdn_optimization::MatchingInitialization::TargetDerived:
        return "target";
    case fdn_optimization::MatchingInitialization::Neutral:
        return "neutral";
    }
    return "neutral";
}

} // namespace fdn_opt_app
