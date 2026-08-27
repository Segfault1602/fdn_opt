#include "application.h"

#include "model.h"
#include "optimization_workflows.h"
#include "utils.h"

#include <armadillo>
#include <audio_utils/audio_file_manager.h>
#include <audio_utils/fft.h>
#include <quill/LogMacros.h>

#include <chrono>
#include <format>
#include <fstream>
#include <iostream>
#include <ranges>
#include <utility>

namespace
{

std::string Timestamp()
{
    const auto now = std::chrono::system_clock::now();
#ifndef __APPLE__
    const auto local_now = std::chrono::current_zone()->to_local(std::chrono::floor<std::chrono::seconds>(now));
#else
    const auto local_now = std::chrono::floor<std::chrono::seconds>(now);
#endif
    return std::format("{:%Y%m%d_%H%M%S}", local_now);
}

} // namespace

namespace fdn_opt_app
{

OptimizationApplication::OptimizationApplication(quill::Logger* logger, ParsedCliOptions options)
    : logger_(logger)
    , options_(std::move(options))
    , fdn_order_(options_.input.fdn_order)
{
}

int OptimizationApplication::Run()
{
    ConfigureLoggingAndReportRun();
    PrepareOutputDirectory();
    arma::arma_rng::set_seed(options_.colorless_execution.seed);

    auto initial_config = CreateOrLoadInitialConfig();
    if (!initial_config)
    {
        if (!initial_config.error().empty())
        {
            std::cerr << initial_config.error() << '\n';
        }
        return -1;
    }
    SaveInitialArtifacts(*initial_config);

    if (!options_.input.matching_only)
    {
        auto colorless_result = RunColorlessStage(*initial_config);
        if (!colorless_result)
        {
            std::cerr << colorless_result.error() << '\n';
            return -1;
        }
        LogStageResult("Colorless", *colorless_result);
        if (!WriteColorlessOutputs(*colorless_result))
        {
            return -1;
        }
        *initial_config = colorless_result->optimized_fdn_config;
        if (options_.input.colorless_only)
        {
            LOG_INFO(logger_, "Colorless-only optimization flag set. Exiting.");
            return 0;
        }
    }

    auto target_rir = LoadTargetRir();
    if (!target_rir)
    {
        return -1;
    }
    auto early_fir = LoadEarlyFir(target_rir->size());
    if (!early_fir)
    {
        return -1;
    }
    auto matching_result = OptimizeMatching(*initial_config, *target_rir, *early_fir);
    if (!matching_result)
    {
        std::cerr << matching_result.error() << '\n';
        return -1;
    }
    LogStageResult("Spectrum", *matching_result);
    return WriteMatchingOutputs(*matching_result, *target_rir, *early_fir) ? 0 : -1;
}

void OptimizationApplication::ConfigureLoggingAndReportRun()
{
    if (options_.verbose)
    {
        logger_->set_log_level(quill::LogLevel::Debug);
    }
    LOG_INFO(logger_, "Requested gradient threads: {}, optimizer threads: {}.",
             options_.colorless_execution.gradient_threads, options_.colorless_execution.optimizer_threads);
    LOG_INFO(logger_, "Using configuration file: {}", options_.config_filename);
    LOG_INFO(logger_, "Selected optimization algorithm: {}", options_.selected_optimizer);
}

void OptimizationApplication::PrepareOutputDirectory()
{
    const std::string timestamp = Timestamp();
    LOG_INFO(logger_, "Optimization timestamp: {}", timestamp);
    if (options_.save_output)
    {
        output_directory_ =
            std::filesystem::path(options_.output.output_dir) / (timestamp + "_" + options_.selected_optimizer);
        std::filesystem::create_directories(output_directory_);
    }
}

std::expected<sfFDN::FDNConfig, std::string> OptimizationApplication::CreateOrLoadInitialConfig()
{
    if (options_.input.matching_only)
    {
        auto config = ReadConfigFromFile(options_.input.colorless_config_path, logger_);
        if (!config)
        {
            return std::unexpected("");
        }
        try
        {
            fdn_optimization::ValidateFDNConfigurationForOptimization(*config);
            if (config->sample_rate != kSampleRate)
            {
                throw std::invalid_argument("FDN config sample rate must be 48000 Hz.");
            }
        }
        catch (const std::exception& error)
        {
            return std::unexpected("Invalid colorless FDN config: " + std::string(error.what()));
        }
        fdn_order_ = config->fdn_size;
        return std::move(*config);
    }

    auto config = CreateInitialFDNConfig(fdn_order_, options_.initialization.randomize_initial,
                                         options_.initialization.random_delays,
                                         options_.initialization.init_seed.value_or(options_.colorless_execution.seed));
    const auto shortest_delay = *std::ranges::min_element(config.delay_bank_config.delays);
    if (options_.colorless_execution.sparsity_window_samples <= shortest_delay)
    {
        return std::unexpected(std::format(
            "--sparsity_window_samples must be greater than the shortest delay ({:g} samples).", shortest_delay));
    }
    return config;
}

void OptimizationApplication::SaveInitialArtifacts(const sfFDN::FDNConfig& config) const
{
    if (options_.save_output)
    {
        SaveImpulseResponse(config, kSampleRate * 2.f, output_directory_ / "initial_ir.wav", logger_);
        WriteConfigToFile(config, output_directory_ / "initial_fdn_config.txt", logger_);
    }
}

std::expected<fdn_optimization::OptimizationResult, std::string> OptimizationApplication::RunColorlessStage(
    const sfFDN::FDNConfig& initial_config) const
{
    LOG_INFO(logger_, "Starting colorless optimization...");
    try
    {
        return OptimizeColorless(logger_, initial_config, options_.optimizer_params,
                                 std::make_tuple(options_.losses.spectral_flatness_weight,
                                                 options_.losses.sparsity_weight,
                                                 options_.losses.power_envelope_weight),
                                 options_.colorless_execution, options_.verbose,
                                 options_.initialization.matrix_parameterization);
    }
    catch (const std::exception& error)
    {
        return std::unexpected("Colorless optimization failed: " + std::string(error.what()));
    }
}

bool OptimizationApplication::WriteColorlessOutputs(const fdn_optimization::OptimizationResult& result) const
{
    if (!options_.output.trajectory_jsonl_path.empty() &&
        !WriteTrajectoryJsonl(result.trajectory, result.loss_names, options_.output.trajectory_jsonl_path, logger_))
    {
        return false;
    }
    if (!options_.output.result_json_path.empty() &&
        !WriteJsonToFile(OptimizationResultToJson(result, options_.optimizer_params, BuildColorlessMetadata()),
                         options_.output.result_json_path, logger_))
    {
        return false;
    }
    if (options_.save_output)
    {
        // Re-save the initial artifacts from the configuration the optimizer actually started from.
        // With a structured parameterization (Householder, circulant) that starting point need not be
        // the configuration built before the stage, and the reported `initial_loss` describes this one.
        SaveInitialArtifacts(result.initial_fdn_config);
        WriteConfigToFile(result.optimized_fdn_config, output_directory_ / "colorless_fdn_config.txt", logger_);
        WriteInfoToFile(result, options_.optimizer_params, output_directory_ / "colorless_fdn_info.txt", logger_);
        SaveImpulseResponse(result.optimized_fdn_config, kSampleRate * 3.f, output_directory_ / "colorless_ir.wav",
                            logger_);
        WriteLossHistoryToFile(result.loss_history, result.loss_names, output_directory_ / "colorless_loss_history.txt",
                               logger_);
    }
    return true;
}

std::expected<std::vector<float>, std::string> OptimizationApplication::LoadMonoAudio(
    const std::string& path, const std::string& label, const std::string& failure_message) const
{
    std::vector<float> audio;
    int sample_rate = kSampleRate;
    int channels = 0;
    if (!audio_utils::audio_file::ReadWavFile(path, audio, sample_rate, channels))
    {
        LOG_ERROR(logger_, "{}", failure_message);
        return std::unexpected("read failed");
    }
    LOG_INFO(logger_, "Loaded {} with {} samples at {} Hz.", path, audio.size(), sample_rate);
    if (sample_rate != kSampleRate || channels != 1)
    {
        LOG_ERROR(logger_, "{} must be mono at {} Hz.", label, kSampleRate);
        return std::unexpected("invalid format");
    }
    return audio;
}

std::expected<std::vector<float>, std::string> OptimizationApplication::LoadTargetRir() const
{
    return LoadMonoAudio(options_.input.ir_filename, "Target RIR", "Failed to load target RIR.");
}

std::expected<std::vector<float>, std::string> OptimizationApplication::LoadEarlyFir(size_t target_size) const
{
    if (options_.input.early_fir_path.empty())
    {
        return std::vector<float>{};
    }
    auto early_fir = LoadMonoAudio(options_.input.early_fir_path, "Early FIR", "Failed to load early reflection FIR.");
    if (!early_fir)
    {
        return early_fir;
    }
    if (early_fir->size() > target_size)
    {
        LOG_ERROR(logger_, "Early FIR must be mono at {} Hz and no longer than the target RIR.", kSampleRate);
        return std::unexpected("invalid length");
    }
    return early_fir;
}

std::expected<fdn_optimization::OptimizationResult, std::string> OptimizationApplication::OptimizeMatching(
    const sfFDN::FDNConfig& initial_config, const std::vector<float>& target_rir,
    const std::vector<float>& early_fir) const
{
    try
    {
        return OptimizeSpectrum(logger_, initial_config, options_.optimizer_params, target_rir, early_fir,
                                std::make_tuple(options_.losses.edc_weight, options_.losses.mel_edr_weight,
                                                options_.losses.weighted_edr_weight),
                                options_.matching, options_.matching_execution, options_.verbose);
    }
    catch (const std::exception& error)
    {
        return std::unexpected("RIR-matching optimization failed: " + std::string(error.what()));
    }
}

bool OptimizationApplication::WriteMatchingOutputs(const fdn_optimization::OptimizationResult& result,
                                                   const std::vector<float>& target_rir,
                                                   const std::vector<float>& early_fir) const
{
    if (!options_.output.spectrum_trajectory_jsonl_path.empty() &&
        !WriteTrajectoryJsonl(result.trajectory, result.loss_names, options_.output.spectrum_trajectory_jsonl_path,
                              logger_))
    {
        return false;
    }
    if (!options_.output.spectrum_result_json_path.empty() &&
        !WriteJsonToFile(
            OptimizationResultToJson(result, options_.optimizer_params, BuildMatchingMetadata(target_rir.size())),
            options_.output.spectrum_result_json_path, logger_))
    {
        return false;
    }
    if (!options_.save_output)
    {
        return true;
    }

    WriteConfigToFile(result.optimized_fdn_config, output_directory_ / "optimized_fdn_config.txt", logger_);
    WriteFilterConfigToFile(result.optimized_fdn_config, output_directory_ / "optimized_filter_config.txt", logger_);
    SaveImpulseResponse(result.initial_fdn_config, target_rir.size(), output_directory_ / "spectrum_initial_ir.wav",
                        logger_, early_fir, options_.matching.early_fir_mode);
    SaveImpulseResponse(result.optimized_fdn_config, target_rir.size(), output_directory_ / "spectrum_optimized_ir.wav",
                        logger_, early_fir, options_.matching.early_fir_mode);
    WriteLossHistoryToFile(result.loss_history, result.loss_names, output_directory_ / "spectrum_loss_history.txt",
                           logger_);
    if (!WriteTargetRirName())
    {
        return false;
    }
    RenderExampleAudio(result.optimized_fdn_config, early_fir);
    return true;
}

bool OptimizationApplication::WriteTargetRirName() const
{
    const std::filesystem::path filename = output_directory_ / "target_rir_name.txt";
    std::ofstream file(filename, std::ios::out);
    if (!file.is_open())
    {
        LOG_ERROR(logger_, "Failed to open file {} for writing target RIR name.", filename.string());
        return false;
    }
    file << options_.input.ir_filename << '\n';
    return file.good();
}

void OptimizationApplication::RenderExampleAudio(const sfFDN::FDNConfig& config,
                                                 const std::vector<float>& early_fir) const
{
    RenderAudio(config, "./audio/drumloop.wav", output_directory_, logger_, early_fir,
                options_.matching.early_fir_mode);
    RenderAudio(config, "./audio/saxophone.wav", output_directory_, logger_, early_fir,
                options_.matching.early_fir_mode);
    RenderAudio(config, "./audio/bleepsandbloops.wav", output_directory_, logger_, early_fir,
                options_.matching.early_fir_mode);
}

void OptimizationApplication::LogStageResult(const char* stage,
                                             const fdn_optimization::OptimizationResult& result) const
{
    LOG_INFO(logger_, "[{}] Final loss: {:.6f}", stage, result.best_loss);
    LOG_INFO(logger_, "[{}] Elapsed time: {:.4f} s", stage, result.total_time.count());
    LOG_INFO(logger_, "[{}] Total evaluations: {}", stage, result.total_evaluations);
    LOG_INFO(logger_, "[{}] Objective evaluations: {}", stage, result.objective_evaluations);
    LOG_INFO(logger_, "[{}] Termination reason: {}", stage, result.termination_reason);
}

nlohmann::json OptimizationApplication::BuildColorlessMetadata() const
{
    const uint32_t actual_fft_size =
        audio_utils::FFT::NextSupportedFFTSize(options_.colorless_execution.spectral_fft_size);
    return {{"stage", "colorless"},
            {"optimizer", options_.selected_optimizer},
            {"fdn_order", fdn_order_},
            {"sample_rate", kSampleRate},
            {"ir_samples", kSampleRate},
            {"seed", options_.colorless_execution.seed},
            {"init_seed", options_.initialization.init_seed.value_or(options_.colorless_execution.seed)},
            {"matrix_parameterization",
             MatrixParameterizationName(options_.initialization.matrix_parameterization)},
            {"gradient_method", GradientMethodName(options_.colorless_execution.gradient_method)},
            {"gradient_threads", options_.colorless_execution.gradient_threads},
            {"optimizer_threads", options_.colorless_execution.optimizer_threads},
            {"max_time_seconds", options_.colorless_execution.max_time_seconds},
            {"max_objective_evaluations", options_.colorless_execution.max_objective_evaluations},
            {"spectral_flatness_target", 0.5575},
            {"spectral_flatness_weight", options_.losses.spectral_flatness_weight},
            {"sparsity_weight", options_.losses.sparsity_weight},
            {"sparsity_window_samples", options_.colorless_execution.sparsity_window_samples},
            {"fft_backend", "unknown"},
            {"requested_fft_size", options_.colorless_execution.spectral_fft_size},
            {"actual_fft_size", actual_fft_size}};
}

nlohmann::json OptimizationApplication::BuildMatchingMetadata(size_t target_size) const
{
    return {{"stage", "spectrum"},
            {"optimizer", options_.selected_optimizer},
            {"matching_filter_type", MatchingFilterTypeName(options_.matching.filter_type)},
            {"fdn_order", fdn_order_},
            {"sample_rate", kSampleRate},
            {"target_rir", options_.input.ir_filename},
            {"target_rir_samples", target_size},
            {"early_fir", options_.input.early_fir_path},
            {"early_fir_mode", EarlyFirModeName(options_.matching.early_fir_mode)},
            {"colorless_config", options_.input.colorless_config_path},
            {"matching_parameterization",
             MatchingParameterizationName(options_.matching.parameter_config.parameterization)},
            {"matching_initialization", MatchingInitializationName(options_.matching.parameter_config.initialization)},
            {"matching_min_t60", options_.matching.parameter_config.minimum_t60},
            {"matching_max_t60", options_.matching.parameter_config.maximum_t60},
            {"matching_tone_scale_db", options_.matching.parameter_config.tone_gain_scale_db},
            {"matching_zero_mean_tone_gains", options_.matching.parameter_config.zero_mean_tone_gains},
            {"seed", options_.matching_execution.seed},
            {"colorless_init_seed",
             options_.input.matching_only
                 ? nlohmann::json(nullptr)
                 : nlohmann::json(options_.initialization.init_seed.value_or(options_.colorless_execution.seed))},
            {"gradient_method", GradientMethodName(options_.matching_execution.gradient_method)},
            {"gradient_threads", options_.matching_execution.gradient_threads},
            {"optimizer_threads", options_.matching_execution.optimizer_threads},
            {"max_time_seconds", options_.matching_execution.max_time_seconds},
            {"max_objective_evaluations", options_.matching_execution.max_objective_evaluations},
            {"fft_backend", "unknown"},
            {"mel_edr_fft_size", options_.matching.fft_length},
            {"mel_edr_hop_size", options_.matching.hop_size},
            {"mel_edr_window_size", options_.matching.window_size},
            {"mel_edr_bands", options_.matching.mel_bands},
            {"edc_weight", options_.losses.edc_weight},
            {"mel_edr_weight", options_.losses.mel_edr_weight},
            {"weighted_edr_weight", options_.losses.weighted_edr_weight}};
}

} // namespace fdn_opt_app
