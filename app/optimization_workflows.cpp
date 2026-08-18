#include "optimization_workflows.h"

#include "model.h"
#include "utils.h"

#include <armadillo>
#include <audio_utils/audio_analysis.h>
#include <audio_utils/audio_file_manager.h>
#include <quill/LogMacros.h>

#include <array>
#include <chrono>
#include <cmath>
#include <iostream>
#include <thread>
#include <vector>

namespace fdn_opt_app
{

sfFDN::FDNConfig CreateInitialFDNConfig(uint32_t fdn_order, bool randomize, bool random_delays, uint32_t seed)
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
            {
                last_loss = static_cast<float>(progress.loss_history.front().back());
            }

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

    fdn_optimization::OptimizationInfo opt_info{.parameters_to_optimize = params_to_optimize,
                                                .initial_fdn_config = initial_fdn_config,
                                                .ir_size = kSampleRate,
                                                .gradient_method = execution_options.gradient_method,
                                                .target_rir = {},
                                                .early_fir = {},
                                                .t60_estimates = {},
                                                .early_fir_mode = fdn_optimization::EarlyFirMode::DirectPath,
                                                .optimizer_params = optimizer_params,
                                                .seed = execution_options.seed,
                                                .gradient_threads = execution_options.gradient_threads,
                                                .optimizer_threads = execution_options.optimizer_threads,
                                                .max_time_seconds = execution_options.max_time_seconds,
                                                .max_objective_evaluations =
                                                    execution_options.max_objective_evaluations,
                                                .record_trajectory = execution_options.record_trajectory,
                                                .matching_parameters = {}};

    fdn_optimization::FDNOptimizer optimizer(logger, verbose);

    optimizer.SetLossFunctions(loss_functions);
    optimizer.StartOptimization(opt_info);

    if (WaitForOptimization(optimizer, logger) != fdn_optimization::OptimizationStatus::Completed)
    {
        throw std::runtime_error("Colorless optimization did not complete successfully.");
    }

    auto result = optimizer.GetResult();
    return result;
}

std::vector<float> EstimateMatchingT60s(const std::vector<float>& target_rir, MatchingFilterType filter_type)
{
    auto decay_curves = audio_utils::analysis::EnergyDecayCurve_FilterBank(target_rir, true, kSampleRate);
    std::vector<float> time(target_rir.size());
    for (size_t index = 0; index < time.size(); ++index)
    {
        time[index] = static_cast<float>(index) / static_cast<float>(kSampleRate);
    }

    std::array<float, audio_utils::analysis::kNumOctaveBands> estimates{};
    for (size_t band = 0; band < decay_curves.size(); ++band)
    {
        const auto result = audio_utils::analysis::EstimateT60(
            decay_curves[band], time, {.decay_start_db = -5.0f, .decay_end_db = -25.0f, .use_linear_regression = true});
        estimates[band] = result.t60 > 0.0f && std::isfinite(result.t60) ? result.t60 : 1.0f;
    }

    if (filter_type == MatchingFilterType::TenBand)
    {
        std::vector<float> result;
        result.reserve(10);
        result.push_back(estimates.front());
        result.insert(result.end(), estimates.begin(), estimates.end());
        return result;
    }

    auto mean_range = [&estimates](size_t begin, size_t end) {
        float sum = 0.0f;
        for (size_t index = begin; index < end; ++index)
        {
            sum += estimates[index];
        }
        return sum / static_cast<float>(end - begin);
    };
    return {mean_range(0, 4), mean_range(4, 7), mean_range(7, 9)};
}

fdn_optimization::OptimizationResult OptimizeSpectrum(quill::Logger* logger, const sfFDN::FDNConfig& initial_fdn_config,
                                                      const fdn_optimization::OptimizationAlgoParams& optimizer_params,
                                                      const std::vector<float>& target_rir,
                                                      const std::vector<float>& early_fir,
                                                      const std::tuple<double, double, double>& loss_weights,
                                                      const MatchingAnalysisOptions& matching_options,
                                                      const ExecutionOptions& execution_options, bool verbose)
{
    const auto attenuation_type = matching_options.filter_type == MatchingFilterType::TenBand
                                      ? fdn_optimization::OptimizationParamType::AttenuationFilters
                                      : fdn_optimization::OptimizationParamType::AttenuationFilters_3Band;
    std::vector params_to_optimize = {attenuation_type, fdn_optimization::OptimizationParamType::TonecorrectionFilters,
                                      fdn_optimization::OptimizationParamType::OverallGain};
    std::vector<float> t60_estimates;
    if (matching_options.parameter_config.initialization == fdn_optimization::MatchingInitialization::TargetDerived)
    {
        t60_estimates = EstimateMatchingT60s(target_rir, matching_options.filter_type);
    }

    fdn_optimization::OptimizationInfo opt_info{.parameters_to_optimize = params_to_optimize,
                                                .initial_fdn_config = initial_fdn_config,
                                                .ir_size = static_cast<uint32_t>(target_rir.size()),
                                                .gradient_method = execution_options.gradient_method,
                                                .target_rir = target_rir,
                                                .early_fir = early_fir,
                                                .t60_estimates = std::move(t60_estimates),
                                                .early_fir_mode = matching_options.early_fir_mode,
                                                .optimizer_params = optimizer_params,
                                                .seed = execution_options.seed,
                                                .gradient_threads = execution_options.gradient_threads,
                                                .optimizer_threads = execution_options.optimizer_threads,
                                                .max_time_seconds = execution_options.max_time_seconds,
                                                .max_objective_evaluations =
                                                    execution_options.max_objective_evaluations,
                                                .record_trajectory = execution_options.record_trajectory,
                                                .matching_parameters = matching_options.parameter_config};

    fdn_optimization::FDNOptimizer optimizer(logger, verbose);

    std::vector<std::shared_ptr<fdn_optimization::AudioLoss>> loss_functions;

    const double edc_weight = std::get<0>(loss_weights);
    if (edc_weight > 0.0)
    {
        loss_functions.push_back(std::make_shared<fdn_optimization::EnergyDecayCurveLoss>(target_rir, edc_weight));
    }

    const double mel_edr_weight = std::get<1>(loss_weights);
    if (mel_edr_weight > 0.0)
    {
        audio_utils::analysis::EnergyDecayReliefOptions edr_options{.fft_length = matching_options.fft_length,
                                                                    .hop_size = matching_options.hop_size,
                                                                    .window_size = matching_options.window_size,
                                                                    .window_type = audio_utils::FFTWindowType::Hann,
                                                                    .n_mels = matching_options.mel_bands,
                                                                    .to_db = true};
        loss_functions.push_back(
            std::make_shared<fdn_optimization::EnergyDecayReliefLoss>(target_rir, edr_options, mel_edr_weight));
    }

    const double weighted_edr_weight = std::get<2>(loss_weights);
    if (weighted_edr_weight > 0.0)
    {
        audio_utils::analysis::EnergyDecayReliefOptions edr_options{.fft_length = matching_options.fft_length,
                                                                    .hop_size = matching_options.hop_size,
                                                                    .window_size = matching_options.window_size,
                                                                    .window_type = audio_utils::FFTWindowType::Hann,
                                                                    .n_mels = matching_options.mel_bands,
                                                                    .to_db = true};
        loss_functions.push_back(
            std::make_shared<fdn_optimization::WeightedEDRLoss>(target_rir, edr_options, -20.0f, weighted_edr_weight));
    }

    optimizer.SetLossFunctions(loss_functions);
    optimizer.StartOptimization(opt_info);

    if (WaitForOptimization(optimizer, logger) != fdn_optimization::OptimizationStatus::Completed)
    {
        throw std::runtime_error("Spectrum optimization did not complete successfully.");
    }

    auto result = optimizer.GetResult();
    return result;
}

void RenderAudio(const sfFDN::FDNConfig& fdn_config, const std::string& input_filename,
                 const std::filesystem::path& output_dir, quill::Logger* logger, std::span<const float> early_fir,
                 fdn_optimization::EarlyFirMode early_fir_mode)
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

    std::vector<float> fdn_input = audio_file;
    if (!early_fir.empty() && early_fir_mode == fdn_optimization::EarlyFirMode::Excitation)
    {
        fdn_input = audio_utils::analysis::Convolve(audio_file, early_fir);
        fdn_input.resize(audio_file.size());
    }

    auto fdn = sfFDN::CreateFDNFromConfig(fdn_config);
    fdn->SetDirectGain(0.0f);

    std::vector<float> output_audio(audio_file.size(), 0.0f);
    sfFDN::AudioBuffer input_buffer(fdn_input);
    sfFDN::AudioBuffer output_buffer(output_audio);

    fdn->Process(input_buffer, output_buffer);

    if (!early_fir.empty() && early_fir_mode == fdn_optimization::EarlyFirMode::DirectPath)
    {
        auto direct_audio = audio_utils::analysis::Convolve(audio_file, early_fir);
        direct_audio.resize(output_audio.size());
        for (size_t index = 0; index < output_audio.size(); ++index)
        {
            output_audio[index] += direct_audio[index];
        }
    }

    std::filesystem::path output_path =
        output_dir / (std::filesystem::path(input_filename).stem().string() + "_wet.wav");
    audio_utils::audio_file::WriteWavFile(output_path.string(), output_audio, kSampleRate);
}

} // namespace fdn_opt_app
