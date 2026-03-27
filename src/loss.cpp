#include "loss.h"

#include <audio_utils/audio_analysis.h>
#include <audio_utils/fft.h>

#include <armadillo>

#include <cassert>
#include <list>
#include <mutex>
#include <ranges>

namespace
{
static std::list<std::unique_ptr<audio_utils::FFT>> gFFTPool;
static std::mutex gFFTPoolMutex;
std::unique_ptr<audio_utils::FFT> BorrowFFTForSize(uint32_t size)
{
    const uint32_t fft_size = audio_utils::FFT::NextSupportedFFTSize(size);

    std::scoped_lock lock(gFFTPoolMutex);
    for (auto it = gFFTPool.begin(); it != gFFTPool.end(); ++it)
    {
        if ((*it)->GetFFTSize() == fft_size)
        {
            auto fft = std::move(*it);
            gFFTPool.erase(it);
            return fft;
        }
    }

    auto fft = std::make_unique<audio_utils::FFT>(fft_size);
    return fft;
}

void ReturnFFTToPool(std::unique_ptr<audio_utils::FFT> fft)
{
    std::scoped_lock lock(gFFTPoolMutex);
    gFFTPool.push_back(std::move(fft));
}

enum class ReductionType
{
    Mean,
    Sum,
    NormalizedMean
};

template <typename T>
float L1Loss(const T& signal, const T& target, ReductionType reduction = ReductionType::NormalizedMean)
{
    auto loss = arma::abs(signal - target);
    switch (reduction)
    {
    case ReductionType::Mean:
        return arma::mean(loss);
    case ReductionType::Sum:
        return arma::accu(loss);
    case ReductionType::NormalizedMean:
        return arma::accu(loss) / arma::accu(arma::abs(target));
    }
    // return arma::accu(arma::abs(signal - target)) / arma::accu(arma::abs(target));
}

template <typename T>
float L2Loss(const T& signal, const T& target)
{
    return arma::accu(arma::square(signal - target)) / arma::accu(arma::square(target));
}

} // namespace

namespace fdn_optimization
{

float RMS(std::span<const float> signal)
{
    float sum_squares = 0.0f;
    for (const auto& sample : signal)
    {
        sum_squares += sample * sample;
    }

    float rms = std::sqrt(sum_squares / static_cast<float>(signal.size()));
    return rms;
}

float SpectralFlatnessLoss(std::span<const float> signal)
{
    // uint32_t fft_size = std::max(static_cast<uint32_t>(signal.size()), static_cast<uint32_t>(48000));
    uint32_t fft_size = signal.size();
    auto fft_ptr = BorrowFFTForSize(fft_size);

    std::vector<float> spectrum((fft_ptr->GetFFTSize() / 2) + 1, 0.0f);
    fft_ptr->ForwardMag(signal, std::span(spectrum),
                        {.output_type = audio_utils::FFTOutputType::Power, .to_db = false});
    ReturnFFTToPool(std::move(fft_ptr));

    float flatness = audio_utils::analysis::SpectralFlatness(spectrum);
    // return 1.f - flatness;
    // return std::abs(flatness - 0.55f);
    return flatness;
}

float RMSLoss(std::span<const float> signal, float target_rms)
{
    float rms = RMS(signal);
    return std::abs(rms - target_rms);
}

float PowerEnvelopeLoss(std::span<const float> signal, uint32_t window_size, uint32_t hop_size, uint32_t sample_rate)
{
    // Compute the power envelope of the signal
    std::vector<float> power_envelope;
    const size_t num_frames = (signal.size() - window_size) / hop_size + 1;
    power_envelope.reserve(num_frames);

    for (size_t i = 0; i < signal.size(); i += hop_size)
    {
        if (i + window_size > signal.size())
        {
            break;
        }

        std::span<const float> frame = signal.subspan(i, window_size);
        const float frame_rms = RMS(frame);
        power_envelope.push_back(frame_rms); // Power is RMS squared
    }

    // Find maximum index in power envelope
    auto max_it = std::max_element(power_envelope.begin(), power_envelope.end());
    if (max_it != power_envelope.end())
    {
        size_t max_index = std::distance(power_envelope.begin(), max_it);
        float time_seconds = static_cast<float>(max_index * hop_size) / static_cast<float>(sample_rate);
        return time_seconds;
    }

    assert(false); // Should not reach here
    return 1.f;
}

float MixingTimeLoss(std::span<const float> signal, uint32_t sample_rate)
{
    constexpr float kWindowSizeMs = 20.0f;
    constexpr float kHopSizeMs = 5.0f;
    const uint32_t window_size = static_cast<uint32_t>((kWindowSizeMs / 1000.0f) * static_cast<float>(sample_rate));
    const uint32_t hop_size = static_cast<uint32_t>((kHopSizeMs / 1000.0f) * static_cast<float>(sample_rate));

    audio_utils::analysis::EchoDensityOptions options;
    options.window_size = window_size;
    options.hop_size = hop_size;
    options.sample_rate = sample_rate;

    auto results = audio_utils::analysis::EchoDensity(signal, options);

    const arma::fvec sparse_echo_density(results.echo_densities.data(), results.echo_densities.size(), false, true);
    arma::fvec sparse_indices(results.sparse_indices.size());

    for (size_t i = 0; i < results.sparse_indices.size(); ++i)
    {
        sparse_indices(i) = static_cast<float>(results.sparse_indices[i]);
    }

    arma::fvec time_index = arma::regspace<arma::fvec>(0.0f, signal.size() - 1);
    arma::fvec echo_density;
    arma::interp1(sparse_indices, sparse_echo_density, time_index, echo_density);

    constexpr float kEchoDensityThreshold = 0.9f;
    arma::uvec above_threshold_indices = arma::find(echo_density >= kEchoDensityThreshold, 1, "first");

    if (above_threshold_indices.empty())
    {
        // If the echo density never reaches the threshold, return a large loss
        return 2.0f - echo_density.max(); // Encourage higher echo density
    }

    float mixing_time = static_cast<float>(above_threshold_indices(0)) / static_cast<float>(sample_rate);

    return mixing_time;
}

float SparsityLoss(std::span<const float> signal)
{
    // Time domain sparsity loss from [1] G. Dal Santo, K. Prawda, S. J. Schlecht, and V. Välimäki, “Differentiable
    // Feedback Delay Network for Colorless Reverberation,” in Proceedings of the 26th International Conference on
    // Digital Audio Effects (DAFx23), Sept. 2023.

    float l1_norm = 0.0f;
    float l2_norm = 0.0f;
    for (const auto& sample : signal)
    {
        l1_norm += std::abs(sample);
        l2_norm += sample * sample;
    }

    l2_norm = std::sqrt(l2_norm);
    return l2_norm / l1_norm;
}

float MultiResolutionSTFTLoss(std::span<const float> signal, std::span<audio_utils::analysis::STFTResult> target_stfts,
                              const MultiResolutionSTFTLossOptions& options)
{
    float total_loss = 0.0f;

    [[maybe_unused]] const uint32_t num_combination = options.fft_sizes.size();
    assert(target_stfts.size() == options.fft_sizes.size());
    assert(options.fft_sizes.size() == num_combination);
    assert(options.hop_sizes.size() == num_combination);
    assert(options.window_sizes.size() == num_combination);

    for (auto [fft_size, hop_size, window_size, target_stft] :
         std::views::zip(options.fft_sizes, options.hop_sizes, options.window_sizes, target_stfts))
    {
        audio_utils::analysis::STFTOptions stft_options;
        stft_options.fft_size = fft_size;
        stft_options.overlap = window_size - hop_size;
        stft_options.window_size = window_size;
        stft_options.window_type = options.window_type;

        audio_utils::analysis::STFTResult stft_result;
        if (options.mel_scale)
        {
            stft_result = audio_utils::analysis::MelSpectrogram(signal, stft_options, options.n_mels);
        }
        else
        {
            stft_result = audio_utils::analysis::STFT(signal, stft_options);
        }

        if (stft_result.num_bins != target_stft.num_bins || stft_result.num_frames != target_stft.num_frames)
        {
            throw std::runtime_error("MultiResolutionSTFTLoss: STFT result size does not match target size.");
        }

        const arma::fmat stft_mat(stft_result.data.data(), stft_result.num_bins, stft_result.num_frames);
        const arma::fmat target_mat(const_cast<float*>(target_stft.data.data()), target_stft.num_bins,
                                    target_stft.num_frames);

        float loss = L1Loss(stft_mat.as_col(), target_mat.as_col());
        total_loss += loss;
    }
    return total_loss;
}

float EDCLoss(std::span<const float> signal, const std::vector<float>& target_edc)
{
    auto edc = audio_utils::analysis::EnergyDecayCurve(signal, false);

    if (edc.size() != target_edc.size())
    {
        throw std::runtime_error("EDCLoss: EDC result size does not match target size.");
    }

    const arma::fvec edc_vec(const_cast<float*>(edc.data()), edc.size(), false, true);
    const arma::fvec target_vec(const_cast<float*>(target_edc.data()), target_edc.size(), false, true);

    return L2Loss(edc_vec, target_vec);
}

float EDRLoss(std::span<const float> signal, const audio_utils::analysis::EnergyDecayReliefResult& target_edr,
              const audio_utils::analysis::EnergyDecayReliefOptions& options)
{
    audio_utils::analysis::EnergyDecayReliefResult edr_result =
        audio_utils::analysis::EnergyDecayRelief(signal, options);

    if (edr_result.num_bins != target_edr.num_bins || edr_result.num_frames != target_edr.num_frames)
    {
        throw std::runtime_error("EDRLoss: EDR result size does not match target size.");
    }

    const arma::fmat edr_mat(edr_result.data.data(), edr_result.num_bins, edr_result.num_frames);
    const arma::fmat target_mat(const_cast<float*>(target_edr.data.data()), target_edr.num_bins, target_edr.num_frames);

    // Mezza et al. loss
    return L1Loss(edr_mat.as_col(), target_mat.as_col());

    // St-Onge loss
    // float loss = arma::mean(arma::square(edr_vec - target_vec));
    // return std::sqrt(loss);
}

float WeightedEDRLoss(std::span<const float> signal, const audio_utils::analysis::EnergyDecayReliefResult& target_edr,
                      const audio_utils::analysis::EnergyDecayReliefOptions& options)
{
    audio_utils::analysis::EnergyDecayReliefResult edr_result =
        audio_utils::analysis::EnergyDecayRelief(signal, options);

    if (edr_result.num_bins != target_edr.num_bins || edr_result.num_frames != target_edr.num_frames)
    {
        throw std::runtime_error("WeightedEDRLoss: EDR result size does not match target size.");
    }

    const arma::fmat edr_mat(edr_result.data.data(), edr_result.num_bins, edr_result.num_frames);
    const arma::fmat target_mat(const_cast<float*>(target_edr.data.data()), target_edr.num_bins, target_edr.num_frames);

    float error = 0.0f;
    for (auto bin = 0; bin < edr_result.num_bins; ++bin)
    {
        float start_db = target_mat(bin, 0);
        const float end_db = start_db - 25.f; // Only weight the first 10 dB of decay

        size_t end_idx = edr_result.num_frames - 1;
        arma::uvec end_db_idx = arma::find(target_mat.row(bin) <= end_db, 1, "first");
        if (!end_db_idx.empty())
        {
            end_idx = end_db_idx(0);
        }

        // auto diff = arma::sum(arma::abs(edr_mat.row(bin).subvec(0, end_idx) - target_mat.row(bin).subvec(0,
        // end_idx))); auto sum = arma::sum(arma::abs(target_mat.row(bin).subvec(0, end_idx))); error += diff / sum;
        error += L1Loss(edr_mat.row(bin).subvec(0, end_idx), target_mat.row(bin).subvec(0, end_idx));
    }

    return error;
}

} // namespace fdn_optimization