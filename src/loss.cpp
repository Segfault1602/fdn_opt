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

float L1Loss(std::span<const float> signal, std::span<const float> target)
{
    if (signal.size() != target.size())
    {
        throw std::runtime_error("L1Loss: Signal and target must have the same size.");
    }

    for (size_t i = 0; i < signal.size(); ++i)
    {
        if (std::isnan(signal[i]) || std::isinf(signal[i]))
        {
            throw std::runtime_error("L1Loss: Signal contains NaN or Inf values.");
        }

        if (std::isnan(target[i]) || std::isinf(target[i]))
        {
            throw std::runtime_error("L1Loss: Target contains NaN or Inf values.");
        }
    }

    const arma::fvec signal_vec(const_cast<float*>(signal.data()), signal.size(), false, true);
    const arma::fvec target_vec(const_cast<float*>(target.data()), target.size(), false, true);

    auto diff = arma::sum(arma::abs(signal_vec - target_vec));
    auto target_sum = arma::sum(arma::abs(target_vec));

    assert(!std::isnan(diff));
    assert(!std::isinf(diff));

    assert(!std::isnan(target_sum));
    assert(!std::isinf(target_sum));

    return diff / target_sum;

    // return arma::sum(arma::abs(signal_vec - target_vec)) / arma::sum(arma::abs(target_vec));
}

float L2Loss(std::span<const float> signal, std::span<const float> target)
{
    if (signal.size() != target.size())
    {
        throw std::runtime_error("L2Loss: Signal and target must have the same size.");
    }

    const arma::fvec signal_vec(const_cast<float*>(signal.data()), signal.size(), false, true);
    const arma::fvec target_vec(const_cast<float*>(target.data()), target.size(), false, true);

    return arma::sum(arma::square(signal_vec - target_vec)) / arma::sum(arma::square(target_vec));
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
    uint32_t fft_size = std::max(static_cast<uint32_t>(signal.size()), static_cast<uint32_t>(48000));
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

float MixingTimeLoss(std::span<const float> signal, uint32_t sample_rate, float target_mixing_time)
{
    constexpr float kWindowSizeMs = 50.0f;
    constexpr float kHopSizeMs = 10.0f;
    const uint32_t window_size = static_cast<uint32_t>((kWindowSizeMs / 1000.0f) * static_cast<float>(sample_rate));
    const uint32_t hop_size = static_cast<uint32_t>((kHopSizeMs / 1000.0f) * static_cast<float>(sample_rate));

    audio_utils::analysis::EchoDensityOptions options;
    options.window_size = window_size;
    options.hop_size = hop_size;
    options.sample_rate = sample_rate;

    auto results = audio_utils::analysis::EchoDensity(signal, options);

    return std::abs(results.mixing_time - target_mixing_time);
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

// float EDCLoss(std::span<const float> signal,
//               const std::array<std::vector<float>, audio_utils::analysis::kNumOctaveBands>& target_relief)
// {
//     std::array<std::vector<float>, audio_utils::analysis::kNumOctaveBands> edc_result =
//         audio_utils::analysis::EnergyDecayCurve_FilterBank(signal, false);

//     float loss = 0.0f;

//     constexpr float kDropLastPercent = 0.50f; // Drop last % of EDC to avoid tail artifacts

//     for (auto&& [decay_curve, target_curve] : std::views::zip(edc_result, target_relief))
//     {
//         const size_t min_size = std::min(decay_curve.size(), target_curve.size());
//         const size_t analysis_size = static_cast<size_t>(min_size * (1.0f - kDropLastPercent));

//         auto decay_curve_span = std::span<float>(decay_curve).subspan(0, analysis_size);
//         auto target_curve_span = std::span<const float>(target_curve).subspan(0, analysis_size);

//         const arma::fvec decay_vec(decay_curve_span.data(), decay_curve_span.size(), false, true);
//         const arma::fvec target_vec(const_cast<float*>(target_curve_span.data()), target_curve_span.size(), false,
//                                     true);

//         float curve_loss = arma::mean(arma::square(decay_vec - target_vec));
//         loss += curve_loss;
//     }

//     return std::sqrt(loss);
// }

float EDCLoss(std::span<const float> signal, const std::vector<float>& target_edc)
{
    auto edc = audio_utils::analysis::EnergyDecayCurve(signal, false);

    if (edc.size() != target_edc.size())
    {
        throw std::runtime_error("EDCLoss: EDC result size does not match target size.");
    }

    return L2Loss(edc, target_edc);
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

    // Mezza et al. loss
    return L1Loss(edr_result.data, target_edr.data);

    // St-Onge loss
    // float loss = arma::mean(arma::square(edr_vec - target_vec));
    // return std::sqrt(loss);
}
} // namespace fdn_optimization