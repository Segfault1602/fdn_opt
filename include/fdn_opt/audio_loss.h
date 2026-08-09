#pragma once

#include <audio_utils/audio_analysis.h>
#include <audio_utils/fft.h>

#include <span>
#include <string>
#include <vector>

namespace fdn_optimization
{

class AudioLoss
{
  public:
    AudioLoss(std::string name, float weight)
        : weight_(weight)
        , name_(std::move(name))
    {
    }

    AudioLoss(const AudioLoss&) = default;
    AudioLoss(AudioLoss&&) = default;

    AudioLoss& operator=(const AudioLoss&) = default;
    AudioLoss& operator=(AudioLoss&&) = default;

    std::string GetName() const
    {
        return name_;
    }

    virtual ~AudioLoss() = default;
    virtual float ComputeLoss(std::span<const float> signal) const = 0;

  protected:
    float weight_ = 1.0f;

  private:
    std::string name_;
};

/**
 * @brief Spectral flatness loss
 *
 * The spectral flatness loss computes the spectral flatness of the input signal and compares it to a target value. The
 * loss is calculated as the absolute difference between the target and the computed spectral flatness, scaled by a
 * weight factor.
 */
class SpectralFlatnessLoss : public AudioLoss
{
  public:
    SpectralFlatnessLoss(float target, float weight = 1.0f);
    float ComputeLoss(std::span<const float> signal) const override;

  private:
    float target_;
};

/**
 * @brief Time domain sparsity loss
 *
 * The time domain sparsity loss computes the sparsity of the input signal in the time domain. It calculates the L1 and
 * L2 norms of the signal and returns the ratio of the L2 norm to the L1 norm, scaled by a weight factor. A lower ratio
 * indicates a sparser signal.
 */
class TimeDomainSparsityLoss : public AudioLoss
{
  public:
    TimeDomainSparsityLoss(float weight = 1.0f);
    float ComputeLoss(std::span<const float> signal) const override;
};

/**
 * @brief Energy decay curve loss
 *
 * The energy decay curve loss computes the energy decay curve of the input signal and compares it to a target energy
 * decay curve. The loss is calculated as the mean squared error between the target and the computed energy decay curve,
 * scaled by a weight factor.
 */
class EnergyDecayCurveLoss : public AudioLoss
{
  public:
    EnergyDecayCurveLoss(std::span<const float> target_signal, float weight = 1.0f);
    float ComputeLoss(std::span<const float> signal) const override;

  private:
    std::vector<float> target_edc_;
};

/**
 * @brief Energy decay relief loss
 *
 * The energy decay relief loss computes the energy decay relief of the input signal and compares it to a target energy
 * decay relief. The loss is calculated as the mean squared error between the target and the computed energy decay
 * relief, scaled by a weight factor.
 */
class EnergyDecayReliefLoss : public AudioLoss
{
  public:
    EnergyDecayReliefLoss(std::span<const float> target_signal,
                          const audio_utils::analysis::EnergyDecayReliefOptions& options, float weight = 1.0f);
    float ComputeLoss(std::span<const float> signal) const override;

  private:
    audio_utils::analysis::EnergyDecayReliefResult target_edr_;
    audio_utils::analysis::EnergyDecayReliefOptions options_;
};

/**
 * @brief Weighted energy decay relief loss
 *
 * The weighted energy decay relief loss computes the energy decay relief of the input signal and compares it to a
 * target energy decay relief. The loss is calculated as the mean squared error between the target and the computed
 * energy decay relief. Only the portion of the energy decay relief that is above a specified minimum dB threshold is
 * considered in the loss calculation. The loss is scaled by a weight factor.
 */
class WeightedEDRLoss : public AudioLoss
{
  public:
    WeightedEDRLoss(std::span<const float> target_signal,
                    const audio_utils::analysis::EnergyDecayReliefOptions& options, float min_db = -60.f,
                    float weight = 1.0f);
    float ComputeLoss(std::span<const float> signal) const override;

  private:
    audio_utils::analysis::EnergyDecayReliefResult target_edr_;
    audio_utils::analysis::EnergyDecayReliefOptions options_;

    float min_db_;
};

struct STFTLossOptions
{
    float spectral_convergence_weight = 1.0f;
    float log_magnitude_loss_weight = 1.0f;
    float linear_magnitude_loss_weight = 0.0f;
    bool mel_scale = false;
    uint32_t n_mels = 32;
};

/**
 * @brief Short-time Fourier transform loss
 *
 * The STFT loss computes the STFT of the input signal and compares it to a target STFT. The loss is calculated as a
 * combination of spectral convergence, log-magnitude loss, and linear-magnitude loss, each scaled by their respective
 * weight factors.
 */
class STFTLoss : public AudioLoss
{
  public:
    STFTLoss(std::span<const float> target_signal, const audio_utils::analysis::STFTOptions& options,
             const STFTLossOptions& loss_options = {}, float weight = 1.0f);
    float ComputeLoss(std::span<const float> signal) const override;

    static float SpectralConvergence(const audio_utils::analysis::STFTResult& x,
                                     const audio_utils::analysis::STFTResult& y);

    static float MagnitudeLoss(const audio_utils::analysis::STFTResult& x, const audio_utils::analysis::STFTResult& y,
                               bool log);

  private:
    audio_utils::analysis::STFTResult target_stft_;
    audio_utils::analysis::STFTOptions options_;
    STFTLossOptions loss_options_;
};

} // namespace fdn_optimization