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

class SpectralFlatnessLoss : public AudioLoss
{
  public:
    SpectralFlatnessLoss(float target, float weight = 1.0f);
    float ComputeLoss(std::span<const float> signal) const override;

  private:
    float target_;
};

class TimeDomainSparsityLoss : public AudioLoss
{
  public:
    TimeDomainSparsityLoss(float weight = 1.0f);
    float ComputeLoss(std::span<const float> signal) const override;
};

class EnergyDecayCurveLoss : public AudioLoss
{
  public:
    EnergyDecayCurveLoss(std::span<const float> target_signal, float weight = 1.0f);
    float ComputeLoss(std::span<const float> signal) const override;

  private:
    std::vector<float> target_edc_;
};

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