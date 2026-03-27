#pragma once

#include <audio_utils/audio_analysis.h>
#include <audio_utils/fft.h>

#include <array>
#include <cstdint>
#include <functional>
#include <span>
#include <vector>

namespace fdn_optimization
{

struct LossFunction
{
    std::function<float(std::span<const float>)> func;
    float weight;
    std::string name;
};

float RMS(std::span<const float> signal);

float SpectralFlatnessLoss(std::span<const float> signal);

float RMSLoss(std::span<const float> signal, float target_rms);

float PowerEnvelopeLoss(std::span<const float> signal, uint32_t window_size, uint32_t hop_size, uint32_t sample_rate);

float MixingTimeLoss(std::span<const float> signal, uint32_t sample_rate);

float SparsityLoss(std::span<const float> signal);

struct MultiResolutionSTFTLossOptions
{
    std::vector<uint32_t> fft_sizes = {1024, 2048, 512};
    std::vector<uint32_t> hop_sizes = {120, 240, 50};
    std::vector<uint32_t> window_sizes = {600, 1200, 240};
    audio_utils::FFTWindowType window_type = audio_utils::FFTWindowType::Hann;
    uint32_t sample_rate = 48000;
    bool mel_scale = false;
    uint32_t n_mels = 32;
};

float MultiResolutionSTFTLoss(std::span<const float> signal, std::span<audio_utils::analysis::STFTResult> target_stfts,
                              const MultiResolutionSTFTLossOptions& options);

float EDCLoss(std::span<const float> signal, const std::vector<float>& target_edc);

float EDRLoss(std::span<const float> signal, const audio_utils::analysis::EnergyDecayReliefResult& target_edr,
              const audio_utils::analysis::EnergyDecayReliefOptions& options);

float WeightedEDRLoss(std::span<const float> signal, const audio_utils::analysis::EnergyDecayReliefResult& target_edr,
                      const audio_utils::analysis::EnergyDecayReliefOptions& options);

} // namespace fdn_optimization