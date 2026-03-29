#include "audio_loss.h"

#include <audio_utils/audio_analysis.h>
#include <audio_utils/fft.h>

#include <Eigen/Core>

#include <iostream>
#include <list>
#include <mutex>
#include <queue>

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

static std::queue<std::vector<float>> gVectorPool;
static std::mutex gVectorPoolMutex;
std::vector<float> BorrowVector(size_t size)
{
    std::scoped_lock lock(gVectorPoolMutex);
    if (!gVectorPool.empty())
    {
        auto vec = std::move(gVectorPool.front());
        gVectorPool.pop();

        if (vec.size() != size)
        {
            vec.resize(size, 0.0f);
        }

        return vec;
    }
    else
    {
        return std::vector<float>(size, 0.0f);
    }
}

void ReturnVectorToPool(std::vector<float>&& vec)
{
    std::scoped_lock lock(gVectorPoolMutex);
    gVectorPool.push(std::move(vec));
}

enum class ReductionType
{
    Mean,
    Sum,
    NormalizedMean
};

template <typename Derived>
float L2Loss(const Eigen::ArrayBase<Derived>& signal, const Eigen::ArrayBase<Derived>& target,
             ReductionType reduction = ReductionType::NormalizedMean)
{
    auto diff = (signal - target).square();
    switch (reduction)
    {
    case ReductionType::Mean:
        return diff.mean();
    case ReductionType::Sum:
        return diff.sum();
    case ReductionType::NormalizedMean:
        return diff.sum() / target.square().sum();
    }
}

template <typename Derived>
float L1Loss(const Eigen::ArrayBase<Derived>& signal, const Eigen::ArrayBase<Derived>& target,
             ReductionType reduction = ReductionType::NormalizedMean)
{
    auto loss = (signal - target).abs();
    switch (reduction)
    {
    case ReductionType::Mean:
        return loss.mean();
    case ReductionType::Sum:
        return loss.sum();
    case ReductionType::NormalizedMean:
        return loss.sum() / target.cwiseAbs().sum();
    }
}

} // namespace

namespace fdn_optimization
{
SpectralFlatnessLoss::SpectralFlatnessLoss(float target, float weight)
    : AudioLoss("SpectralFlatness", weight)
    , target_(target)
{
}

float SpectralFlatnessLoss::ComputeLoss(std::span<const float> signal)
{
    uint32_t fft_size = signal.size();
    auto fft_ptr = BorrowFFTForSize(fft_size);

    constexpr audio_utils::ForwardFFTOptions fft_options{
        .output_type = audio_utils::FFTOutputType::Power,
        .to_db = false,
    };

    // std::vector<float> spectrum(fft_ptr->GetSpectrumSize(), 0.0f);
    std::vector<float> spectrum = BorrowVector(fft_ptr->GetSpectrumSize());
    fft_ptr->ForwardMag(signal, std::span(spectrum), fft_options);
    ReturnFFTToPool(std::move(fft_ptr));

    auto flatness = audio_utils::analysis::SpectralFlatness(spectrum);
    ReturnVectorToPool(std::move(spectrum));

    return std::abs(target_ - flatness) * weight_;
}

TimeDomainSparsityLoss::TimeDomainSparsityLoss(float weight)
    : AudioLoss("TimeDomainSparsity", weight)
{
}

float TimeDomainSparsityLoss::ComputeLoss(std::span<const float> signal)
{
    float l1_norm = 0.0f;
    float l2_norm = 0.0f;
    for (const auto& sample : signal)
    {
        l1_norm += std::abs(sample);
        l2_norm += sample * sample;
    }

    l2_norm = std::sqrt(l2_norm);
    return (l2_norm / l1_norm) * weight_;
}

EnergyDecayCurveLoss::EnergyDecayCurveLoss(std::span<const float> target_signal, float weight)
    : AudioLoss("EnergyDecayCurve", weight)
    , target_edc_(audio_utils::analysis::EnergyDecayCurve(target_signal, false))
{
}

float EnergyDecayCurveLoss::ComputeLoss(std::span<const float> signal)
{
    auto edc = audio_utils::analysis::EnergyDecayCurve(signal, false);

    if (edc.size() != target_edc_.size())
    {
        throw std::runtime_error("EDCLoss: EDC result size does not match target size.");
    }

    const Eigen::Map<const Eigen::ArrayXf> edc_vec(edc.data(), edc.size());
    const Eigen::Map<const Eigen::ArrayXf> target_vec(target_edc_.data(), target_edc_.size());

    return L2Loss(edc_vec, target_vec) * weight_;
}

EnergyDecayReliefLoss::EnergyDecayReliefLoss(std::span<const float> target_signal,
                                             const audio_utils::analysis::EnergyDecayReliefOptions& options,
                                             float weight)
    : AudioLoss("EnergyDecayRelief", weight)
    , target_edr_(audio_utils::analysis::EnergyDecayRelief(target_signal, options))
    , options_(options)
{
}

float EnergyDecayReliefLoss::ComputeLoss(std::span<const float> signal)
{
    audio_utils::analysis::EnergyDecayReliefResult edr_result =
        audio_utils::analysis::EnergyDecayRelief(signal, options_);

    if (edr_result.num_bins != target_edr_.num_bins || edr_result.num_frames != target_edr_.num_frames)
    {
        throw std::runtime_error("EDRLoss: EDR result size does not match target size.");
    }

    const Eigen::Map<const Eigen::ArrayXXf> edr_mat(edr_result.data.data(), edr_result.num_bins, edr_result.num_frames);
    const Eigen::Map<const Eigen::ArrayXXf> target_mat(target_edr_.data.data(), target_edr_.num_bins,
                                                       target_edr_.num_frames);

    return L1Loss(edr_mat, target_mat, ReductionType::NormalizedMean) * weight_;
}

WeightedEDRLoss::WeightedEDRLoss(std::span<const float> target_signal,
                                 const audio_utils::analysis::EnergyDecayReliefOptions& options, float min_db,
                                 float weight)
    : AudioLoss("WeightedEDR", weight)
    , target_edr_(audio_utils::analysis::EnergyDecayRelief(target_signal, options))
    , options_(options)
    , min_db_(min_db)
{
}

float WeightedEDRLoss::ComputeLoss(std::span<const float> signal)
{
    audio_utils::analysis::EnergyDecayReliefResult edr_result =
        audio_utils::analysis::EnergyDecayRelief(signal, options_);

    if (edr_result.num_bins != target_edr_.num_bins || edr_result.num_frames != target_edr_.num_frames)
    {
        throw std::runtime_error("WeightedEDRLoss: EDR result size does not match target size.");
    }

    const Eigen::Map<const Eigen::ArrayXXf> edr_mat(edr_result.data.data(), edr_result.num_bins, edr_result.num_frames);
    const Eigen::Map<const Eigen::ArrayXXf> target_mat(target_edr_.data.data(), target_edr_.num_bins,
                                                       target_edr_.num_frames);

    float error = 0.0f;
    for (auto bin = 0; bin < edr_result.num_bins; ++bin)
    {
        float start_db = target_mat(bin, 0);
        const float end_db = start_db - min_db_;

        size_t end_idx = edr_result.num_frames - 1;

        auto end_db_idx = (target_mat.row(bin) <= end_db).template cast<int>();
        if (!end_db_idx.any())
        {
            end_idx = edr_result.num_frames - 1;
        }

        error +=
            L1Loss(edr_mat.row(bin).head(end_idx), target_mat.row(bin).head(end_idx), ReductionType::NormalizedMean);
    }
    return error * weight_;
}

STFTLoss::STFTLoss(std::span<const float> target_signal, const audio_utils::analysis::STFTOptions& options,
                   const STFTLossOptions& loss_options, float weight)
    : AudioLoss("STFT", weight)
    , target_stft_((loss_options.mel_scale)
                       ? audio_utils::analysis::MelSpectrogram(target_signal, options, loss_options.n_mels)
                       : audio_utils::analysis::STFT(target_signal, options))
    , options_(options)
    , loss_options_(loss_options)
{
}

float STFTLoss::SpectralConvergence(const audio_utils::analysis::STFTResult& x,
                                    const audio_utils::analysis::STFTResult& y)
{
    const Eigen::Map<const Eigen::MatrixXf> x_mat(x.data.data(), x.num_bins, x.num_frames);
    const Eigen::Map<const Eigen::MatrixXf> y_mat(y.data.data(), y.num_bins, y.num_frames);

    return (x_mat - y_mat).norm() / x_mat.norm();
}

float STFTLoss::MagnitudeLoss(const audio_utils::analysis::STFTResult& x, const audio_utils::analysis::STFTResult& y,
                              bool log)
{
    const Eigen::Map<const Eigen::ArrayXXf> x_mat(x.data.data(), x.num_bins, x.num_frames);
    const Eigen::Map<const Eigen::ArrayXXf> y_mat(y.data.data(), y.num_bins, y.num_frames);

    if (log)
    {
        return L1Loss(x_mat.log(), y_mat.log(), ReductionType::Mean);
    }

    return L1Loss(x_mat, y_mat, ReductionType::Mean);
}

float STFTLoss::ComputeLoss(std::span<const float> signal)
{
    audio_utils::analysis::STFTResult stft_result;
    if (loss_options_.mel_scale)
    {
        stft_result = audio_utils::analysis::MelSpectrogram(signal, options_, loss_options_.n_mels);
    }
    else
    {
        stft_result = audio_utils::analysis::STFT(signal, options_);
    }

    if (stft_result.num_bins != target_stft_.num_bins || stft_result.num_frames != target_stft_.num_frames)
    {
        throw std::runtime_error("STFTLoss: STFT result size does not match target size.");
    }

    float loss = 0.0f;
    if (loss_options_.spectral_convergence_weight > 0.0f)
    {
        loss += loss_options_.spectral_convergence_weight * SpectralConvergence(stft_result, target_stft_);
    }
    if (loss_options_.log_magnitude_loss_weight > 0.0f)
    {
        loss += loss_options_.log_magnitude_loss_weight * MagnitudeLoss(stft_result, target_stft_, true);
    }
    if (loss_options_.linear_magnitude_loss_weight > 0.0f)
    {
        loss += loss_options_.linear_magnitude_loss_weight * MagnitudeLoss(stft_result, target_stft_, false);
    }

    return loss * weight_;
}
} // namespace fdn_optimization