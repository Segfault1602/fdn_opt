#include "initialization.h"

#include <audio_utils/audio_analysis.h>

#include <armadillo>

#include <array>
#include <cmath>
#include <cstddef>
#include <random>
#include <stdexcept>

namespace fdn_optimization
{

RandomGainPair GenerateRandomNormalizedGains(std::uint32_t order, std::uint32_t seed)
{
    if (order == 0)
        throw std::invalid_argument("Random gain order must be positive.");

    std::mt19937 generator(seed);
    std::normal_distribution<float> gaussian(0.0f, 1.0f);
    arma::fvec input(order);
    arma::fvec output(order);
    for (std::uint32_t index = 0; index < order; ++index)
    {
        input(index) = gaussian(generator);
        output(index) = gaussian(generator);
    }
    input /= arma::norm(input, 2);
    output /= arma::norm(output, 2);
    return {
        .input = std::vector<float>(input.begin(), input.end()),
        .output = std::vector<float>(output.begin(), output.end()),
    };
}

std::vector<float> EstimateMatchingT60s(std::span<const float> target_rir, OptimizationParamType attenuation_type,
                                        float sample_rate)
{
    if (target_rir.empty() || sample_rate <= 0.0f)
    {
        return {};
    }

    const auto decay_curves = audio_utils::analysis::EnergyDecayCurve_FilterBank(target_rir, true, sample_rate);

    std::vector<float> time(target_rir.size());
    for (std::size_t index = 0; index < time.size(); ++index)
    {
        time[index] = static_cast<float>(index) / sample_rate;
    }

    std::array<float, audio_utils::analysis::kNumOctaveBands> estimates{};
    for (std::size_t band = 0; band < decay_curves.size(); ++band)
    {
        const auto result = audio_utils::analysis::EstimateT60(
            decay_curves[band], time, {.decay_start_db = -5.0f, .decay_end_db = -25.0f, .use_linear_regression = true});
        // A band with no usable decay yields a non-finite or negative T60. Feeding that into
        // MapFromT60 would poison the optimizer's starting point, so fall back to one second.
        estimates[band] = result.t60 > 0.0f && std::isfinite(result.t60) ? result.t60 : 1.0f;
    }

    if (attenuation_type == OptimizationParamType::AttenuationFilters)
    {
        // The ten-band filter has one more band than the octave analysis, so the lowest estimate is
        // repeated to cover it.
        std::vector<float> result;
        result.reserve(estimates.size() + 1);
        result.push_back(estimates.front());
        result.insert(result.end(), estimates.begin(), estimates.end());
        return result;
    }

    const auto mean_range = [&estimates](std::size_t begin, std::size_t end) {
        float sum = 0.0f;
        for (std::size_t index = begin; index < end; ++index)
        {
            sum += estimates[index];
        }
        return sum / static_cast<float>(end - begin);
    };
    return {mean_range(0, 4), mean_range(4, 7), mean_range(7, 9)};
}

} // namespace fdn_optimization
