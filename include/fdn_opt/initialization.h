#pragma once

#include "optim_types.h"

#include <cstdint>
#include <span>
#include <vector>

namespace fdn_optimization
{

struct RandomGainPair
{
    std::vector<float> input;
    std::vector<float> output;
};

RandomGainPair GenerateRandomNormalizedGains(std::uint32_t order, std::uint32_t seed);

/// Estimates per-band T60s from a target RIR for use as matching starting values.
///
/// Returns one estimate per band of the requested attenuation filter, ready to be assigned to
/// `OptimizationInfo::t60_estimates`. Only meaningful together with
/// `MatchingInitialization::TargetDerived`, which is what consumes them.
///
/// `attenuation_type` selects the band layout: `AttenuationFilters` yields ten values and
/// `AttenuationFilters_3Band` yields three. Any other value yields the three-band layout.
///
/// Bands whose decay cannot be estimated fall back to one second rather than propagating a
/// non-finite or negative T60 into the optimizer's initial parameters.
std::vector<float> EstimateMatchingT60s(std::span<const float> target_rir, OptimizationParamType attenuation_type,
                                        float sample_rate);

} // namespace fdn_optimization
