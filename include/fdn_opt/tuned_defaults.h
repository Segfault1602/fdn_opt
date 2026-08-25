#pragma once

#include "optim_types.h"
#include "optimizer.h"

#include <cstdint>

namespace fdn_optimization
{

// Selects which tuning campaign a set of defaults comes from. The two campaigns optimize different
// objectives and their tuned hyperparameters differ by orders of magnitude for some algorithms, so
// a single objective-agnostic default cannot serve both.
enum class OptimizationObjective : uint8_t
{
    // Colorless N=8 campaign: research-defined NSGA-II studies, thesis init, 4 threads,
    // 30 trials x 4 tune cases, accepted only after 12 disjoint held-out validate cases.
    Colorless,
    // Concert-hall RIR-matching campaign: N=8, thesis-faithful 3-band matching, random RT60
    // initialization, direct-path early FIR, 4 threads, accepted after 12 held-out seeds.
    RirMatching,
};

constexpr const char* OptimizationObjectiveToString(OptimizationObjective objective)
{
    return objective == OptimizationObjective::RirMatching ? "RIR Matching" : "Colorless";
}

// Objective loss weights shared by every algorithm within one campaign.
struct TunedLossDefaults
{
    double spectral_flatness_weight = 1.0;
    double sparsity_weight = 0.5;
    double power_envelope_weight = 0.0;
    double edc_weight = 0.1;
    double mel_edr_weight = 1.0;
    double weighted_edr_weight = 0.0;
};

// Model settings that only apply to the RIR-matching campaign.
struct TunedMatchingSettings
{
    // Either OptimizationParamType::AttenuationFilters or AttenuationFilters_3Band.
    OptimizationParamType attenuation_filter_type = OptimizationParamType::AttenuationFilters_3Band;
    MatchingInitialization initialization = MatchingInitialization::SeededRandom;
};

// Returns the tuned hyperparameters for one algorithm under one objective.
//
// The returned variant always holds the alternative that corresponds to `algorithm`; callers can
// rely on that to avoid the silent-fallthrough class of bug where an unhandled algorithm yields a
// default-constructed variant holding an unrelated alternative.
//
// Keys absent from a campaign's configuration keep the struct initializer from `optimizer.h`, so
// there remains exactly one definition of an untuned value.
OptimizationAlgoParams TunedDefaultsFor(OptimizationAlgoType algorithm, OptimizationObjective objective);

// Returns the tuned objective weights for one campaign.
TunedLossDefaults TunedLossDefaultsFor(OptimizationObjective objective);

// Returns the tuned RIR-matching model settings.
TunedMatchingSettings TunedMatchingDefaults();

} // namespace fdn_optimization
