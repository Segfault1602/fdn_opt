#pragma once

#include <algorithm>
#include <cstdint>
#include <string>
#include <thread>

namespace fdn_optimization
{
inline uint32_t DefaultGradientThreadCount()
{
    const uint32_t hardware_threads = std::thread::hardware_concurrency();
    return hardware_threads == 0 ? 1 : std::min(4u, hardware_threads);
}

enum class OptimizationParamType : uint8_t
{
    Gains,
    Matrix,
    Matrix_Householder,
    Matrix_Circulant,
    AttenuationFilters,
    AttenuationFilters_3Band,
    TonecorrectionFilters,
    OverallGain,
};

constexpr std::string OptimizationParamTypeToString(OptimizationParamType type)
{
    switch (type)
    {
    case OptimizationParamType::Gains:
        return "Gains";
    case OptimizationParamType::Matrix:
        return "Matrix";
    case OptimizationParamType::Matrix_Householder:
        return "Matrix_Householder";
    case OptimizationParamType::Matrix_Circulant:
        return "Matrix_Circulant";
    case OptimizationParamType::AttenuationFilters:
        return "AttenuationFilters";
    case OptimizationParamType::AttenuationFilters_3Band:
        return "AttenuationFilters_3Band";
    case OptimizationParamType::TonecorrectionFilters:
        return "TonecorrectionFilters";
    case OptimizationParamType::OverallGain:
        return "OverallGain";
    default:
        return "Unknown";
    }
}

enum class GradientMethod : uint8_t
{
    CentralDifferences,
    ForwardDifferences,
};

// Controls whether the early FIR is added directly or used to excite the FDN.
enum class EarlyFirMode : uint8_t
{
    // Filter the signal entering the FDN; retained for legacy comparisons.
    Excitation,
    // Add the early FIR alongside the FDN late-reverberation output.
    DirectPath,
};

// Defines how unconstrained optimizer coordinates map to physical matching parameters.
enum class MatchingParameterization : uint8_t
{
    // Optimize physical values directly and clamp them to valid bounds.
    RawClamped,
    // Map unconstrained coordinates smoothly to bounded physical values.
    ScaledSmooth,
};

// Selects the initial matching coordinates before optimization.
enum class MatchingInitialization : uint8_t
{
    // Initialize reproducibly from the configured random seed.
    SeededRandom,
    // Initialize RT60 to one second and correction gains to zero.
    Neutral,
    // Initialize RT60 from decay estimates measured from the target RIR.
    TargetDerived,
};

// Bounds and scaling used when converting matching coordinates to an FDN config.
struct MatchingParameterConfig
{
    MatchingParameterization parameterization = MatchingParameterization::ScaledSmooth;
    MatchingInitialization initialization = MatchingInitialization::Neutral;
    double minimum_t60 = 0.1;
    double maximum_t60 = 20.0;
    double tone_gain_scale_db = 12.0;
    bool zero_mean_tone_gains = true;
};
} // namespace fdn_optimization