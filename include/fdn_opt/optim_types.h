#pragma once

#include <cstdint>

namespace fdn_optimization
{
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

enum class GradientMethod : uint8_t
{
    CentralDifferences,
    ForwardDifferences,
};
} // namespace fdn_optimization