#pragma once

#include <cstdint>
#include <string>

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
} // namespace fdn_optimization