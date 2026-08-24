#pragma once

#include "optim_types.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace fdn_optimization
{

// Selects how independent three-band attenuation coordinates form semantic blocks.
enum class ThreeBandBlockGrouping : std::uint8_t
{
    // Group the low, mid, and high attenuation coordinates for each FDN channel.
    ChannelTriplets,
    // Group one attenuation band across all FDN channels.
    FrequencyBands,
};

constexpr const char* ThreeBandBlockGroupingToString(ThreeBandBlockGrouping grouping)
{
    return grouping == ThreeBandBlockGrouping::FrequencyBands ? "frequency-bands" : "channel-triplets";
}

// Describes one contiguous optimizer-coordinate range for a parameter type.
struct ParameterRange
{
    OptimizationParamType type;
    std::size_t offset;
    std::size_t size;
    std::size_t occurrence;
};

// Describes the complete ordered optimizer-coordinate layout.
struct ParameterLayout
{
    std::vector<ParameterRange> ranges;
    std::size_t total_size = 0;
};

// Describes one contiguous block used by a block-coordinate optimizer.
struct ParameterBlock
{
    std::vector<std::size_t> coordinates;
    std::optional<OptimizationParamType> semantic_type;
    std::size_t semantic_index = 0;
};

ParameterLayout BuildParameterLayout(std::uint32_t fdn_order, std::span<const OptimizationParamType> parameter_types);

std::vector<ParameterBlock> BuildSemanticParameterBlocks(
    const ParameterLayout& layout, std::uint32_t fdn_order,
    ThreeBandBlockGrouping three_band_grouping = ThreeBandBlockGrouping::ChannelTriplets);

std::vector<ParameterBlock> BuildContiguousParameterBlocks(std::size_t parameter_count, std::size_t block_size);

} // namespace fdn_optimization
