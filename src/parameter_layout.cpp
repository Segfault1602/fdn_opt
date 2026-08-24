#include "parameter_layout.h"

#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace
{

constexpr std::size_t kAttenuationBandCount = 10;

std::size_t ParameterSize(fdn_optimization::OptimizationParamType type, std::uint32_t fdn_order)
{
    using fdn_optimization::OptimizationParamType;

    switch (type)
    {
    case OptimizationParamType::Gains:
        return 2 * static_cast<std::size_t>(fdn_order);
    case OptimizationParamType::Matrix:
        return static_cast<std::size_t>(fdn_order) * fdn_order;
    case OptimizationParamType::Matrix_Householder:
    case OptimizationParamType::Matrix_Circulant:
        return fdn_order;
    case OptimizationParamType::AttenuationFilters:
    case OptimizationParamType::TonecorrectionFilters:
        return kAttenuationBandCount;
    case OptimizationParamType::AttenuationFilters_3Band:
        return 3 * static_cast<std::size_t>(fdn_order);
    case OptimizationParamType::OverallGain:
        return 1;
    }

    throw std::invalid_argument("Unknown optimization parameter type.");
}

void AddBlock(std::vector<fdn_optimization::ParameterBlock>& blocks, const fdn_optimization::ParameterRange& range,
              std::size_t offset, std::size_t size, std::size_t semantic_index)
{
    std::vector<std::size_t> coordinates(size);
    std::iota(coordinates.begin(), coordinates.end(), offset);
    blocks.push_back(
        {.coordinates = std::move(coordinates), .semantic_type = range.type, .semantic_index = semantic_index});
}

} // namespace

namespace fdn_optimization
{

ParameterLayout BuildParameterLayout(std::uint32_t fdn_order, std::span<const OptimizationParamType> parameter_types)
{
    if (fdn_order == 0)
    {
        throw std::invalid_argument("FDN order must be positive.");
    }

    ParameterLayout layout;
    std::size_t offset = 0;

    for (const auto type : parameter_types)
    {
        const auto occurrence = static_cast<std::size_t>(
            std::ranges::count_if(layout.ranges, [type](const ParameterRange& range) { return range.type == type; }));
        const std::size_t size = ParameterSize(type, fdn_order);
        layout.ranges.push_back({.type = type, .offset = offset, .size = size, .occurrence = occurrence});
        offset += size;
    }

    layout.total_size = offset;
    return layout;
}

std::vector<ParameterBlock> BuildSemanticParameterBlocks(const ParameterLayout& layout, std::uint32_t fdn_order,
                                                         ThreeBandBlockGrouping three_band_grouping)
{
    if (fdn_order == 0)
    {
        throw std::invalid_argument("FDN order must be positive.");
    }

    std::vector<ParameterBlock> blocks;
    for (const auto& range : layout.ranges)
    {
        switch (range.type)
        {
        case OptimizationParamType::Gains:
            AddBlock(blocks, range, range.offset, fdn_order, 0);
            AddBlock(blocks, range, range.offset + fdn_order, fdn_order, 1);
            break;
        case OptimizationParamType::AttenuationFilters_3Band:
            if (three_band_grouping == ThreeBandBlockGrouping::ChannelTriplets)
            {
                for (std::size_t channel = 0; channel < fdn_order; ++channel)
                {
                    AddBlock(blocks, range, range.offset + 3 * channel, 3, channel);
                }
            }
            else
            {
                for (std::size_t band = 0; band < 3; ++band)
                {
                    std::vector<std::size_t> coordinates(fdn_order);
                    for (std::size_t channel = 0; channel < fdn_order; ++channel)
                    {
                        coordinates[channel] = range.offset + 3 * channel + band;
                    }
                    blocks.push_back(
                        {.coordinates = std::move(coordinates), .semantic_type = range.type, .semantic_index = band});
                }
            }
            break;
        default:
            AddBlock(blocks, range, range.offset, range.size, range.occurrence);
            break;
        }
    }
    return blocks;
}

std::vector<ParameterBlock> BuildContiguousParameterBlocks(std::size_t parameter_count, std::size_t block_size)
{
    if (block_size == 0)
    {
        throw std::invalid_argument("Contiguous block size must be positive.");
    }
    if (parameter_count == 0)
    {
        throw std::invalid_argument("Parameter count must be positive.");
    }

    std::vector<ParameterBlock> blocks;
    for (std::size_t offset = 0; offset < parameter_count; offset += block_size)
    {
        std::vector<std::size_t> coordinates(std::min(block_size, parameter_count - offset));
        std::iota(coordinates.begin(), coordinates.end(), offset);
        blocks.push_back(
            {.coordinates = std::move(coordinates), .semantic_type = std::nullopt, .semantic_index = blocks.size()});
    }
    return blocks;
}

} // namespace fdn_optimization
