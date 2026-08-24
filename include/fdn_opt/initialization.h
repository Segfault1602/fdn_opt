#pragma once

#include <cstdint>
#include <vector>

namespace fdn_optimization
{

struct RandomGainPair
{
    std::vector<float> input;
    std::vector<float> output;
};

RandomGainPair GenerateRandomNormalizedGains(std::uint32_t order, std::uint32_t seed);

} // namespace fdn_optimization
