#include "initialization.h"

#include <armadillo>

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

} // namespace fdn_optimization
