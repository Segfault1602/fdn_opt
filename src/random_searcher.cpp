#include "random_searcher.h"

#include <armadillo>

#include <limits>
#include <mutex>
#include <stop_token>
#include <thread>
#include <vector>

namespace fdn_optimization
{
RandomSearcher::RandomSearcher()
    : best_objective_(std::numeric_limits<double>::infinity())
{
}

} // namespace fdn_optimization