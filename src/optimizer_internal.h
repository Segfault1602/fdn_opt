#pragma once

#include <armadillo>

namespace fdn_optimization::detail
{

inline bool ShouldUseStoredBest(const arma::mat& best_coordinates, double best_objective, double final_objective)
{
    return !best_coordinates.is_empty() && best_objective <= final_objective;
}

} // namespace fdn_optimization::detail
