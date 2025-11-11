#pragma once

#include "lib/StateId.h"
#include <string>
#include <vector>

// Forward declarations
namespace ceres_nav {
class StateCollection;
}

namespace ceres {
class Problem;
}

// namespace ceres_nav {
// class StateID;
// }

namespace ceres_nav {

/**
 * @brief Computes the covariance for a given state in the StateCollection
 * using the provided Ceres Problem.
 *
 * It is assumed that the state exists in both the StateCollection and the Ceres
 * problem.
 */
bool calculateCovariance(ceres::Problem &graph, StateCollection &states,
                         const StateID &state_id);

/**
 * @brief Computes the covariance for a given set of states in the
 * StateCollection using the provided Ceres Problem.
 */
bool calculateCovariance(ceres::Problem &graph, StateCollection &states,
                         const std::vector<StateID> &state_ids);

}; // namespace ceres_nav