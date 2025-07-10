#pragma once

#include <string>

// Forward declarations 
class StateCollection;
namespace ceres {
    class Problem;
}

namespace ceres_nav {
bool calculateCovariance(ceres::Problem &graph, StateCollection &states,
                         const std::string &key, double timestamp);
}; // namespace ceres_nav