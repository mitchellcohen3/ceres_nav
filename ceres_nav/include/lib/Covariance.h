#pragma once

#include "lib/StateCollection.h"
#include <ceres/ceres.h>

namespace ceres_swf {
bool calculateCovariance(ceres::Problem &graph, StateCollection &states,
                         const std::string &key, double timestamp);
}; // namespace ceres_swf