#pragma once

#include "ExtendedPoseLocalParameterization.h"
#include <ceres/ceres.h>

namespace ceres_nav {
class SE23LocalParameterization : public ExtendedPoseLocalParameterization {
public:
  using ExtendedPoseLocalParameterization::ExtendedPoseLocalParameterization;
  ~SE23LocalParameterization() override = default;

  /**
   * @brief State update funciton for the extended Pose state.
   */
  bool Plus(const double *x, const double *delta, double *x_plus_delta) const;
};

} // namespace ceres_nav
