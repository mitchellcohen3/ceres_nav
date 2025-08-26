#pragma once

#include "ExtendedPoseLocalParameterization.h"  
// #include <ceres/ceres.h>

class DecoupledExtendedPoseLocalParameterization : public ExtendedPoseLocalParameterization {
public:
  using ExtendedPoseLocalParameterization::ExtendedPoseLocalParameterization;
  ~DecoupledExtendedPoseLocalParameterization() override = default;
  
  /**
   * @brief State update funciton for the extended Pose state.
   */
  bool Plus(const double *x, const double *delta, double *x_plus_delta) const;
};
