#pragma once

#include "lie/LieDirection.h"
#include <Eigen/Dense>
#include <ceres/ceres.h>

namespace ceres_nav {

class SO3LocalParameterization : public ceres::LocalParameterization {
public:
  SO3LocalParameterization(LieDirection direction = LieDirection::left)
      : _direction(direction) {}

  /**
   * @brief State update function for the SO3 state
   */
  bool Plus(const double *x, const double *delta, double *x_plus_delta) const;

  /**
   * @brief Computes the jacobian in respect to the local parameterization
   */
  bool ComputeJacobian(const double *x, double *jacobian) const;

  // The global size is 9, representing the flattened 3x3 rotation matrix
  int GlobalSize() const { return 9; };

  // Local size is 3, representing the 3 DoF Lie algebra element
  int LocalSize() const { return 3; };

  // Direction: left or right
  LieDirection direction() const { return _direction; }

  /**
   * @brief returns the Eigen Jacobian with respect to the local
   * parameterization. Usefor for debugging.
   */
  Eigen::Matrix<double, 9, 3> getEigenJacobian() const;

protected:
  LieDirection _direction;
};
} // namespace ceres_nav