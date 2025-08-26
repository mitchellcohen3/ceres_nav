#pragma once

#include "lie/LieDirection.h"
#include <Eigen/Dense>
#include <ceres/ceres.h>

class ExtendedPoseLocalParameterization : public ceres::LocalParameterization {
public:
  ExtendedPoseLocalParameterization(LieDirection direction = LieDirection::left)
      : _direction(direction) {}

  // Destructor
  ~ExtendedPoseLocalParameterization() override = default;
  
  /**
   * @brief State update funciton for the extended Pose state.
   */
  virtual bool Plus(const double *x, const double *delta, double *x_plus_delta) const = 0;
  bool ComputeJacobian(const double *x, double *jacobian) const;
  int GlobalSize() const { return 15; };
  int LocalSize() const { return 9; };

  /**
   * @brief returns the Eigen Jacobian with respect to the local
   * parameterization. Usefor for debugging. Usefor for debugggining
   */
  Eigen::Matrix<double, 15, 9> getEigenJacobian() const;

  // Get the direction
  LieDirection direction() const { return _direction; }

  // Set the direction
  void setDirection(LieDirection direction) { _direction = direction; }


protected:
  LieDirection _direction;
};
