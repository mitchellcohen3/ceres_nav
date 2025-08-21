#pragma once

#include <Eigen/Dense>
#include <ceres/ceres.h>
#include <vector>

#include "lie/LieDirection.h"
#include "lib/ExtendedPoseParameterBlock.h"

class AbsolutePositionFactor : public ceres::SizedCostFunction<3, 15> {
public:
  Eigen::Vector3d meas;
  Eigen::Matrix3d sqrt_info;
  bool print_debug = false;

  // Pose representation options
  ExtendedPoseRepresentation pose_type = ExtendedPoseRepresentation::SE23;
  LieDirection direction;

  AbsolutePositionFactor(
      const Eigen::Vector3d &meas_, const LieDirection &direction_,
      const Eigen::Matrix3d &sqrt_info_, bool print_debug_ = false,
      ExtendedPoseRepresentation pose_type_ = ExtendedPoseRepresentation::SE23)
      : meas{meas_}, direction{direction_}, sqrt_info{sqrt_info_},
        print_debug{print_debug_}, pose_type{pose_type_} {}

  /**
   * @brief Residual and Jacobian computation
   */
  bool Evaluate(double const *const *parameters, double *residuals,
                double **jacobians) const override;
};
