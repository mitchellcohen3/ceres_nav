#pragma once

#include <Eigen/Dense>
#include <ceres/ceres.h>
#include <vector>

#include "lib/ExtendedPoseParameterBlock.h"
#include "lie/LieDirection.h"

namespace ceres_nav {

class AbsolutePositionFactor : public ceres::SizedCostFunction<3, 15> {
public:
  Eigen::Vector3d meas;
  Eigen::Matrix3d sqrt_info;

  // Pose representation options
  ExtendedPoseRepresentation pose_type = ExtendedPoseRepresentation::SE23;
  LieDirection direction;

  AbsolutePositionFactor(
      const Eigen::Vector3d &meas_, const LieDirection &direction_,
      const Eigen::Matrix3d &sqrt_info_,
      ExtendedPoseRepresentation pose_type_ = ExtendedPoseRepresentation::SE23)
      : meas{meas_}, direction{direction_}, sqrt_info{sqrt_info_},
        pose_type{pose_type_} {}

  /**
   * @brief Residual and Jacobian computation
   */
  bool Evaluate(double const *const *parameters, double *residuals,
                double **jacobians) const override;
};

} // namespace ceres_nav