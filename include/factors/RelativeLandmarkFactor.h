#pragma once

#include "lie/LieDirection.h"
#include <Eigen/Dense>
#include <ceres/ceres.h>

#include "lib/ExtendedPoseParameterBlock.h"

namespace ceres_nav {
/**
 * @brief A factor for relative landmark measurements of the form
 *    y = C_ab.T * (r_pw_a - r_zw_a),
 *
 * where C_ab is the attitude of the robot, r_pw_a is the landmark position in
 * the global frame, and r_zw_a is the position of the robot in the global
 * frame.
 *
 * TODO: this factor currently only supports SE23 pose parameters (i.e., for use
 * in visual-inertial algorithms), it will take some thought on how to
 * genearalize it to SE3 pose parameters.
 */
class RelativeLandmarkFactor : public ceres::SizedCostFunction<3, 15, 3> {
public:
  Eigen::Vector3d meas;
  Eigen::Matrix3d sqrt_info;
  double stamp;

  // Pose representation options
  LieDirection direction;
  ExtendedPoseRepresentation pose_type = ExtendedPoseRepresentation::SE23;

  RelativeLandmarkFactor(const Eigen::Vector3d &meas_,
                         const Eigen::Matrix3d &sqrt_info_,
                         const double &stamp_, const LieDirection &direction_,
                         const ExtendedPoseRepresentation &pose_type_)
      : meas{meas_}, sqrt_info{sqrt_info_}, stamp{stamp_},
        direction{direction_}, pose_type{pose_type_} {}

  /**
   * @brief Residual and Jacobian computation
   */
  bool Evaluate(double const *const *parameters, double *residuals,
                double **jacobians) const;
};

} // namespace ceres_nav
