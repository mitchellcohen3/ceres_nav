#include "factors/AbsolutePositionFactor.h"
#include "lie/SE23.h"
#include "lie/SE3.h"
#include "lie/SO3.h"

namespace ceres_nav {

bool AbsolutePositionFactor::Evaluate(double const *const *parameters,
                                      double *residuals,
                                      double **jacobians) const {
  // Extract the pose from the parameters
  Eigen::Matrix<double, 5, 5> T_imu = SE23::fromCeresParameters(parameters[0]);
  Eigen::Matrix3d imu_att = T_imu.block<3, 3>(0, 0);
  Eigen::Vector3d imu_vel = T_imu.block<3, 1>(0, 3);
  Eigen::Vector3d imu_pos = T_imu.block<3, 1>(0, 4);

  Eigen::Vector3d pred_meas = imu_pos;

  // Compute the residual
  Eigen::Matrix<double, 3, 1> error = meas - pred_meas;
  Eigen::Map<Eigen::Matrix<double, 3, 1>> residual(residuals);
  residual = sqrt_info * error;

  Eigen::Matrix3d att_jacobian;
  Eigen::Matrix3d pos_jacobian;

  // Compute the Jacobians of the measurement model
  // based on the pose representation
  switch (pose_type) {
  case ExtendedPoseRepresentation::SE23: {
    if (direction == LieDirection::right) {
      att_jacobian = Eigen::Matrix3d::Zero();
      pos_jacobian = -imu_att;
    } else if (direction == LieDirection::left) {
      att_jacobian = SO3::cross(imu_pos);
      pos_jacobian = -Eigen::Matrix3d::Identity();
    } else {
      std::cerr << "Invalid LieDirection" << std::endl;
    }
    break;
  }
  case ExtendedPoseRepresentation::Decoupled: {
    pos_jacobian = -Eigen::Matrix3d::Identity();
    att_jacobian = Eigen::Matrix3d::Zero();
    break;
  }
  default:
    std::cerr << "Unknown state representation type" << std::endl;
    return false;
  }

  att_jacobian *= sqrt_info;
  pos_jacobian *= sqrt_info;

  if (jacobians) {
    // Jacobian of measurement model with respect to the Poses
    if (jacobians[0]) {
      Eigen::Map<Eigen::Matrix<double, 3, 15, Eigen::RowMajor>> jacobian_pose(
          jacobians[0]);
      jacobian_pose.setZero();
      jacobian_pose.leftCols(3) = att_jacobian;
      jacobian_pose.rightCols(3) = pos_jacobian;
    }
  }

  return true;
}

} // namespace ceres_nav