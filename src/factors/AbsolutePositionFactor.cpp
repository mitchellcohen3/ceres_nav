#include "factors/AbsolutePositionFactor.h"
#include "lie/SO3.h"
#include "lie/SE3.h"
#include "lie/SE23.h"

bool AbsolutePositionFactor::Evaluate(double const *const *parameters,
                                      double *residuals,
                                      double **jacobians) const {
  // Extract quadric parameters
  Eigen::Matrix3d imu_att;
  Eigen::Vector3d imu_vel;
  Eigen::Vector3d imu_pos;

  if (pose_type == "SE3") {
    Eigen::Matrix<double, 4, 4> imu_pose =
        SE3::fromCeresParameters(parameters[0]);
    SE3::toComponents(imu_pose, imu_att, imu_pos);
  } else if (pose_type == "SE23") {
    Eigen::Matrix<double, 5, 5> T_imu =
        SE23::fromCeresParameters(parameters[0]);
    SE23::toComponents(T_imu, imu_att, imu_vel, imu_pos);
  } else {
    throw std::runtime_error("Unknown pose type");
  }

  Eigen::Vector3d pred_meas = imu_pos;

  // Compute the residual
  Eigen::Matrix<double, 3, 1> error = meas - pred_meas;
  Eigen::Map<Eigen::Matrix<double, 3, 1>> residual(residuals);
  residual = sqrt_info * error;

  Eigen::Matrix<double, 3, 9> H_x = Eigen::Matrix<double, 3, 9>::Zero();
  if (direction == LieDirection::right) {
    H_x.block<3, 3>(0, 6) = imu_att;
  } else if (direction == LieDirection::left) {
    H_x.block<3, 3>(0, 0) = -SO3::cross(imu_pos);
    H_x.block<3, 3>(0, 6) = Eigen::Matrix3d::Identity();
  } else {
    std::cerr << "Invalid LieDirection" << std::endl;
  }

  H_x = -sqrt_info * H_x;
  if (jacobians) {
    // Jacobian of measurement model with respect to the Poses
    if (jacobians[0]) {
      // Fill in the relevant parts of the Jacobian
      if (pose_type == "SE3") {
        Eigen::Map<Eigen::Matrix<double, 3, 12, Eigen::RowMajor>> jacobian_pose(
            jacobians[0]);
        jacobian_pose.setZero();
        jacobian_pose.leftCols(3) = H_x.leftCols(3);
        jacobian_pose.rightCols(3) = H_x.rightCols(3);
      } else if (pose_type == "SE23") {
        Eigen::Map<Eigen::Matrix<double, 3, 15, Eigen::RowMajor>> jacobian_pose(
            jacobians[0]);
        jacobian_pose.setZero();
        jacobian_pose.leftCols(3) = H_x.leftCols(3);
        jacobian_pose.rightCols(3) = H_x.rightCols(3);
      }
    }
  }

  return true;
}