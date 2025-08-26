#include "factors/RelativeLandmarkFactor.h"
#include <iostream>

#include "lie/SE23.h"
#include "lie/SE3.h"
#include "lie/SO3.h"

bool RelativeLandmarkFactor::Evaluate(double const *const *parameters,
                                      double *residuals,
                                      double **jacobians) const {
  // Extract extended pose parameters
  Eigen::Matrix<double, 5, 5> T_ab = SE23::fromCeresParameters(parameters[0]);
  Eigen::Matrix3d C_ab = T_ab.block<3, 3>(0, 0);
  Eigen::Vector3d r_zw_a = T_ab.block<3, 1>(0, 4);

  // Extract landmark position
  Eigen::Vector3d r_pw_a(parameters[1][0], parameters[1][1], parameters[1][2]);

  // Evaluate measurement model and compute error
  Eigen::Vector3d y_check = C_ab.transpose() * (r_pw_a - r_zw_a);

  Eigen::Vector3d error = meas - y_check;
  Eigen::Map<Eigen::Vector3d> residual(residuals);
  residual = sqrt_info * error;

  // Compute the Jacobians of the factor
  // with respect to the pose and landmark parameters.
  // The Jacobians depend on the pose representation used and the Lie direction.
  Eigen::Matrix3d att_jacobian = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d pos_jacobian = Eigen::Matrix3d::Zero();

  switch (pose_type) {
  case ExtendedPoseRepresentation::SE23: {
    if (direction == LieDirection::right) {
      att_jacobian = -SO3::cross(y_check);
      pos_jacobian = Eigen::Matrix3d::Identity();
    } else if (direction == LieDirection::left) {
      att_jacobian = -C_ab.transpose() * SO3::cross(r_pw_a);
      pos_jacobian = C_ab.transpose();
    } else {
      std::cerr << "Invalid LieDirection" << std::endl;
    }
    break;
  }
  case ExtendedPoseRepresentation::Decoupled: {
    if (direction == LieDirection::right) {
      att_jacobian = -SO3::cross(y_check);
      pos_jacobian = C_ab.transpose();
    } else if (direction == LieDirection::left) {
      att_jacobian = -C_ab.transpose() * SO3::cross(r_pw_a - r_zw_a);
      pos_jacobian = C_ab.transpose();
    } else {
      std::cerr << "Invalid LieDirection" << std::endl;
    }
    break;
  }
  default:
    std::cerr << "Unknown state representation type" << std::endl;
  }

  att_jacobian *= sqrt_info;
  pos_jacobian *= sqrt_info;
  // Compute the Jacobians
  if (jacobians) {
    if (jacobians[0]) {
      Eigen::Map<Eigen::Matrix<double, 3, 15, Eigen::RowMajor>> jac_pose(
          jacobians[0]);
      jac_pose.setZero();
      jac_pose.leftCols(3) = att_jacobian;
      jac_pose.rightCols(3) = pos_jacobian;
    }
    if (jacobians[1]) {
      // Jacobian of residual with respect to landmark parameters
      Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> jac_landmark(
          jacobians[1]);
      jac_landmark.setZero();
      jac_landmark = -C_ab.transpose();
      jac_landmark = sqrt_info * jac_landmark;
    }
  }
  return true;
}