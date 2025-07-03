#include "factors/RelativeLandmarkFactor.h"
#include <iostream>

#include "lie/SE23.h"
#include "lie/SE3.h"
#include "lie/SO3.h"

bool RelativeLandmarkFactor::Evaluate(double const *const *parameters,
                                      double *residuals,
                                      double **jacobians) const {
  // Extract extended pose
  Eigen::Matrix3d C_ab;
  Eigen::Vector3d v_zw_a;
  Eigen::Vector3d r_zw_a;

  if (pose_type == "SE3") {
    Eigen::Matrix<double, 4, 4> T_ab = SE3::fromCeresParameters(parameters[0]);
    SE3::toComponents(T_ab, C_ab, r_zw_a);
  } else if (pose_type == "SE23") {
    Eigen::Matrix<double, 5, 5> T_ab = SE23::fromCeresParameters(parameters[0]);
    SE23::toComponents(T_ab, C_ab, v_zw_a, r_zw_a);
  } else {
    throw std::runtime_error("Unknown pose type");
  }

  // Extract landmark position
  Eigen::Vector3d r_pw_a(parameters[1][0], parameters[1][1], parameters[1][2]);

  // Evaluate measurement model and compute error
  Eigen::Vector3d y_check = C_ab.transpose() * (r_pw_a - r_zw_a);

  Eigen::Vector3d error = meas - y_check;
  Eigen::Map<Eigen::Vector3d> residual(residuals);
  residual = sqrt_info * error;

  // Compute the Jacobians
  if (jacobians) {
    if (jacobians[0]) {
      // Compute jacobian with respect to attitude and position
      Eigen::Matrix3d jac_att = Eigen::Matrix3d::Zero();
      Eigen::Matrix3d jac_position = Eigen::Matrix3d::Zero();
      if (direction == LieDirection::left) {
        jac_att = -C_ab.transpose() * SO3::cross(r_pw_a);
        jac_position = C_ab.transpose();
      } else if (direction == LieDirection::right) {
        jac_att = -SO3::cross(y_check);
        jac_position = Eigen::Matrix3d::Identity();
      } else {
        std::cerr << "Invalid LieDirection" << std::endl;
      }

      // size of Jacobian depends on the pose type
      if (pose_type == "SE3") {
        Eigen::Map<Eigen::Matrix<double, 3, 12, Eigen::RowMajor>> jac_pose(
            jacobians[0]);

        jac_pose.setZero();
        jac_pose.leftCols(3) = jac_att;
        jac_pose.rightCols(3) = jac_position;
        jac_pose = sqrt_info * jac_pose;
      } else if (pose_type == "SE23") {
        Eigen::Map<Eigen::Matrix<double, 3, 15, Eigen::RowMajor>> jac_pose(
            jacobians[0]);
        jac_pose.setZero();
        jac_pose.leftCols(3) = jac_att;
        jac_pose.rightCols(3) = jac_position;
        jac_pose = sqrt_info * jac_pose;
      }
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