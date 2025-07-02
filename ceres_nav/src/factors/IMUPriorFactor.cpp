#include "factors/IMUPriorFactor.h"
#include "lie/SE23.h"
#include "lie/SE3.h"
#include "utils/Utils.h"

#include <string>

IMUPriorFactor::IMUPriorFactor(
    const Eigen::Matrix<double, 5, 5> &prior_nav_state,
    const Eigen::Matrix<double, 6, 1> &prior_imu_bias,
    const Eigen::Matrix<double, 15, 15> &prior_covariance, LieDirection direction)
    : prior_nav_state_(prior_nav_state), prior_bias_(prior_imu_bias),
      prior_covariance_(prior_covariance), direction_(direction) {
  // Compute square root info matrix
  sqrt_info_ = computeSquareRootInformation(prior_covariance_);
}

bool IMUPriorFactor::Evaluate(double const *const *parameters,
                              double *residuals, double **jacobians) const {
  // Extract prior state
  Eigen::Matrix3d C;
  Eigen::Vector3d v;
  Eigen::Vector3d r;
  Eigen::Matrix<double, 5, 5> T_state =
      SE23::fromCeresParameters(parameters[0]);
  SE23::toComponents(T_state, C, v, r);

  Eigen::Vector3d bg(parameters[1][0], parameters[1][1], parameters[1][2]);
  Eigen::Vector3d ba(parameters[1][3], parameters[1][4], parameters[1][5]);

  // Compute the resudual
  Eigen::Matrix<double, 15, 1> error;
  error.block<9, 1>(0, 0) =
      SE23::minus(T_state, prior_nav_state_, direction_);
  error.block<3, 1>(9, 0) = bg - prior_bias_.block<3, 1>(0, 0);
  error.block<3, 1>(12, 0) = ba - prior_bias_.block<3, 1>(3, 0);  

  Eigen::Map<Eigen::Matrix<double, 15, 1>> residual(residuals);
  residual = error;
  residual = sqrt_info_ * residual;

  Eigen::Matrix<double, 15, 15> full_jac = sqrt_info_;
  Eigen::Matrix<double, 15, 9> jac_pose = full_jac.block<15, 9>(0, 0);
  Eigen::Matrix<double, 15, 6> jac_bias = full_jac.block<15, 6>(0, 9);

  // Compute Jacobians
  if (jacobians) {
    // Jacobian of residual with respect to extended pose km1
    if (jacobians[0]) {
      // Insert column of zeros to jac_extended pose at the 4th position
      Eigen::Map<Eigen::Matrix<double, 15, 15, Eigen::RowMajor>> jac_pose_ceres(
          jacobians[0]);
      jac_pose_ceres.setZero();
      jac_pose_ceres.leftCols(3) = jac_pose.leftCols(3);
      jac_pose_ceres.rightCols(6) = jac_pose.rightCols(6);
    }
    // Jacobian w.r.t. bias km1
    if (jacobians[1]) {
      Eigen::Map<Eigen::Matrix<double, 15, 6, Eigen::RowMajor>> jac_bias_ceres(
          jacobians[1]);
      jac_bias_ceres.setZero();
      jac_bias_ceres = jac_bias;
    }
  }

  return true;
}
