#include "factors/IMUPriorFactor.h"
#include "lie/SE23.h"
#include "lie/SE3.h"
#include "utils/Utils.h"

#include <string>

IMUPriorFactor::IMUPriorFactor(
    const Eigen::Matrix<double, 5, 5> &prior_nav_state,
    const Eigen::Matrix<double, 6, 1> &prior_imu_bias,
    const Eigen::Matrix<double, 15, 15> &prior_covariance,
    LieDirection direction, ExtendedPoseRepresentation pose_rep)
    : prior_nav_state_(prior_nav_state), prior_bias_(prior_imu_bias),
      prior_covariance_(prior_covariance), direction_(direction),
      pose_rep_(pose_rep) {
  // Compute square root info matrix
  sqrt_info_ = ceres_nav::computeSquareRootInformation(prior_covariance_);
}

bool IMUPriorFactor::Evaluate(double const *const *parameters,
                              double *residuals, double **jacobians) const {
  // Extract our state
  Eigen::Matrix<double, 5, 5> T_state =
      SE23::fromCeresParameters(parameters[0]);
  Eigen::Vector3d bg = Eigen::Map<const Eigen::Vector3d>(parameters[1]);
  Eigen::Vector3d ba = Eigen::Map<const Eigen::Vector3d>(parameters[1] + 3);

  // Compute the resudual
  Eigen::Matrix<double, 15, 1> error;
  if (pose_rep_ == ExtendedPoseRepresentation::SE23) {
    // Compute the error directly on SE_2(3)
    error.block<9, 1>(0, 0) =
        SE23::minus(T_state, prior_nav_state_, direction_);
  } else if (pose_rep_ == ExtendedPoseRepresentation::Decoupled) {
    Eigen::Matrix3d C = T_state.block<3, 3>(0, 0);
    Eigen::Vector3d v = T_state.block<3, 1>(0, 3);
    Eigen::Vector3d r = T_state.block<3, 1>(0, 4);
    Eigen::Matrix3d C_prior = prior_nav_state_.block<3, 3>(0, 0);
    Eigen::Vector3d v_prior = prior_nav_state_.block<3, 1>(0, 3);
    Eigen::Vector3d r_prior = prior_nav_state_.block<3, 1>(0, 4);
    error.block<3, 1>(0, 0) = SO3::minus(C, C_prior, direction_);
    error.block<3, 1>(3, 0) = v - v_prior;
    error.block<3, 1>(6, 0) = r - r_prior;
  }

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
