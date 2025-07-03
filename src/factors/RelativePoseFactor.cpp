#include "factors/RelativePoseFactor.h"
#include "lie/SE3.h"

RelativePoseFactor::RelativePoseFactor(
    const Eigen::Matrix4d &meas_,
    const Eigen::Matrix<double, 6, 6> &sqrt_info_,
    const LieDirection &direction_)
    : relative_pose_meas{meas_}, sqrt_info{sqrt_info_}, direction{direction_} {}

bool RelativePoseFactor::Evaluate(double const *const *parameters,
                                  double *residuals, double **jacobians) const {
  Eigen::Matrix3d C_i;
  Eigen::Vector3d r_i;
  Eigen::Matrix<double, 4, 4> T_i = SE3::fromCeresParameters(parameters[0]);
  SE3::toComponents(T_i, C_i, r_i);

  Eigen::Matrix3d C_j;
  Eigen::Vector3d r_j;
  Eigen::Matrix<double, 4, 4> T_j = SE3::fromCeresParameters(parameters[1]);
  SE3::toComponents(T_j, C_j, r_j);

  // Construct the predicted relative pose
  Eigen::Matrix4d T_ij_hat = SE3::inverse(T_i) * T_j;
  // Compute the residual
  Eigen::Matrix4d T_ij_meas = relative_pose_meas;

  Eigen::Matrix<double, 6, 1> error;
  error.setZero();
  if (direction == LieDirection::right) {
    error = SE3::logMap(SE3::inverse(T_j) * T_i * T_ij_meas);
  } else if (direction == LieDirection::left) {
    error = SE3::logMap(T_ij_meas * SE3::inverse(T_ij_hat));
  } else {
    std::cout << "Warning: Unknown LieDirection in RelativePoseFactor"
              << std::endl;
  }

  // Form the residual
  Eigen::Map<Eigen::Matrix<double, 6, 1>> residual(residuals);
  residual = error;
  residual = sqrt_info * residual;

  if (jacobians) {
    // Jacobian of residual with respect to extended pose km1
    if (jacobians[0]) {
      Eigen::Matrix<double, 6, 6> full_jac_i =
          Eigen::Matrix<double, 6, 6>::Zero();
      if (direction == LieDirection::left) {
        full_jac_i = -SE3::adjoint(SE3::inverse(T_i));
        full_jac_i = -SE3::rightJacobianInverse(residual) * full_jac_i;
      } else if (direction == LieDirection::right) {
        Eigen::Matrix<double, 4, 4> T_ij_bar = SE3::inverse(T_i) * T_j;
        full_jac_i = -SE3::adjoint(SE3::inverse(T_ij_bar));
        full_jac_i = -SE3::leftJacobianInverse(residual) * full_jac_i;
      } else {
        std::cout << "Warning: Unknown LieDirection in RelativePoseFactor"
                  << std::endl;
      }
      // Assign to full Jacobian
      Eigen::Map<Eigen::Matrix<double, 6, 12, Eigen::RowMajor>> jac_pose_i(
          jacobians[0]);
      jac_pose_i.setZero();
      jac_pose_i.leftCols(3) = full_jac_i.leftCols(3);
      jac_pose_i.rightCols(3) = full_jac_i.rightCols(3);
      jac_pose_i = sqrt_info * jac_pose_i;
    }
    // Jacobian w.r.t. bias km1
    if (jacobians[1]) {
      Eigen::Matrix<double, 6, 6> full_jac_j =
          Eigen::Matrix<double, 6, 6>::Zero();
      if (direction == LieDirection::left) {
        full_jac_j = SE3::adjoint(SE3::inverse(T_i));
        full_jac_j = -SE3::rightJacobianInverse(residual) * full_jac_j;
      }
      if (direction == LieDirection::right) {
        full_jac_j = -SE3::leftJacobianInverse(residual);
      }
      Eigen::Map<Eigen::Matrix<double, 6, 12, Eigen::RowMajor>> jac_pose_j(
          jacobians[1]);
      jac_pose_j.setZero();
      jac_pose_j.leftCols(3) = full_jac_j.leftCols(3);
      jac_pose_j.rightCols(3) = full_jac_j.rightCols(3);
      jac_pose_j = sqrt_info * jac_pose_j;
    }
  }

  return true;
}