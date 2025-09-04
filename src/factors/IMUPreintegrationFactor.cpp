#include "factors/IMUPreintegrationFactor.h"

#include "lie/LieDirection.h"
#include "lie/SE23.h"
#include "lie/SO3.h"
#include "utils/Utils.h"

#include <glog/logging.h>

namespace ceres_nav {

IMUPreintegrationFactor::IMUPreintegrationFactor(
    const IMUIncrement &imu_increment_, bool use_group_jacobians_)
    : helper{imu_increment_, use_group_jacobians_} {}

bool IMUPreintegrationFactor::Evaluate(double const *const *parameters,
                                       double *residuals,
                                       double **jacobians) const {
  // Extract parameters
  Eigen::Matrix<double, 5, 5> T_i = SE23::fromCeresParameters(parameters[0]);
  Eigen::Matrix3d C_i = T_i.block<3, 3>(0, 0);
  Eigen::Vector3d v_i = T_i.block<3, 1>(0, 3);
  Eigen::Vector3d r_i = T_i.block<3, 1>(0, 4);

  Eigen::Matrix<double, 5, 5> T_j = SE23::fromCeresParameters(parameters[2]);
  Eigen::Matrix3d C_j = T_j.block<3, 3>(0, 0);
  Eigen::Vector3d v_j = T_j.block<3, 1>(0, 3);
  Eigen::Vector3d r_j = T_j.block<3, 1>(0, 4);
  
  Eigen::Map<const Eigen::Vector3d> bg_i(parameters[1]);
  Eigen::Map<const Eigen::Vector3d> ba_i(parameters[1] + 3);
  Eigen::Map<const Eigen::Vector3d> bg_j(parameters[3]);
  Eigen::Map<const Eigen::Vector3d> ba_j(parameters[3] + 3);

  // Construct full IMU states
  IMUStateHolder X_i(C_i, v_i, r_i, bg_i, ba_i);
  IMUStateHolder X_j(C_j, v_j, r_j, bg_j, ba_j);

  // Compute error and weight
  Eigen::Matrix<double, 15, 1> error =
      helper.computePreintegrationError(X_i, X_j);
  Eigen::Matrix<double, 15, 15> covariance = helper.covariance();
  // Eigen::Matrix<double, 15, 15> De_D_delta_xy =
  //     -Eigen::Matrix<double, 15, 15>::Identity();
  // if (use_group_jacobians) {
  //   Eigen::Matrix<double, 15, 1> error = computeRawError(X_i, X_j);
  //   Eigen::Matrix<double, 9, 1> e_nav = error.block<9, 1>(0, 0);
  //   De_D_delta_xy.block<9, 9>(0, 0) = -SE23::rightJacobianInverse(e_nav);
  // }
  // covariance = De_D_delta_xy * covariance * De_D_delta_xy.transpose();
  // covariance = 0.5 * (covariance + covariance.transpose());

  Eigen::Matrix<double, 15, 15> sqrt_info =
      ceres_nav::computeSquareRootInformation(covariance);

  Eigen::Map<Eigen::Matrix<double, 15, 1>> residual(residuals);
  residual = error;
  residual = sqrt_info * residual;

  // Compute Jacobians
  std::vector<Eigen::Matrix<double, 15, 15>> jac_vec =
      helper.computeRawJacobians(X_i, X_j);
  Eigen::Matrix<double, 15, 15> jac_i = sqrt_info * jac_vec[0];
  Eigen::Matrix<double, 15, 15> jac_j = sqrt_info * jac_vec[1];
  if (jacobians) {
    // Jacobian of residual with respect to extended pose km1
    if (jacobians[0]) {
      Eigen::Matrix<double, 15, 9> jac_extended_pose_i =
          jac_i.block<15, 9>(0, 0);

      // Insert column of zeros to jac_extended pose at the 4th position
      Eigen::Map<Eigen::Matrix<double, 15, 15, Eigen::RowMajor>>
          jac_pose_i_ceres(jacobians[0]);
      jac_pose_i_ceres.setZero();
      jac_pose_i_ceres.leftCols(3) = jac_extended_pose_i.leftCols(3);
      jac_pose_i_ceres.rightCols(6) = jac_extended_pose_i.rightCols(6);
    }
    // Jacobian w.r.t. bias km1
    if (jacobians[1]) {
      Eigen::Matrix<double, 15, 6> jac_bias_i = jac_i.block<15, 6>(0, 9);

      Eigen::Map<Eigen::Matrix<double, 15, 6, Eigen::RowMajor>>
          jac_bias_i_ceres(jacobians[1]);
      jac_bias_i_ceres.setZero();
      jac_bias_i_ceres = jac_bias_i;
    }
    // Jacobian w.r.t. extended pose k
    if (jacobians[2]) {
      Eigen::Matrix<double, 15, 9> jac_extended_pose_j =
          jac_j.block<15, 9>(0, 0);
      Eigen::Map<Eigen::Matrix<double, 15, 15, Eigen::RowMajor>>
          jac_pose_j_ceres(jacobians[2]);
      jac_pose_j_ceres.setZero();
      jac_pose_j_ceres.leftCols(3) = jac_extended_pose_j.leftCols(3);
      jac_pose_j_ceres.rightCols(6) = jac_extended_pose_j.rightCols(6);
    }

    if (jacobians[3]) {

      Eigen::Matrix<double, 15, 6> jac_bias_j = jac_j.block<15, 6>(0, 9);
      Eigen::Map<Eigen::Matrix<double, 15, 6, Eigen::RowMajor>>
          jac_bias_j_ceres(jacobians[3]);
      jac_bias_j_ceres.setZero();
      jac_bias_j_ceres = jac_bias_j;
    }
  }
  return true;
}
} // namespace ceres_nav