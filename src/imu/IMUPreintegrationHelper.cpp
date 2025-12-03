#include "imu/IMUPreintegrationHelper.h"

namespace ceres_nav {

IMUPreintegrationHelper::IMUPreintegrationHelper(
    const IMUIncrement &imu_increment, bool use_group_jacobians_)
    : rmi{imu_increment}, use_group_jacobians{use_group_jacobians_},
      direction{imu_increment.options()->direction},
      pose_rep{imu_increment.options()->pose_rep} {}

Eigen::Matrix<double, 15, 1>
IMUPreintegrationHelper::computePreintegrationError(
    const IMUStateHolder &X_i, const IMUStateHolder &X_j) const {
  // Correct RMI for new value of bias
  Eigen::Matrix<double, 5, 5> Y_meas = getUpdatedRMI(X_i);
  // Get predicted RMI based on states and form the error
  Eigen::Matrix<double, 5, 5> Y_pred = predictNavRMI(X_i, X_j);

  // Compute the error on the navigation RMI
  Eigen::Matrix<double, 9, 1> e_nav = Eigen::Matrix<double, 9, 1>::Zero();
  if (pose_rep == ExtendedPoseRepresentation::SE23) {
    // If we're using SE_2(3) representation, compute the error between the
    // predicted and measured RMI on SE_2(3)
    e_nav = SE23::minus(Y_pred, Y_meas, direction);
  } else if (pose_rep == ExtendedPoseRepresentation::Decoupled) {
    Eigen::Vector3d delta_phi = SO3::minus(Y_pred.block<3, 3>(0, 0),
                                           Y_meas.block<3, 3>(0, 0), direction);
    Eigen::Vector3d delta_v =
        Y_pred.block<3, 1>(0, 3) - Y_meas.block<3, 1>(0, 3);
    Eigen::Vector3d delta_r =
        Y_pred.block<3, 1>(0, 4) - Y_meas.block<3, 1>(0, 4);

    // Else, we compute the error using the decoupled representation
    e_nav.block<3, 1>(0, 0) = delta_phi;
    e_nav.block<3, 1>(3, 0) = delta_v;
    e_nav.block<3, 1>(6, 0) = delta_r;
  }

  Eigen::Vector3d e_bg = X_j.bias_gyro - X_i.bias_gyro;
  Eigen::Vector3d e_ba = X_j.bias_accel - X_i.bias_accel;

  Eigen::Matrix<double, 15, 1> error;
  error.block<9, 1>(0, 0) = e_nav;
  error.block<3, 1>(9, 0) = e_bg;
  error.block<3, 1>(12, 0) = e_ba;
  return error;
}

Eigen::Matrix<double, 5, 5>
IMUPreintegrationHelper::getUpdatedRMI(const IMUStateHolder &X_i) const {
  Eigen::Matrix<double, 6, 1> dbias;
  Eigen::Vector3d d_bg = X_i.bias_gyro - rmi.gyroBias();
  Eigen::Vector3d d_ba = X_i.bias_accel - rmi.accelBias();
  dbias.block<3, 1>(0, 0) = d_bg;
  dbias.block<3, 1>(3, 0) = d_ba;

  Eigen::Matrix<double, 5, 5> delta_X = rmi.meanRMI();
  // If we're using an SE_2(3) representation, perform first-order
  // bias correct directly on the group
  if (pose_rep == ExtendedPoseRepresentation::SE23) {
    if (direction == LieDirection::left) {
      return SE23::expMap(rmi.biasJacobian() * dbias) * delta_X;
    } else if (direction == LieDirection::right) {
      return delta_X * SE23::expMap(rmi.biasJacobian() * dbias);
    } else {
      std::cout << "WARNING: Unknown Lie direction!" << std::endl;
      return Eigen::Matrix<double, 5, 5>::Identity();
    }
  }

  // Perform first-order bias correct for decoupled representation
  // This corresponds to the scheme presented in "On-Manifold Preintegration for
  // Real-Time "Visual–Inertial Odometry" by Forster et al. (2017). (see
  // equation (44))
  else if (pose_rep == ExtendedPoseRepresentation::Decoupled) {
    // Extract subcomponents
    Eigen::Matrix3d delta_C = delta_X.block<3, 3>(0, 0);
    Eigen::Vector3d delta_v = delta_X.block<3, 1>(0, 3);
    Eigen::Vector3d delta_r = delta_X.block<3, 1>(0, 4);

    // Extract relevant parts of the bias Jacobian
    Eigen::Matrix<double, 9, 6> bias_jacobian = rmi.biasJacobian();
    Eigen::Matrix3d dC_dbg = bias_jacobian.block<3, 3>(0, 0);

    Eigen::Matrix<double, 3, 6> dv_db = bias_jacobian.block<3, 6>(3, 0);
    Eigen::Matrix<double, 3, 6> dr_db = bias_jacobian.block<3, 6>(6, 0);
    Eigen::Vector3d delta_v_updated = delta_v + dv_db * dbias;
    Eigen::Vector3d delta_r_updated = delta_r + dr_db * dbias;

    Eigen::Matrix3d delta_C_updated;
    // Update delta_C with the correct perturbation
    if (direction == LieDirection::left) {
      LOG(INFO) << "Using left jacobians for decoupled navigation state "
                   "representation.";
      delta_C_updated = SO3::expMap(dC_dbg * d_bg) * delta_C;
    } else if (direction == LieDirection::right) {
      // Perform first-order correction
      delta_C_updated = delta_C * SO3::expMap(dC_dbg * d_bg);
    }
    // Assemble the updated delta_X matrix
    Eigen::Matrix<double, 5, 5> delta_X_updated =
        Eigen::Matrix<double, 5, 5>::Identity();
    delta_X_updated.block<3, 3>(0, 0) = delta_C_updated;
    delta_X_updated.block<3, 1>(0, 3) = delta_v_updated;
    delta_X_updated.block<3, 1>(0, 4) = delta_r_updated;
    return delta_X_updated;
  } else {
    LOG(ERROR) << "Unknown ExtendedPoseRepresentation type!";
    return Eigen::Matrix<double, 5, 5>::Identity();
  }
}

Eigen::Matrix<double, 5, 5>
IMUPreintegrationHelper::predictNavRMI(const IMUStateHolder &X_i,
                                       const IMUStateHolder &X_j) const {
  Eigen::Vector3d g_a = rmi.options()->gravity;
  Eigen::Matrix3d C_i = X_i.attitude;
  Eigen::Vector3d v_i = X_i.velocity;
  Eigen::Vector3d r_i = X_i.position;

  Eigen::Matrix3d C_j = X_j.attitude;
  Eigen::Vector3d v_j = X_j.velocity;
  Eigen::Vector3d r_j = X_j.position;

  double delta_t = rmi.deltaT();
  Eigen::Matrix3d delta_C = C_i.transpose() * C_j;
  Eigen::Vector3d delta_v = C_i.transpose() * (v_j - v_i - g_a * delta_t);
  Eigen::Vector3d delta_r = C_i.transpose() * (r_j - r_i - v_i * delta_t -
                                               0.5 * g_a * (delta_t * delta_t));

  Eigen::Matrix<double, 5, 5> delta_X_nav =
      SE23::fromComponents(delta_C, delta_v, delta_r);
  return delta_X_nav;
}

std::vector<Eigen::Matrix<double, 15, 15>>
IMUPreintegrationHelper::computeRawJacobians(const IMUStateHolder &X_i,
                                             const IMUStateHolder &X_j) const {
  if (pose_rep == ExtendedPoseRepresentation::SE23) {
    if (direction == LieDirection::left) {
      return computeRawJacobiansLeftSE23(X_i, X_j);
    } else if (direction == LieDirection::right) {
      return computeRawJacobiansRightSE23(X_i, X_j);
    } else {
      std::cout << "WARNING: Unknown Lie direction!" << std::endl;
      return std::vector<Eigen::Matrix<double, 15, 15>>();
    }
  } else if (pose_rep == ExtendedPoseRepresentation::Decoupled) {
    if (direction == LieDirection::left) {
      LOG(INFO) << "Left Jacobians not implemented for decoupled navigation "
                   "state representation.";
      return std::vector<Eigen::Matrix<double, 15, 15>>();
    } else if (direction == LieDirection::right) {
      return computeRawJacobiansRightDecoupled(X_i, X_j);
    } else {
      LOG(INFO) << "Unknown Lie direction for decoupled navigation state "
                   "representation.";
      return std::vector<Eigen::Matrix<double, 15, 15>>();
    }
  }

  return std::vector<Eigen::Matrix<double, 15, 15>>();
}

std::vector<Eigen::Matrix<double, 15, 15>>
IMUPreintegrationHelper::computeRawJacobiansLeftSE23(
    const IMUStateHolder &X_i, const IMUStateHolder &X_j) const {
  // Extract individual states
  Eigen::Vector3d g_a = rmi.options()->gravity;
  Eigen::Vector3d ba_i = X_i.bias_accel;
  double delta_t = rmi.deltaT();

  // Compute Jacobian of \Delta_X with respect to X_i
  Eigen::Matrix<double, 15, 15> D_deltaX_D_Xi =
      Eigen::Matrix<double, 15, 15>::Zero();
  Eigen::Matrix<double, 5, 5> T_i = Eigen::Matrix<double, 5, 5>::Identity();
  T_i.block<3, 3>(0, 0) = X_i.attitude;
  T_i.block<3, 1>(0, 3) = X_i.velocity;
  T_i.block<3, 1>(0, 4) = X_i.position;

  Eigen::Matrix<double, 5, 5> Phi_Ti_inv =
      SE23::inverse(computePhiMatrix(T_i, delta_t));
  Eigen::Matrix<double, 9, 9> d_Upsilon_dTi =
      -SE23::adjoint(Phi_Ti_inv) * computeFMatrix(delta_t);

  // Jacobian of \Delta X^X with respect to X_i
  D_deltaX_D_Xi.block<9, 9>(0, 0) = d_Upsilon_dTi;
  D_deltaX_D_Xi.block<6, 6>(9, 9) = -Eigen::Matrix<double, 6, 6>::Identity();

  // Jacobian of \Delta X_hat with respect to X_i (due to bias update)
  Eigen::Matrix<double, 15, 15> D_deltaXhat_D_Xi =
      Eigen::Matrix<double, 15, 15>::Zero();
  Eigen::Matrix<double, 6, 1> dbias;
  dbias.block<3, 1>(0, 0) = X_i.bias_gyro - rmi.gyroBias();
  dbias.block<3, 1>(3, 0) = X_i.bias_accel - rmi.accelBias();
  Eigen::Matrix<double, 9, 6> bias_jac = rmi.biasJacobian();
  Eigen::Matrix<double, 9, 1> tau = bias_jac * dbias;
  Eigen::Matrix<double, 9, 6> Ji_X_b =
      SE23::leftJacobian(bias_jac * dbias) * bias_jac;
  D_deltaXhat_D_Xi.block<9, 6>(0, 9) = Ji_X_b;

  // If we want to include the group jacobians, need to compute
  // the Jacobian of the error with respect to the delta_X^Y
  // and delta_X^X
  Eigen::Matrix<double, 15, 15> De_D_delta_Xhat =
      -Eigen::Matrix<double, 15, 15>::Identity();
  Eigen::Matrix<double, 15, 15> De_D_delta_X =
      Eigen::Matrix<double, 15, 15>::Identity();

  if (use_group_jacobians) {
    LOG(INFO) << "Using group Jacobians!";
    Eigen::Matrix<double, 15, 1> error = computePreintegrationError(X_i, X_j);
    Eigen::Matrix<double, 9, 1> e_nav = error.block<9, 1>(0, 0);

    De_D_delta_X.block<9, 9>(0, 0) = SE23::leftJacobianInverse(e_nav);
    De_D_delta_Xhat.block<9, 9>(0, 0) = -SE23::rightJacobianInverse(e_nav);
  }

  Eigen::Matrix<double, 15, 15> jac_i =
      De_D_delta_X * D_deltaX_D_Xi + De_D_delta_Xhat * D_deltaXhat_D_Xi;

  // Compute Jacobians of error w.r.t X_j
  Eigen::Matrix<double, 15, 15> D_deltaX_D_Xj =
      Eigen::Matrix<double, 15, 15>::Zero();
  Eigen::Matrix<double, 5, 5> gammaInv =
      SE23::inverse(computeGammaMatrix(delta_t, g_a));
  D_deltaX_D_Xj.block<9, 9>(0, 0) = SE23::adjoint(Phi_Ti_inv * gammaInv);
  D_deltaX_D_Xj.block<6, 6>(9, 9) = Eigen::Matrix<double, 6, 6>::Identity();
  Eigen::Matrix<double, 15, 15> jac_j = De_D_delta_X * D_deltaX_D_Xj;

  std::vector<Eigen::Matrix<double, 15, 15>> raw_jacobians;
  raw_jacobians.push_back(jac_i);
  raw_jacobians.push_back(jac_j);
  return raw_jacobians;
}

std::vector<Eigen::Matrix<double, 15, 15>>
IMUPreintegrationHelper::computeRawJacobiansRightSE23(
    const IMUStateHolder &X_i, const IMUStateHolder &X_j) const {
  Eigen::Matrix<double, 5, 5> Upsilon_ij_bar = predictNavRMI(X_i, X_j);

  double dt = rmi.deltaT();
  Eigen::Matrix<double, 15, 15> D_deltaX_D_Xi =
      Eigen::Matrix<double, 15, 15>::Zero();
  D_deltaX_D_Xi.block<9, 9>(0, 0) =
      -SE23::adjoint(SE23::inverse(Upsilon_ij_bar)) * computeFMatrix(dt);
  D_deltaX_D_Xi.block<6, 6>(9, 9) = -Eigen::Matrix<double, 6, 6>::Identity();

  // Jacobian of \Delta_X_hat with respect to X_i (due to bias update)
  Eigen::Matrix<double, 15, 15> D_deltaXhat_D_Xi =
      Eigen::Matrix<double, 15, 15>::Zero();
  Eigen::Matrix<double, 6, 1> dbias;
  dbias.block<3, 1>(0, 0) = X_i.bias_gyro - rmi.gyroBias();
  dbias.block<3, 1>(3, 0) = X_i.bias_accel - rmi.accelBias();
  Eigen::Matrix<double, 9, 6> bias_jac = rmi.biasJacobian();
  Eigen::Matrix<double, 9, 1> tau = bias_jac * dbias;
  D_deltaXhat_D_Xi.block<9, 6>(0, 9) = SE23::rightJacobian(tau) * bias_jac;

  // Jacobian of \Delta X^X with respect to X_j
  Eigen::Matrix<double, 15, 15> D_deltaX_DXj =
      Eigen::Matrix<double, 15, 15>::Identity();

  // Jacobians of errors with respect to RMI
  Eigen::Matrix<double, 15, 15> De_D_DeltaX =
      Eigen::Matrix<double, 15, 15>::Identity();
  Eigen::Matrix<double, 15, 15> De_D_DeltaXhat =
      Eigen::Matrix<double, 15, 15>::Identity();

  if (use_group_jacobians) {
    Eigen::Matrix<double, 15, 1> error = computePreintegrationError(X_i, X_j);
    Eigen::Matrix<double, 9, 1> e_nav = error.block<9, 1>(0, 0);
    De_D_DeltaXhat.block<9, 9>(0, 0) = -SE23::leftJacobianInverse(e_nav);
    De_D_DeltaX.block<9, 9>(0, 0) = SE23::rightJacobianInverse(e_nav);
  }

  Eigen::Matrix<double, 15, 15> jac_i =
      De_D_DeltaX * D_deltaX_D_Xi + De_D_DeltaXhat * D_deltaXhat_D_Xi;
  Eigen::Matrix<double, 15, 15> jac_j = De_D_DeltaX * D_deltaX_DXj;

  std::vector<Eigen::Matrix<double, 15, 15>> raw_jacobians;
  raw_jacobians.push_back(jac_i);
  raw_jacobians.push_back(jac_j);
  return raw_jacobians;
}

std::vector<Eigen::Matrix<double, 15, 15>>
IMUPreintegrationHelper::computeRawJacobiansRightDecoupled(
    const IMUStateHolder &X_i, const IMUStateHolder &X_j) const {
  Eigen::Matrix<double, 5, 5> del_Xx_mat = predictNavRMI(X_i, X_j);

  // Extract required subcomponents of matrix
  Eigen::Matrix3d del_Xx_C = del_Xx_mat.block<3, 3>(0, 0);
  Eigen::Vector3d del_Xx_v = del_Xx_mat.block<3, 1>(0, 3);
  Eigen::Vector3d del_Xx_r = del_Xx_mat.block<3, 1>(0, 4);
  double delta_t = rmi.deltaT();

  Eigen::Matrix3d C_i = X_i.attitude;

  // Compute Jacobian of \Delta_X^X with respect to X_i
  Eigen::Matrix<double, 15, 15> D_delta_xx_Xi =
      Eigen::Matrix<double, 15, 15>::Zero();
  D_delta_xx_Xi.block<3, 3>(0, 0) = -del_Xx_C.transpose();
  D_delta_xx_Xi.block<3, 3>(3, 0) = SO3::cross(del_Xx_v);
  D_delta_xx_Xi.block<3, 3>(3, 3) = -C_i.transpose();
  D_delta_xx_Xi.block<3, 3>(6, 0) = SO3::cross(del_Xx_r);
  D_delta_xx_Xi.block<3, 3>(6, 3) = -delta_t * C_i.transpose();
  D_delta_xx_Xi.block<3, 3>(6, 6) = -C_i.transpose();
  D_delta_xx_Xi.block<6, 6>(9, 9) = -Eigen::Matrix<double, 6, 6>::Identity();

  // Compute Jacobian of \Delta_X^Y with respect to X_i
  // We need the bias Jacobian here
  Eigen::Matrix<double, 15, 15> D_delta_xy_Xi =
      Eigen::Matrix<double, 15, 15>::Zero();
  Eigen::Matrix<double, 9, 9> right_jac_part =
      Eigen::Matrix<double, 9, 9>::Zero();
  Eigen::Matrix<double, 9, 6> bias_jac = rmi.biasJacobian();
  D_delta_xy_Xi.block<9, 6>(0, 9) = bias_jac;

  // Compute Jacobian of \Delta X^X with respect to X_j
  Eigen::Matrix<double, 15, 15> D_delta_xx_Xj =
      Eigen::Matrix<double, 15, 15>::Identity();
  D_delta_xx_Xj.block<3, 3>(3, 3) = C_i.transpose();
  D_delta_xx_Xj.block<3, 3>(6, 6) = C_i.transpose();

  // Group Jacobians
  Eigen::Matrix<double, 15, 15> De_D_delta_xy =
      -Eigen::Matrix<double, 15, 15>::Identity();
  Eigen::Matrix<double, 15, 15> De_D_delta_xx =
      Eigen::Matrix<double, 15, 15>::Identity();

  if (use_group_jacobians) {
    Eigen::Matrix<double, 15, 1> error = computePreintegrationError(X_i, X_j);
    Eigen::Matrix<double, 3, 1> e_att = error.block<3, 1>(0, 0);

    De_D_delta_xx.block<3, 3>(0, 0) = SO3::rightJacobianInverse(e_att);
    De_D_delta_xy.block<3, 3>(0, 0) = -SO3::leftJacobianInverse(e_att);
  }

  Eigen::Matrix<double, 15, 15> jac_i =
      De_D_delta_xx * D_delta_xx_Xi + De_D_delta_xy * D_delta_xy_Xi;
  Eigen::Matrix<double, 15, 15> jac_j = De_D_delta_xx * D_delta_xx_Xj;

  std::vector<Eigen::Matrix<double, 15, 15>> raw_jacobians;
  raw_jacobians.push_back(jac_i);
  raw_jacobians.push_back(jac_j);
  return raw_jacobians;
}

Eigen::Matrix<double, 5, 5>
IMUPreintegrationHelper::computePhiMatrix(const Eigen::Matrix<double, 5, 5> &T,
                                          double dt) const {
  Eigen::Matrix<double, 5, 5> Phi = Eigen::Matrix<double, 5, 5>::Identity();

  Eigen::Vector3d v = T.block<3, 1>(0, 3);
  Eigen::Vector3d r = T.block<3, 1>(0, 4);

  Phi.block<3, 3>(0, 0) = T.block<3, 3>(0, 0);
  Phi.block<3, 1>(0, 3) = T.block<3, 1>(0, 3);
  Phi.block<3, 1>(0, 4) = r + dt * v;
  return Phi;
}

Eigen::Matrix<double, 9, 9>
IMUPreintegrationHelper::computeFMatrix(double dt) const {
  Eigen::Matrix<double, 9, 9> F = Eigen::Matrix<double, 9, 9>::Identity();
  F.block<3, 3>(6, 3) = Eigen::Matrix3d::Identity() * dt;
  return F;
}

Eigen::Matrix<double, 5, 5> IMUPreintegrationHelper::computeGammaMatrix(
    double dt, const Eigen::Vector3d &gravity) const {
  Eigen::Matrix<double, 5, 5> Gamma = Eigen::Matrix<double, 5, 5>::Identity();
  Gamma.block<3, 1>(0, 3) = dt * gravity;
  Gamma.block<3, 1>(0, 4) = 0.5 * dt * dt * gravity;
  return Gamma;
}

} // namespace ceres_nav