#include "imu/IMUPreintegrationHelper.h"

IMUPreintegrationHelper::IMUPreintegrationHelper(
    const IMUIncrement &imu_increment, bool use_group_jacobians_,
    const LieDirection &direction_, ExtendedPoseRepresentation pose_rep_)
    : rmi{imu_increment}, use_group_jacobians{use_group_jacobians_},
      direction{direction_}, pose_rep{pose_rep_} {}

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
    // Else, we compute the error using the decoupled representation
    e_nav.block<3, 1>(0, 0) = SO3::minus(Y_pred.block<3, 3>(0, 0),
                                         Y_meas.block<3, 3>(0, 0), direction);
    e_nav.block<3, 1>(3, 0) =
        Y_pred.block<3, 1>(0, 3) - Y_meas.block<3, 1>(0, 3);
    e_nav.block<3, 1>(6, 0) =
        Y_pred.block<3, 1>(0, 4) - Y_meas.block<3, 1>(0, 4);
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
  Eigen::Vector3d d_bg = X_i.bias_gyro - rmi.gyro_bias;
  Eigen::Vector3d d_ba = X_i.bias_accel - rmi.accel_bias;
  dbias.block<3, 1>(0, 0) = d_bg;
  dbias.block<3, 1>(3, 0) = d_ba;

  Eigen::Matrix<double, 5, 5> delta_X = Eigen::Matrix<double, 5, 5>::Identity();
  delta_X.block<3, 3>(0, 0) = rmi.delta_U.block<3, 3>(0, 0);
  delta_X.block<3, 1>(0, 3) = rmi.delta_U.block<3, 1>(0, 3);
  delta_X.block<3, 1>(0, 4) = rmi.delta_U.block<3, 1>(0, 4);

  // If we're using an SE_2(3) representation, perform first-order
  // bias correct directly on the group
  if (pose_rep == ExtendedPoseRepresentation::SE23) {
    if (direction == LieDirection::left) {
      return SE23::expMap(rmi.bias_jacobian * dbias) * delta_X;
    } else if (direction == LieDirection::right) {
      return delta_X * SE23::expMap(rmi.bias_jacobian * dbias);
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
    Eigen::Matrix3d dC_dbg = rmi.bias_jacobian.block<3, 3>(0, 0);

    Eigen::Matrix<double, 3, 6> dv_db = rmi.bias_jacobian.block<3, 6>(3, 0);
    Eigen::Matrix<double, 3, 6> dr_db = rmi.bias_jacobian.block<3, 6>(6, 0);

    Eigen::Vector3d delta_v_updated = delta_v + dv_db * dbias;
    Eigen::Vector3d delta_r_updated = delta_r + dr_db * dbias;

    Eigen::Matrix3d delta_C_updated;
    // Update delta_C with the correct perturbation
    if (direction == LieDirection::left) {
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
  Eigen::Vector3d g_a = rmi.gravity;
  Eigen::Matrix3d C_i = X_i.attitude;
  Eigen::Vector3d v_i = X_i.velocity;
  Eigen::Vector3d r_i = X_i.position;

  Eigen::Matrix3d C_j = X_j.attitude;
  Eigen::Vector3d v_j = X_j.velocity;
  Eigen::Vector3d r_j = X_j.position;

  double delta_t = rmi.end_stamp - rmi.start_stamp;
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
    } else if (direction == LieDirection::right) {
      return computeRawJacobiansRightDecoupled(X_i, X_j);
    } else {
      LOG(INFO) << "Unknown Lie direction for decoupled navigation state "
                   "representation.";
    }
  }
}

std::vector<Eigen::Matrix<double, 15, 15>>
IMUPreintegrationHelper::computeRawJacobiansLeftSE23(
    const IMUStateHolder &X_i, const IMUStateHolder &X_j) const {
  // Extract individual states
  Eigen::Vector3d g_a = rmi.gravity;
  Eigen::Matrix3d C_i = X_i.attitude;
  Eigen::Vector3d v_i = X_i.velocity;
  Eigen::Vector3d r_i = X_i.position;
  Eigen::Vector3d ba_i = X_i.bias_accel;

  Eigen::Vector3d v_j = X_j.velocity;
  Eigen::Vector3d r_j = X_j.position;

  Eigen::Matrix<double, 5, 5> del_Xy_mat = getUpdatedRMI(X_i);
  Eigen::Matrix<double, 5, 5> del_Xx_mat = predictNavRMI(X_i, X_j);

  // Extract required subcomponents of matrix
  Eigen::Matrix<double, 3, 1> del_Xx_v = del_Xx_mat.block<3, 1>(0, 3);
  Eigen::Matrix<double, 3, 1> del_Xx_r = del_Xx_mat.block<3, 1>(0, 4);
  double delta_t = rmi.end_stamp - rmi.start_stamp;

  // Jacobian of \Delta X^X with respect to X_i
  Eigen::Matrix<double, 15, 15> D_delta_xx_Xi =
      Eigen::Matrix<double, 15, 15>::Zero();
  D_delta_xx_Xi.block<3, 3>(0, 0) = -C_i.transpose();
  D_delta_xx_Xi.block<3, 3>(3, 0) = C_i.transpose() * SO3::cross(v_i);
  D_delta_xx_Xi.block<3, 3>(3, 3) = -C_i.transpose();
  D_delta_xx_Xi.block<3, 3>(6, 0) =
      C_i.transpose() * SO3::cross(r_i + v_i * delta_t);
  D_delta_xx_Xi.block<3, 3>(6, 3) = -delta_t * C_i.transpose();
  D_delta_xx_Xi.block<3, 3>(6, 6) = -C_i.transpose();
  D_delta_xx_Xi.block<6, 6>(9, 9) = -Eigen::Matrix<double, 6, 6>::Identity();

  // Compute Jacobian of error w.r.t del_Xy
  Eigen::Matrix<double, 6, 1> dbias;
  dbias.block<3, 1>(0, 0) = X_i.bias_gyro - rmi.gyro_bias;
  dbias.block<3, 1>(3, 0) = X_i.bias_accel - rmi.accel_bias;
  Eigen::Matrix<double, 9, 6> bias_jac = rmi.bias_jacobian;
  Eigen::Matrix<double, 9, 1> tau = bias_jac * dbias;

  Eigen::Matrix<double, 9, 6> Ji_X_b =
      SE23::leftJacobian(bias_jac * dbias) * bias_jac;
  Eigen::Matrix<double, 15, 15> xi_i_y = Eigen::Matrix<double, 15, 15>::Zero();
  xi_i_y.block<9, 6>(0, 9) = Ji_X_b;

  // If we want to include the group jacobians, need to compute
  // the Jacobian of the error with respect to the delta_X^Y
  // and delta_X^X
  Eigen::Matrix<double, 15, 15> De_D_delta_xy =
      -Eigen::Matrix<double, 15, 15>::Identity();
  Eigen::Matrix<double, 15, 15> De_D_delta_xx =
      Eigen::Matrix<double, 15, 15>::Identity();

  if (use_group_jacobians) {
    Eigen::Matrix<double, 15, 1> error = computePreintegrationError(X_i, X_j);
    Eigen::Matrix<double, 9, 1> e_nav = error.block<9, 1>(0, 0);

    De_D_delta_xx.block<9, 9>(0, 0) = SE23::leftJacobianInverse(e_nav);
    De_D_delta_xy.block<9, 9>(0, 0) = -SE23::rightJacobianInverse(e_nav);
  }

  Eigen::Matrix<double, 15, 15> jac_i =
      De_D_delta_xy * xi_i_y + De_D_delta_xx * D_delta_xx_Xi;

  // Compute Jacobians of error w.r.t X_j
  Eigen::Matrix3d Jj_C_C = C_i.transpose();
  Eigen::Matrix3d Jj_v_C = -C_i.transpose() * SO3::cross(v_j) +
                           SO3::cross(del_Xx_v) * C_i.transpose();
  Eigen::Matrix3d Jj_v_v = C_i.transpose();
  Eigen::Matrix3d Jj_r_C = -C_i.transpose() * SO3::cross(r_j) +
                           SO3::cross(del_Xx_r) * C_i.transpose();
  Eigen::Matrix3d Jj_r_r = C_i.transpose();
  // Jacobian of \Delta X^X with respect to X_j
  Eigen::Matrix<double, 15, 15> xi_j_x = Eigen::Matrix<double, 15, 15>::Zero();
  xi_j_x.block<3, 3>(0, 0) = Jj_C_C;
  xi_j_x.block<3, 3>(3, 0) = Jj_v_C;
  xi_j_x.block<3, 3>(3, 3) = Jj_v_v;
  xi_j_x.block<3, 3>(6, 0) = Jj_r_C;
  xi_j_x.block<3, 3>(6, 6) = Jj_r_r;
  xi_j_x.block<6, 6>(9, 9) = Eigen::Matrix<double, 6, 6>::Identity();
  Eigen::Matrix<double, 15, 15> jac_j = De_D_delta_xx * xi_j_x;

  std::vector<Eigen::Matrix<double, 15, 15>> raw_jacobians;
  raw_jacobians.push_back(jac_i);
  raw_jacobians.push_back(jac_j);
  return raw_jacobians;
}

std::vector<Eigen::Matrix<double, 15, 15>>
IMUPreintegrationHelper::computeRawJacobiansRightSE23(
    const IMUStateHolder &X_i, const IMUStateHolder &X_j) const {
  Eigen::Matrix<double, 5, 5> del_Xx_mat = predictNavRMI(X_i, X_j);

  // Extract required subcomponents of matrix
  Eigen::Matrix3d del_Xx_C = del_Xx_mat.block<3, 3>(0, 0);
  Eigen::Vector3d del_Xx_v = del_Xx_mat.block<3, 1>(0, 3);
  Eigen::Vector3d del_Xx_r = del_Xx_mat.block<3, 1>(0, 4);
  double delta_t = rmi.end_stamp - rmi.start_stamp;

  // Compute Jacobian of \Delta_X^X with respect to X_i
  Eigen::Matrix<double, 15, 15> xi_i_x = Eigen::Matrix<double, 15, 15>::Zero();
  xi_i_x.block<3, 3>(0, 0) = -del_Xx_C.transpose();
  xi_i_x.block<3, 3>(3, 0) = del_Xx_C.transpose() * SO3::cross(del_Xx_v);
  xi_i_x.block<3, 3>(3, 3) = -del_Xx_C.transpose();
  xi_i_x.block<3, 3>(6, 0) = del_Xx_C.transpose() * SO3::cross(del_Xx_r);
  xi_i_x.block<3, 3>(6, 3) = -delta_t * del_Xx_C.transpose();
  xi_i_x.block<3, 3>(6, 6) = -del_Xx_C.transpose();
  xi_i_x.block<6, 6>(9, 9) = -Eigen::Matrix<double, 6, 6>::Identity();

  // Compute Jacobian of error with respect to \Delta X^Y
  Eigen::Matrix<double, 6, 1> dbias;
  dbias.block<3, 1>(0, 0) = X_i.bias_gyro - rmi.gyro_bias;
  dbias.block<3, 1>(3, 0) = X_i.bias_accel - rmi.accel_bias;
  Eigen::Matrix<double, 9, 6> bias_jac = rmi.bias_jacobian;
  Eigen::Matrix<double, 9, 1> tau = bias_jac * dbias;
  Eigen::Matrix<double, 9, 6> Ji_X_b = SE23::rightJacobian(tau) * bias_jac;

  Eigen::Matrix<double, 15, 15> xi_i_y = Eigen::Matrix<double, 15, 15>::Zero();
  xi_i_y.block<9, 6>(0, 9) = Ji_X_b;

  // Jacobian of \Delta X^X with respect to X_j
  Eigen::Matrix<double, 15, 15> jac_j =
      Eigen::Matrix<double, 15, 15>::Identity();

  Eigen::Matrix<double, 15, 15> De_D_delta_xy =
      -Eigen::Matrix<double, 15, 15>::Identity();
  Eigen::Matrix<double, 15, 15> De_D_delta_xx =
      Eigen::Matrix<double, 15, 15>::Identity();

  if (use_group_jacobians) {
    Eigen::Matrix<double, 15, 1> error = computePreintegrationError(X_i, X_j);
    Eigen::Matrix<double, 9, 1> e_nav = error.block<9, 1>(0, 0);
    De_D_delta_xy.block<9, 9>(0, 0) = -SE23::leftJacobianInverse(e_nav);
    De_D_delta_xx.block<9, 9>(0, 0) = SE23::rightJacobianInverse(e_nav);
  }

  Eigen::Matrix<double, 15, 15> jac_i =
      De_D_delta_xy * xi_i_y + De_D_delta_xx * xi_i_x;
  jac_j = De_D_delta_xx * jac_j;

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
  double delta_t = rmi.end_stamp - rmi.start_stamp;

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
  Eigen::Matrix<double, 9, 6> bias_jac = rmi.bias_jacobian;

  Eigen::Matrix<double, 15, 15> D_delta_xy_Xi =
      Eigen::Matrix<double, 15, 15>::Zero();
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