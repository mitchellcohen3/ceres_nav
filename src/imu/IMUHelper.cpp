#include "lie/SE23.h"
#include "lie/SO3.h"

#include "imu/IMUHelper.h"
#include "imu/IMUIncrement.h"

// #include <iostream>
Eigen::Matrix<double, 5, 5> createGMatrix(const Eigen::Vector3d &gravity,
                                          double dt) {
  Eigen::Matrix<double, 5, 5> G = Eigen::Matrix<double, 5, 5>::Identity();
  G.block<3, 1>(0, 3) = dt * gravity;
  G.block<3, 1>(0, 4) = -0.5 * dt * dt * gravity;
  G(3, 4) = -dt;
  return G;
}

Eigen::Matrix3d createNMatrix(const Eigen::Vector3d &phi_vec) {
  double small_angle_tol = 1e-7;
  double phi_norm = phi_vec.norm();
  if (phi_norm < small_angle_tol) {
    return Eigen::Matrix3d::Identity();
  } else {
    Eigen::Vector3d a = phi_vec / phi_norm;
    Eigen::Matrix3d a_cross = SO3::cross(a);
    double c = (1.0 - cos(phi_norm)) / (phi_norm * phi_norm);
    double s = (phi_norm - sin(phi_norm)) / (phi_norm * phi_norm);
    Eigen::Matrix3d N = 2 * c * Eigen::Matrix3d::Identity() +
                        (1 - 2 * c) * (a * a.transpose()) + (2 * s * a_cross);
    return N;
  }
}

Eigen::Matrix<double, 5, 5> createUMatrix(const Eigen::Vector3d &omega,
                                          const Eigen::Vector3d &accel,
                                          double dt) {
  Eigen::Matrix<double, 5, 5> U_mat = Eigen::Matrix<double, 5, 5>::Identity();
  Eigen::Vector3d phi = omega * dt;
  Eigen::Matrix3d O_mat = SO3::expMap(phi);
  Eigen::Matrix3d J_left = SO3::leftJacobian(phi);
  Eigen::Matrix3d V_mat = createNMatrix(phi);
  U_mat.block<3, 3>(0, 0) = O_mat;
  U_mat.block<3, 1>(0, 3) = dt * J_left * accel;
  U_mat.block<3, 1>(0, 4) = (0.5 * dt * dt) * V_mat * accel;
  U_mat(3, 4) = dt;
  return U_mat;
}

Eigen::Matrix<double, 5, 5> inverseIE3(const Eigen::Matrix<double, 5, 5> &X) {
  Eigen::Matrix3d R_mat = X.block<3, 3>(0, 0);
  double c = X(3, 4);
  Eigen::Vector3d a = X.block<3, 1>(0, 3);
  Eigen::Vector3d b = X.block<3, 1>(0, 4);
  Eigen::Matrix<double, 5, 5> X_inv = Eigen::Matrix<double, 5, 5>::Identity();
  X_inv.block<3, 3>(0, 0) = R_mat.transpose();
  X_inv.block<3, 1>(0, 3) = -R_mat.transpose() * a;
  X_inv.block<3, 1>(0, 4) = R_mat.transpose() * (c * a - b);
  X_inv(3, 4) = -c;
  return X_inv;
}

Eigen::Matrix<double, 5, 5> createUMatrixInv(const Eigen::Vector3d &omega,
                                             const Eigen::Vector3d &accel,
                                             double dt) {
  Eigen::Matrix<double, 5, 5> U_matrix = createUMatrix(omega, accel, dt);
  Eigen::Matrix<double, 5, 5> U_inv = inverseIE3(U_matrix);
  return U_inv;
}

Eigen::Matrix<double, 9, 6> createLMatrix(const Eigen::Vector3d &unbiased_gyro,
                                          const Eigen::Vector3d &unbiased_accel,
                                          double dt) {
  Eigen::Vector3d a = unbiased_accel;
  Eigen::Vector3d om = unbiased_gyro;
  Eigen::Vector3d om_dt = om * dt;
  Eigen::Matrix3d J_att_inv_times_N =
      SO3::leftJacobianInv(om_dt) * createNMatrix(om_dt);
  Eigen::VectorXd xi(9);
  xi.segment<3>(0) = dt * om;
  xi.segment<3>(3) = dt * a;
  xi.segment<3>(6) = (0.5 * dt * dt) * J_att_inv_times_N * a;
  Eigen::Matrix<double, 9, 9> J_left = SE23::leftJacobian(-xi);
  Eigen::Matrix3d Om = SO3::cross(om_dt);
  Eigen::Matrix3d OmOm = Om * Om;
  Eigen::Matrix3d A = SO3::cross(a);
  Eigen::Matrix<double, 9, 6> Up = Eigen::Matrix<double, 9, 6>::Zero();
  Up.block<6, 6>(0, 0) = Eigen::Matrix<double, 6, 6>::Identity() * dt;
  double coeff = -0.5 * (dt * dt / 2.0);
  double coeff2 = (1.0 / 360.0) * (dt * dt * dt);
  Up.block<3, 3>(6, 0) =
      coeff *
      (coeff2 * (OmOm * A + Om * (SO3::cross(Om * a) + SO3::cross(OmOm * a))) -
       (1.0 / 6.0) * dt * A);
  Up.block<3, 3>(6, 3) = (0.5 * dt * dt) * J_att_inv_times_N;
  auto L = J_left * Up;
  return L;
}

Eigen::Matrix<double, 9, 9> adjointIE3(const Eigen::Matrix<double, 5, 5> &X) {
  Eigen::Matrix3d R_mat = X.block<3, 3>(0, 0);
  double c = X(3, 4);
  Eigen::Vector3d a = X.block<3, 1>(0, 3);
  Eigen::Vector3d b = X.block<3, 1>(0, 4);
  Eigen::Matrix<double, 9, 9> adjoint;
  adjoint.setZero();
  adjoint.block<3, 3>(0, 0) = R_mat;
  adjoint.block<3, 3>(3, 0) = SO3::cross(a) * R_mat;
  adjoint.block<3, 3>(3, 3) = R_mat;

  adjoint.block<3, 3>(6, 0) = -SO3::cross(c * a - b) * R_mat;
  adjoint.block<3, 3>(6, 3) = -c * R_mat;
  adjoint.block<3, 3>(6, 6) = R_mat;

  return adjoint;
}

// bool preintegrateIMUMeasurements(IMUIncrement &rmi,
//                                  const std::vector<IMU> &imu_meas_vec) {
//   for (int i = 0; i < (imu_meas_vec.size() - 1); i++) {
//     double dt = imu_meas_vec[i + 1].stamp - imu_meas_vec[i].stamp;
//     IMU cur_meas = imu_meas_vec[i];
//     rmi.propagate(dt, cur_meas.gyro, cur_meas.accel);
//   }

//   return true;
// }

// std::vector<IMU> getIMUBetweenTimes(const double &stamp_i,
//                                     const double &stamp_j,
//                                     const std::vector<IMU> &imu_meas_vec) {
//   std::vector<IMU> imu_out;
//   for (auto const &imu_meas : imu_meas_vec) {
//     if ((imu_meas.stamp >= stamp_i) && (imu_meas.stamp <= stamp_j)) {
//       imu_out.push_back(imu_meas);
//     }
//   }

//   return imu_out;
// }

// bool preintegrateBetweenTimes(IMUIncrement &rmi, const double &stamp_i,
//                               const double &stamp_j,
//                               const std::vector<IMU> &imu_meas_vec) {
//   std::vector<IMU> imu_to_preintegrate =
//       getIMUBetweenTimes(stamp_i, stamp_j, imu_meas_vec);
//   preintegrateIMUMeasurements(rmi, imu_to_preintegrate);

//   return true;
// }