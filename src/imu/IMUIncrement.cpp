#include "imu/IMUIncrement.h"

#include "lie/SE23.h"
#include "lie/SO3.h"

#include "utils/Utils.h"

#include <cmath>

#include <glog/logging.h>

namespace ceres_nav {

IMUIncrement::IMUIncrement(const std::shared_ptr<IMUIncrementOptions> &options,
                           const Eigen::Vector3d &init_gyro_bias,
                           const Eigen::Vector3d &init_accel_bias)
    : options_{options}, gyro_bias{init_gyro_bias}, accel_bias{
                                                        init_accel_bias} {
  reset(0.0, init_gyro_bias, init_accel_bias);
}

void IMUIncrement::reset(double new_start_stamp,
                         const Eigen::Vector3d &new_gyro_bias,
                         const Eigen::Vector3d &new_accel_bias) {
  gyro_bias = new_gyro_bias;
  accel_bias = new_accel_bias;

  start_stamp = new_start_stamp;
  end_stamp = new_start_stamp;

  Upsilon_ij_.setIdentity();
  covariance_.setZero();
  jacobian_.setIdentity();

  dt_buf.clear();
  gyr_buf.clear();
  acc_buf.clear();
}

void IMUIncrement::propagate(double dt, const Eigen::Vector3d &omega,
                             const Eigen::Vector3d &accel) {
  dt_buf.push_back(dt);
  gyr_buf.push_back(omega);
  acc_buf.push_back(accel);

  end_stamp += dt;

  Eigen::Vector3d unbiased_gyro = omega - gyro_bias;
  Eigen::Vector3d unbiased_accel = accel - accel_bias;

  // Propagate covariance
  propagateCovarianceAndBiasJacobian(dt, omega, accel);
  // Propagate mean
  Eigen::Matrix<double, 5, 5> Upsilon_ip1 =
      upsilonMat(dt, unbiased_gyro, unbiased_accel, options_->discretization);
  Eigen::Matrix<double, 5, 5> Phi_dt = phiMat(dt, Upsilon_ij_);
  Upsilon_ij_ = Phi_dt * Upsilon_ip1;
}

void IMUIncrement::propagateCovarianceAndBiasJacobian(
    double dt, const Eigen::Vector3d &omega, const Eigen::Vector3d &accel) {
  // Compute continuous-time A matrix
  Eigen::Matrix3d C = Upsilon_ij_.block<3, 3>(0, 0);
  Eigen::Vector3d v = Upsilon_ij_.block<3, 1>(0, 3);
  Eigen::Vector3d r = Upsilon_ij_.block<3, 1>(0, 4);

  // Compute the continuous-time A and L matrices based on the
  // state representation and direction
  Eigen::Matrix<double, 15, 15> A_ct = Eigen::Matrix<double, 15, 15>::Zero();
  Eigen::Matrix<double, 15, 12> L_ct = Eigen::Matrix<double, 15, 12>::Zero();
  if (options_->pose_rep == ExtendedPoseRepresentation::SE23) {
    computeContinuousTimeJacobiansSE23(C, v, r, omega, accel, A_ct, L_ct);
  } else if (options_->pose_rep == ExtendedPoseRepresentation::Decoupled) {
    // LOG(INFO) << "Decoupled Jacobians...";
    computeContinuousTimeJacobiansDecoupled(C, v, r, omega, accel, A_ct, L_ct);
  }

  Eigen::Matrix<double, 12, 12> Q_ct =
      options_->continuousTimeNoiseCovariance();
  Eigen::MatrixXd A_d, Q_d;
  ceres_nav::discretizeSystem(A_ct, L_ct, Q_ct, dt, A_d, Q_d,
                              ceres_nav::DiscretizationMethod::TaylorSeries);
  // Symmetrize Q_d to avoid numerical issues
  Q_d = Q_d.selfadjointView<Eigen::Upper>();

  // Propagate Jacobian forward and extract bias portion
  jacobian_ = A_d * jacobian_;

  // Propagate covariance forward and symmetrize
  covariance_ = A_d * covariance_ * A_d.transpose() + Q_d;
  covariance_ = covariance_.selfadjointView<Eigen::Upper>();
}

void IMUIncrement::computeContinuousTimeJacobiansDecoupled(
    const Eigen::Matrix3d &C, const Eigen::Vector3d &v,
    const Eigen::Vector3d &r, const Eigen::Vector3d &omega,
    const Eigen::Vector3d &accel, Eigen::Matrix<double, 15, 15> &A_ct,
    Eigen::Matrix<double, 15, 12> &L_ct) {
  A_ct.setZero();
  L_ct.setZero();

  if (options_->direction == LieDirection::left) {
    LOG(ERROR) << "Left lie direction for decoupled navigation state "
                  "representation not yet supported with IMU increment!";
    std::exit(EXIT_FAILURE);
  } else if (options_->direction == LieDirection::right) {
    Eigen::Vector3d unbiased_gyro = omega - gyro_bias;
    Eigen::Vector3d unbiased_accel = accel - accel_bias;

    A_ct.block<3, 3>(0, 0) = -SO3::cross(unbiased_gyro);
    A_ct.block<3, 3>(0, 9) = -Eigen::Matrix3d::Identity();
    A_ct.block<3, 3>(3, 0) = -C * SO3::cross(unbiased_accel);
    A_ct.block<3, 3>(3, 12) = -C;
    A_ct.block<3, 3>(6, 3) = Eigen::Matrix3d::Identity();

    L_ct.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
    L_ct.block<3, 3>(3, 3) = C;
    L_ct.block<3, 3>(9, 6) = Eigen::Matrix3d::Identity();
    L_ct.block<3, 3>(12, 9) = Eigen::Matrix3d::Identity();
  }
}

void IMUIncrement::computeContinuousTimeJacobiansSE23(
    const Eigen::Matrix3d &C, const Eigen::Vector3d &v,
    const Eigen::Vector3d &r, const Eigen::Vector3d &omega,
    const Eigen::Vector3d &accel, Eigen::Matrix<double, 15, 15> &A_ct,
    Eigen::Matrix<double, 15, 12> &L_ct) {
  A_ct.setZero();
  L_ct.setZero();
  if (options_->direction == LieDirection::left) {
    A_ct.block<3, 3>(0, 9) = -C;
    A_ct.block<3, 3>(3, 9) = -SO3::cross(v) * C;
    A_ct.block<3, 3>(3, 12) = -C;
    A_ct.block<3, 3>(6, 3) = Eigen::Matrix3d::Identity();
    A_ct.block<3, 3>(6, 9) = -SO3::cross(r) * C;

    L_ct.block<3, 3>(0, 0) = C;
    L_ct.block<3, 3>(3, 0) = SO3::cross(v) * C;
    L_ct.block<3, 3>(3, 3) = C;
    L_ct.block<3, 3>(6, 0) = SO3::cross(r) * C;
    L_ct.block<3, 3>(9, 6) = Eigen::Matrix3d::Identity();
    L_ct.block<3, 3>(12, 9) = Eigen::Matrix3d::Identity();
  } else if (options_->direction == LieDirection::right) {
    // Compute unbiased gyro and accel
    Eigen::Vector3d unbiased_gyro = omega - gyro_bias;
    Eigen::Vector3d unbiased_accel = accel - accel_bias;

    A_ct.block<3, 3>(0, 0) = -SO3::cross(unbiased_gyro);
    A_ct.block<3, 3>(0, 9) = -Eigen::Matrix3d::Identity();
    A_ct.block<3, 3>(3, 0) = -SO3::cross(unbiased_accel);
    A_ct.block<3, 3>(3, 3) = -SO3::cross(unbiased_gyro);
    A_ct.block<3, 3>(3, 12) = -Eigen::Matrix3d::Identity();
    A_ct.block<3, 3>(6, 3) = Eigen::Matrix3d::Identity();
    A_ct.block<3, 3>(6, 6) = -SO3::cross(unbiased_gyro);

    L_ct.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
    L_ct.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity();
    L_ct.block<3, 3>(9, 6) = Eigen::Matrix3d::Identity();
    L_ct.block<3, 3>(12, 9) = Eigen::Matrix3d::Identity();
  }
}

void IMUIncrement::repropagate(const Eigen::Vector3d &init_gyro_bias,
                               const Eigen::Vector3d &init_accel_bias) {
  end_stamp = start_stamp;
  gyro_bias = init_gyro_bias;
  accel_bias = init_accel_bias;

  // Set delta_U and Jacobian to identity
  Upsilon_ij_.setIdentity();
  jacobian_.setIdentity();
  covariance_.setZero();

  // Repropagate using all measurements
  for (size_t i = 0; i < dt_buf.size(); ++i) {
    propagate(dt_buf[i], gyr_buf[i], acc_buf[i]);
  }
}

Eigen::Matrix<double, 5, 5> phiMat(double dt, const Eigen::Matrix<double, 5, 5> &T) {
  Eigen::Matrix<double, 5, 5> Phi = Eigen::Matrix<double, 5, 5>::Identity();
  Phi.block<3, 3>(0, 0) = T.block<3, 3>(0, 0);
  Phi.block<3, 1>(0, 3) = T.block<3, 1>(0, 3);
  Phi.block<3, 1>(0, 4) = T.block<3, 1>(0, 4) + dt * T.block<3, 1>(0, 3);
  return Phi;
}

Eigen::Matrix3d psiMat(const Eigen::Vector3d &omega) {
  if (omega.norm() < 1e-8) {
    return 0.5 * Eigen::Matrix3d::Identity();
  } else {
    double phi = omega.norm();
    Eigen::Matrix3d a_cross = SO3::cross(omega);
    double s = (phi - sin(phi)) / (phi * phi * phi);
    double c = (phi * phi + 2 * cos(phi) - 2) / (2 * phi * phi * phi * phi);
    return 0.5 * Eigen::Matrix3d::Identity() + s * a_cross +
           c * a_cross * a_cross;
  }
}

Eigen::Matrix<double, 5, 5> upsilonMat(double dt, const Eigen::Vector3d &omega,
                         const Eigen::Vector3d &accel, IMUDiscretizationMethod discretization) {
  Eigen::Matrix<double, 5, 5> Upsilon = Eigen::Matrix<double, 5, 5>::Identity();

  if (discretization ==
      IMUDiscretizationMethod::ConstantAccel) {
    LOG(ERROR)
        << "Constant Accel discretization not yet implemented in upsilonMat!";
    std::exit(EXIT_FAILURE);
  } else if (discretization ==
             IMUDiscretizationMethod::ConstantMeas) {
    Eigen::Matrix3d delta_C = SO3::expMap(omega * dt);
    Eigen::Matrix3d Psi_2 = psiMat(omega * dt);

    Upsilon.block<3, 3>(0, 0) = delta_C;
    Upsilon.block<3, 1>(0, 3) = dt * SO3::leftJacobian(omega * dt) * accel;
    Upsilon.block<3, 1>(0, 4) = dt * dt * Psi_2 * accel;
  }
  else {
    LOG(ERROR) << "Unknown discretization method in upsilonMat!";
    std::exit(EXIT_FAILURE);
  }

  return Upsilon;
}

} // namespace ceres_nav