#include "imu/IMUIncrement.h"
#include "imu/IMUHelper.h"

#include "lie/SE23.h"
#include "lie/SO3.h"

#include "utils/Utils.h"

#include <cmath>

#include <glog/logging.h>

IMUIncrement::IMUIncrement(Eigen::Matrix<double, 12, 12> Q_ct_,
                           Eigen::Vector3d init_gyro_bias,
                           Eigen::Vector3d init_accel_bias, double init_stamp,
                           const Eigen::Vector3d &gravity_,
                           const LieDirection &direction_,
                           const ExtendedPoseRepresentation &pose_rep_)
    : Q_ct{Q_ct_}, gyro_bias{init_gyro_bias}, accel_bias{init_accel_bias},
      start_stamp{init_stamp}, end_stamp{init_stamp}, gravity{gravity_},
      direction{direction_}, pose_rep{pose_rep_} {
  // Set bias Jacobians to zero
  bias_jacobian.setZero();

  // Set delta_U and Jacobian to identity
  delta_U.setIdentity();
  jacobian.setIdentity();
  covariance.setZero();
  dt_total = 0.0;
}

void IMUIncrement::reset(double new_start_stamp,
                         const Eigen::Vector3d &new_gyro_bias,
                         const Eigen::Vector3d &new_accel_bias) {
  gyro_bias = new_gyro_bias;
  accel_bias = new_accel_bias;
  start_stamp = new_start_stamp;
  end_stamp = new_start_stamp;
  dt_total = 0.0;
  delta_U.setIdentity();
  jacobian.setIdentity();
  covariance.setZero();
  dt_buf.clear();
  gyr_buf.clear();
  acc_buf.clear();
  bias_jacobian.setZero();
}

void IMUIncrement::pushBack(double dt, const Eigen::Vector3d &omega,
                            const Eigen::Vector3d &accel) {
  dt_buf.push_back(dt);
  gyr_buf.push_back(omega);
  acc_buf.push_back(accel);
  propagate(dt, omega, accel);
}

void IMUIncrement::propagate(double dt, const Eigen::Vector3d &omega,
                             const Eigen::Vector3d &accel) {
  dt_total += dt;
  end_stamp += dt;
  Eigen::Vector3d unbiased_gyro = omega - gyro_bias;
  Eigen::Vector3d unbiased_accel = accel - accel_bias;

  Eigen::Matrix<double, 5, 5> U_mat =
      createUMatrix(unbiased_gyro, unbiased_accel, dt);

  // Propagate forward the covariance and bias Jacobian
  propagateCovarianceAndBiasJacobian(dt, omega, accel);
  symmetrize();

  // Update RMI
  delta_U = delta_U * U_mat;
}

void IMUIncrement::propagateCovarianceAndBiasJacobian(
    double dt, const Eigen::Vector3d &omega, const Eigen::Vector3d &accel) {
  // Compute continuous-time A matrix
  Eigen::Matrix3d C = delta_U.block<3, 3>(0, 0);
  Eigen::Vector3d v = delta_U.block<3, 1>(0, 3);
  Eigen::Vector3d r = delta_U.block<3, 1>(0, 4);

  // Compute the continuous-time A and L matrices based on the
  // state representation and direction
  Eigen::Matrix<double, 15, 15> A_ct = Eigen::Matrix<double, 15, 15>::Zero();
  Eigen::Matrix<double, 15, 12> L_ct = Eigen::Matrix<double, 15, 12>::Zero();
  if (pose_rep == ExtendedPoseRepresentation::SE23) {
    computeContinuousTimeJacobiansSE23(C, v, r, omega, accel, A_ct, L_ct);
  } else if (pose_rep == ExtendedPoseRepresentation::Decoupled) {
    computeContinuousTimeJacobiansDecoupled(C, v, r, omega, accel, A_ct, L_ct);
  } else {
    LOG(FATAL) << "Unknown pose representation: " << static_cast<int>(pose_rep);
  }

  Eigen::MatrixXd A_d, Q_d;
  ceres_nav::discretizeSystem(A_ct, L_ct, Q_ct, dt, A_d, Q_d,
                   ceres_nav::DiscretizationMethod::TaylorSeries);
  Q_d = 0.5 * (Q_d + Q_d.transpose());

  // Propagate Jacobian forward and extract bias portion
  jacobian = A_d * jacobian;
  bias_jacobian = jacobian.block<9, 6>(0, 9);

  // Propagate covariance forward
  covariance = A_d * covariance * A_d.transpose() + Q_d;
}

void IMUIncrement::symmetrize() {
  covariance = 0.5 * (covariance + covariance.transpose());
}

void IMUIncrement::computeContinuousTimeJacobiansDecoupled(
    const Eigen::Matrix3d &C, const Eigen::Vector3d &v,
    const Eigen::Vector3d &r, const Eigen::Vector3d &omega,
    const Eigen::Vector3d &accel, Eigen::Matrix<double, 15, 15> &A_ct,
    Eigen::Matrix<double, 15, 12> &L_ct) {
  A_ct.setZero();
  L_ct.setZero();

  if (direction == LieDirection::left) {
    LOG(INFO) << "Left lie direction for decoupled navigation state "
                 "representation not yet supported with IMU increment!"; 
  } else if (direction == LieDirection::right) {
    Eigen::Vector3d unbiased_gyro = omega - gyro_bias;
    Eigen::Vector3d unbiased_accel = accel - accel_bias;

    A_ct.block<3, 3>(0, 0) = -SO3::cross(unbiased_gyro);
    A_ct.block<3, 3>(0, 9) = -Eigen::Matrix3d::Identity();
    A_ct.block<3, 3>(3, 0) = -C * SO3::cross(unbiased_accel);
    A_ct.block<3, 3>(0, 9) = -C;
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
  if (direction == LieDirection::left) {
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
  } else if (direction == LieDirection::right) {
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
  dt_total = 0.0;
  end_stamp = start_stamp;
  gyro_bias = init_gyro_bias;
  accel_bias = init_accel_bias;

  // Set bias Jacobians to zero
  bias_jacobian.setZero();
  // Set delta_U and Jacobian to identity
  delta_U.setIdentity();
  jacobian.setIdentity();
  covariance.setZero();

  // Repropagate using all measurements
  for (int i = 0; i < static_cast<int>(dt_buf.size()); i++) {
    propagate(dt_buf[i], gyr_buf[i], acc_buf[i]);
  }
}