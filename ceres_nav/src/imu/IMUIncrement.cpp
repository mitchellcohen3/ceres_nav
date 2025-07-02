#include "imu/IMUIncrement.h"
#include "imu/IMUHelper.h"

#include "lie/SE23.h"
#include "lie/SO3.h"

#include <cmath>
#include <iostream>

IMUIncrement::IMUIncrement(Eigen::Matrix<double, 12, 12> Q_ct_,
                           Eigen::Vector3d init_gyro_bias,
                           Eigen::Vector3d init_accel_bias, double init_stamp,
                           const Eigen::Vector3d &gravity_,
                           const std::string &covariance_prop_method_,
                           const LieDirection &direction_)
    : Q_ct{Q_ct_}, gyro_bias{init_gyro_bias}, accel_bias{init_accel_bias},
      start_stamp{init_stamp}, end_stamp{init_stamp}, gravity{gravity_},
      covariance_prop_method{covariance_prop_method_}, direction{direction_} {
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

  if (covariance_prop_method.compare("discrete") == 0) {
    std::cout << "Warning: Discrete covariance propagation not supported!"
              << std::endl;
  } else if (covariance_prop_method.compare("continuous") == 0) {
    propagateCovarianceAndBiasJacContinuous(dt, omega, accel);
  } else {
    std::cout << "Covariance propagation method undefined!" << std::endl;
  }

  symmetrize();

  // Update RMI
  delta_U = delta_U * U_mat;
}

void IMUIncrement::propagateCovarianceAndBiasJacContinuous(
    double dt, const Eigen::Vector3d &omega, const Eigen::Vector3d &accel) {
  // Compute continuous-time A matrix
  Eigen::Matrix3d C;
  Eigen::Vector3d v;
  Eigen::Vector3d r;
  SE23::toComponents(delta_U, C, v, r);

  Eigen::Matrix<double, 15, 15> A_ct = Eigen::Matrix<double, 15, 15>::Zero();
  Eigen::Matrix<double, 15, 12> L_ct = Eigen::Matrix<double, 15, 12>::Zero();
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

  // Discretize the system
  Eigen::Matrix<double, 15, 15> A_dt = A_ct * dt;
  Eigen::Matrix<double, 15, 15> A_dt_square = A_dt * A_dt;
  Eigen::Matrix<double, 15, 15> A_dt_cube = A_dt_square * A_dt;
  Eigen::Matrix<double, 15, 15> A_d =
      Eigen::Matrix<double, 15, 15>::Identity() + A_dt + 0.5 * A_dt_square +
      (1.0 / 6.0) * A_dt_cube;

  // Get discrete-time noise
  Eigen::Matrix<double, 15, 15> Q = L_ct * Q_ct * L_ct.transpose();
  Eigen::Matrix<double, 15, 15> first_term = Q * dt;
  Eigen::Matrix<double, 15, 15> second_term =
      (A_ct * Q + Q * A_ct.transpose()) * (dt * dt) / 2.0;
  Eigen::Matrix<double, 15, 15> third_term =
      (A_ct * A_ct * Q + 2.0 * A_ct * Q * A_ct.transpose() +
       Q * A_ct.transpose() * A_ct.transpose()) *
      (dt * dt * dt) / 6.0;
  Eigen::Matrix<double, 15, 15> Q_d = first_term + second_term + third_term;
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