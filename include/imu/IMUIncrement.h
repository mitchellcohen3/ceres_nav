#pragma once

#include <Eigen/Dense>
#include <vector>
#include "lie/LieDirection.h"

class IMUIncrement {
public:
  // Initial gyro and accel bias
  Eigen::Vector3d gyro_bias;
  Eigen::Vector3d accel_bias;

  // Covariance and Jacobian that we'll propagate forward at each iteration
  Eigen::Matrix<double, 15, 15> covariance;
  Eigen::Matrix<double, 15, 15> jacobian;

  LieDirection direction;

  // Gravity vector
  Eigen::Vector3d gravity;
  // IMU noise parameters (continuous-time!)
  Eigen::Matrix<double, 12, 12> Q_ct;
  // RMI - NOTE: not quite an element of SE_2(3)!
  Eigen::Matrix<double, 5, 5> delta_U;
  // Bias Jacobian computed in compact form
  Eigen::Matrix<double, 9, 6> bias_jacobian;

  // Timestamp information
  double start_stamp;
  double end_stamp;
  double dt_total;

  // Store all measurements to repropagate if needed
  std::vector<double> dt_buf;
  std::vector<Eigen::Vector3d> acc_buf;
  std::vector<Eigen::Vector3d> gyr_buf;

  // Covariance propagation method
  std::string covariance_prop_method;

  IMUIncrement(const Eigen::Matrix<double, 12, 12> Q_ct,
               Eigen::Vector3d init_gyro_bias, Eigen::Vector3d init_accel_bias,
               double init_stamp, const Eigen::Vector3d &gravity,
               const std::string &covariance_prop_method, const LieDirection &direction);

  IMUIncrement(){};

  void reset(double new_start_stamp,
             const Eigen::Vector3d &new_gyro_bias = Eigen::Vector3d::Zero(),
             const Eigen::Vector3d &new_accel_bias = Eigen::Vector3d::Zero());

  // Add measurements to buffer and then call propagate
  void pushBack(double dt, const Eigen::Vector3d &omega,
                const Eigen::Vector3d &accel);

  // Propagate forward the RMI, bias Jacobians, and covariance
  void propagate(double dt, const Eigen::Vector3d &gyro,
                 const Eigen::Vector3d &acc);

  void propagateCovarianceAndBiasJacContinuous(double dt, const Eigen::Vector3d &omega,
                                               const Eigen::Vector3d &accel);

  void symmetrize();
  Eigen::Matrix<double, 5, 5> getDeltaX();

  // Repropagate all relevant quantities from a new initial bias
  void repropagate(const Eigen::Vector3d &init_gyro_bias,
                   const Eigen::Vector3d &init_accel_bias);
};
