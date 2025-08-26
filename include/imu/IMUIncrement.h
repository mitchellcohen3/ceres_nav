#pragma once

#include "lib/ExtendedPoseParameterBlock.h"
#include "lie/LieDirection.h"
#include <Eigen/Dense>
#include <vector>

class IMUIncrement {
public:
  // Initial gyro and accel bias
  Eigen::Vector3d gyro_bias;
  Eigen::Vector3d accel_bias;

  // Covariance and Jacobian that we'll propagate forward at each iteration
  Eigen::Matrix<double, 15, 15> covariance;
  Eigen::Matrix<double, 15, 15> jacobian;

  // Navigation state representation options
  LieDirection direction;
  ExtendedPoseRepresentation pose_rep = ExtendedPoseRepresentation::SE23;

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

  IMUIncrement(const Eigen::Matrix<double, 12, 12> Q_ct,
               Eigen::Vector3d init_gyro_bias, Eigen::Vector3d init_accel_bias,
               double init_stamp, const Eigen::Vector3d &gravity,
               const LieDirection &direction,
               const ExtendedPoseRepresentation &pose_rep =
                   ExtendedPoseRepresentation::SE23);

  // IMUIncrement(){};

  void reset(double new_start_stamp,
             const Eigen::Vector3d &new_gyro_bias = Eigen::Vector3d::Zero(),
             const Eigen::Vector3d &new_accel_bias = Eigen::Vector3d::Zero());

  // Add measurements to buffer and then call propagate
  void pushBack(double dt, const Eigen::Vector3d &omega,
                const Eigen::Vector3d &accel);

  // Propagate forward the RMI, bias Jacobians, and covariance
  void propagate(double dt, const Eigen::Vector3d &gyro,
                 const Eigen::Vector3d &acc);

  Eigen::Matrix<double, 5, 5> getDeltaX();

  // Repropagate all relevant quantities from a new initial bias
  void repropagate(const Eigen::Vector3d &init_gyro_bias,
                   const Eigen::Vector3d &init_accel_bias);

protected:
  void symmetrize();

  /**
   * @brief Propagates forward the covariance and bias Jacobian.
   */
  void propagateCovarianceAndBiasJacobian(double dt,
                                          const Eigen::Vector3d &omega,
                                          const Eigen::Vector3d &accel);

  // Helper functions to compute the continuous-time A and L matrices
  void computeContinuousTimeJacobiansSE23(const Eigen::Matrix3d &C,
                                          const Eigen::Vector3d &v,
                                          const Eigen::Vector3d &r,
                                          const Eigen::Vector3d &omega,
                                          const Eigen::Vector3d &accel,
                                          Eigen::Matrix<double, 15, 15> &A_ct,
                                          Eigen::Matrix<double, 15, 12> &L_ct);

  void computeContinuousTimeJacobiansDecoupled(
      const Eigen::Matrix3d &C, const Eigen::Vector3d &v,
      const Eigen::Vector3d &r, const Eigen::Vector3d &omega,
      const Eigen::Vector3d &accel, Eigen::Matrix<double, 15, 15> &A_ct,
      Eigen::Matrix<double, 15, 12> &L_ct);
};
