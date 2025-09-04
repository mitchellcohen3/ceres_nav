#pragma once

#include <Eigen/Dense>
#include <vector>

#include "imu/IMUIncrement.h"

namespace ceres_nav {
/**
 * @brief Simple struct to hold the IMU state components.
 */
struct IMUStateHolder {
  Eigen::Matrix3d attitude;
  Eigen::Vector3d velocity;
  Eigen::Vector3d position;
  Eigen::Vector3d bias_gyro;
  Eigen::Vector3d bias_accel;

  IMUStateHolder() = default;

  IMUStateHolder(const Eigen::Matrix3d &att, const Eigen::Vector3d &vel,
                 const Eigen::Vector3d &pos, const Eigen::Vector3d &bg,
                 const Eigen::Vector3d &ba)
      : attitude{att}, velocity{vel}, position{pos}, bias_gyro{bg}, bias_accel{
                                                                        ba} {}
};

/**
 * @brief Helper class to comptue IMU preintegration residuals and Jacobians.
 *
 * This class contains methods to compute everything required for the
 * preintegration factor for various state representations and perturbation
 * directions.
 */
class IMUPreintegrationHelper {
public:
  IMUPreintegrationHelper(const IMUIncrement &imu_increment,
                          bool use_group_jacobians);

  // Main method to compute Jacobians based on representation and direction
  std::vector<Eigen::Matrix<double, 15, 15>>
  computeRawJacobians(const IMUStateHolder &X_i,
                      const IMUStateHolder &X_j) const;

  // SE23 Jacobian methods (both left and right)
  std::vector<Eigen::Matrix<double, 15, 15>>
  computeRawJacobiansLeftSE23(const IMUStateHolder &X_i,
                              const IMUStateHolder &X_j) const;

  std::vector<Eigen::Matrix<double, 15, 15>>
  computeRawJacobiansRightSE23(const IMUStateHolder &X_i,
                               const IMUStateHolder &X_j) const;

  // Decoupled representation Jacobian methods
  // TODO: Still need to implement Left Decoupled
  std::vector<Eigen::Matrix<double, 15, 15>>
  computeRawJacobiansRightDecoupled(const IMUStateHolder &X_i,
                                    const IMUStateHolder &X_j) const;

  // Methods to compute the RMI error
  Eigen::Matrix<double, 5, 5> getUpdatedRMI(const IMUStateHolder &X_i) const;
  Eigen::Matrix<double, 5, 5> predictNavRMI(const IMUStateHolder &X_i,
                                            const IMUStateHolder &X_j) const;
  Eigen::Matrix<double, 15, 1>
  computePreintegrationError(const IMUStateHolder &X_i,
                             const IMUStateHolder &X_j) const;

  // Gets the covariance of the preintegrated measurement
  Eigen::Matrix<double, 15, 15> covariance() const { return rmi.covariance; }

  double startStamp() const { return rmi.start_stamp; }
  double endStamp() const { return rmi.end_stamp; }

private:
  const IMUIncrement rmi;
  bool use_group_jacobians;
  LieDirection direction;
  ExtendedPoseRepresentation pose_rep;
};

} // namespace ceres_nav