#pragma once

#include <Eigen/Dense>
#include <ceres/ceres.h>
#include <vector>

#include "imu/IMUIncrement.h"

class IMUStateHolder {
public:
  Eigen::Matrix3d attitude;
  Eigen::Vector3d velocity;
  Eigen::Vector3d position;
  Eigen::Vector3d bias_gyro;
  Eigen::Vector3d bias_accel;
};

class IMUPreintegrationFactor
    : public ceres::SizedCostFunction<15, 15, 6, 15, 6> {
public:
  IMUIncrement rmi;
  bool use_group_jacobians;
  LieDirection direction;

  /*
   * @brief Default constructor
   * Class constructor - each IMU factor needs an interoceptive measurement, a
   * time, and a continuous-time noise matrix.
   */
  IMUPreintegrationFactor(IMUIncrement imu_increment_,
                          bool use_group_jacobians_,
                          const LieDirection &direction_);

  IMUPreintegrationFactor() = delete;
  /**
   * @brief Residual and Jacobian computation
   */
  virtual bool Evaluate(double const *const *parameters, double *residuals,
                        double **jacobians) const;

  // Helper functions for error and Jacobian computation
  Eigen::Matrix<double, 15, 1> computeRawError(const IMUStateHolder &X_i,
                                               const IMUStateHolder &X_j) const;
  Eigen::Matrix<double, 5, 5> getUpdatedRMI(const IMUStateHolder &X_i) const;
  Eigen::Matrix<double, 5, 5> predictNavRMI(const IMUStateHolder &X_i,
                                            const IMUStateHolder &X_j) const;

  /** Functions to compute raw Jacobians of error function.*/
  std::vector<Eigen::Matrix<double, 15, 15>>
  computeRawJacobians(const IMUStateHolder &X_i,
                      const IMUStateHolder &X_j) const;

  std::vector<Eigen::Matrix<double, 15, 15>>
  computeRawJacobiansLeft(const IMUStateHolder &X_i, const IMUStateHolder &X_j) const;

  std::vector<Eigen::Matrix<double, 15, 15>>
  computeRawJacobiansRight(const IMUStateHolder &X_i,
                           const IMUStateHolder &X_j) const;
};
