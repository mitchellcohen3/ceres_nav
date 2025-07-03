#pragma once

#include <Eigen/Dense>
#include <ceres/ceres.h>
#include "lie/LieDirection.h"

class IMUPriorFactor : public ceres::SizedCostFunction<15, 15, 6> {
public:
  /**
   * @brief Construct the IMU prior factor
   */
  IMUPriorFactor(const Eigen::Matrix<double, 5, 5> &prior_nav_state,
                 const Eigen::Matrix<double, 6, 1> &prior_imu_bias,
                 const Eigen::Matrix<double, 15, 15> &prior_covariance,
                 LieDirection direction = LieDirection::left);

  virtual ~IMUPriorFactor() {}

  /**
   * @brief Error residual and Jacobian calculation
   */
  bool Evaluate(double const *const *parameters, double *residuals,
                double **jacobians) const override;

private:
  Eigen::Matrix<double, 5, 5> prior_nav_state_;
  Eigen::Matrix<double, 6, 1> prior_bias_;
  Eigen::Matrix<double, 15, 15> prior_covariance_;
  Eigen::Matrix<double, 15, 15> sqrt_info_;
  LieDirection direction_;
};