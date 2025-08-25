#pragma once

#include <Eigen/Dense>
#include <ceres/ceres.h>
#include <vector>

#include "imu/IMUIncrement.h"
#include "imu/IMUPreintegrationHelper.h"

class IMUPreintegrationFactor
    : public ceres::SizedCostFunction<15, 15, 6, 15, 6> {
public:
  // // The RMI
  // IMUIncrement rmi;

  // // Navigation state representation options
  // // This is used to determine how the Jacobians are computed
  // LieDirection direction;
  // ExtendedPoseRepresentation pose_rep = ExtendedPoseRepresentation::SE23;

  // bool use_group_jacobians;

  // Helper class to compute the residuals and Jacobians
  IMUPreintegrationHelper helper;

  /*
   * @brief Default constructor
   * Class constructor - each IMU factor needs an interoceptive measurement, a
   * time, and a continuous-time noise matrix.
   */
  IMUPreintegrationFactor(
      IMUIncrement imu_increment_, bool use_group_jacobians_,
      const LieDirection &direction_,
      ExtendedPoseRepresentation pose_rep_ = ExtendedPoseRepresentation::SE23);

  IMUPreintegrationFactor() = delete;
  /**
   * @brief Residual and Jacobian computation
   */
  virtual bool Evaluate(double const *const *parameters, double *residuals,
                        double **jacobians) const;
};
