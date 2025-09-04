#pragma once

#include <Eigen/Dense>
#include <ceres/ceres.h>

#include "lie/LieDirection.h"

namespace ceres_nav {

class RelativePoseFactor : public ceres::SizedCostFunction<6, 12, 12> {
public:
  Eigen::Matrix4d relative_pose_meas;
  // RelativePoseMessage relative_pose_meas;
  Eigen::Matrix<double, 6, 6> sqrt_info;
  LieDirection direction;

  RelativePoseFactor(const Eigen::Matrix4d &relative_pose_meas_,
                     const Eigen::Matrix<double, 6, 6> &sqrt_info,
                     const LieDirection &direction);

  /**
   * @brief Residual and Jacobian computation
   */
  virtual bool Evaluate(double const *const *parameters, double *residuals,
                        double **jacobians) const;
};

} // namespace ceres_nav