#pragma once

#include <Eigen/Dense>
#include <ceres/ceres.h>
#include <vector>

#include "lie/LieDirection.h"

class AbsolutePositionFactor : public ceres::CostFunction {
public:
  LieDirection direction;
  Eigen::Vector3d meas;
  Eigen::Matrix3d sqrt_info;
  bool print_debug = false;
  // Whether or not the input state is SE3 or SE23
  std::string pose_type = "SE23";

  AbsolutePositionFactor(const Eigen::Vector3d &meas_,
                         const LieDirection &direction_,
                         const Eigen::Matrix3d &sqrt_info_,
                         const std::string &pose_type_ = "SE23",
                         bool print_debug_ = false)
      : meas{meas_}, direction{direction_}, sqrt_info{sqrt_info_},
        print_debug{print_debug_}, pose_type{pose_type_} {
    // Set the number of residuals and parameters
    set_num_residuals(3);
    if (pose_type == "SE3") {
      mutable_parameter_block_sizes()->push_back(12);
    } else if (pose_type == "SE23") {
      mutable_parameter_block_sizes()->push_back(15);
    } else {
      throw std::runtime_error("Unknown pose type");
    }
  }

  /**
   * @brief Residual and Jacobian computation
   */
  bool Evaluate(double const *const *parameters, double *residuals,
                double **jacobians) const override;
};
