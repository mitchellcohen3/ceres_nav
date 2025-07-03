#pragma once

#include "lie/LieDirection.h"
#include <Eigen/Dense>
#include <ceres/ceres.h>

class RelativeLandmarkFactor : public ceres::CostFunction {
public:
  Eigen::Vector3d meas;
  Eigen::Matrix3d sqrt_info;
  double stamp;
  int landmark_id;
  bool print_debug_info = false;
  LieDirection direction;
  std::string pose_type;

  RelativeLandmarkFactor(const Eigen::Vector3d &meas_,
                         const Eigen::Matrix3d &sqrt_info_,
                         const double &stamp_, const int &landmark_id_,
                         const LieDirection &direction_,
                         const std::string &pose_type_)
      : meas{meas_}, sqrt_info{sqrt_info_}, stamp{stamp_},
        landmark_id{landmark_id_}, direction{direction_}, pose_type{
                                                              pose_type_} {
    set_num_residuals(3);
    if (pose_type == "SE3") {
      mutable_parameter_block_sizes()->push_back(12);
    } else if (pose_type == "SE23") {
      mutable_parameter_block_sizes()->push_back(15);
    } else {
      throw std::runtime_error("Unknown pose type");
    }
    mutable_parameter_block_sizes()->push_back(3);
  }

  /**
   * @brief Residual and Jacobian computation
   */
  bool Evaluate(double const *const *parameters, double *residuals,
                double **jacobians) const;
};