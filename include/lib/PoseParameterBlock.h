#pragma once

#include "ParameterBlock.h"
#include "lie/LieDirection.h"
#include "lie/SE3.h"
#include "lie/SO3.h"
#include "local_parameterizations/PoseLocalParameterization.h"

/**
 * @brief Parameter block for SE(3) poses.
 *
 * The global state is represented as a 12D vector - the first 9 elements
 * represent the rotation in SO(3) flattened form, and the last 3 elements
 * represent the position in R^3.
 *
 * The PoseLocalParameterization allows for the pose to be updated using either
 * a left or right perturbation, written as T = Exp(delta_xi) * T_bar, (left
 * perturbation) T = T_bar * Exp(delta_xi), (right perturbation)
 */
class PoseParameterBlock : public ParameterBlock<12, 6> {
public:
  explicit PoseParameterBlock(const std::string &name = "pose_parameter_block",
                              LieDirection direction = LieDirection::left)
      : ParameterBlock<12, 6>(name) {
    // Create a local parameterization for SE(3) poses
    local_parameterization_ptr_ = new PoseLocalParameterization(direction);
  }

  // Construct directly from a pose in SE(3)
  PoseParameterBlock(const Eigen::Matrix4d &pose, const std::string &name = "pose_parameter_block",
                     LieDirection direction = LieDirection::left)
      : PoseParameterBlock(name, direction) {
    Eigen::Matrix3d C = pose.block<3, 3>(0, 0);
    Eigen::Vector3d r = pose.block<3, 1>(0, 3);
    setFromAttitudeAndPosition(C, r);
  } 
  /**
   * @brief Set the estimate from rotation and position.
   */

  void setFromAttitudeAndPosition(const Eigen::Matrix3d &attitude,
                                  const Eigen::Vector3d &position) {
    Eigen::Matrix<double, 9, 1> flattened_C = SO3::flatten(attitude);
    Eigen::Matrix<double, 12, 1> estimate;
    estimate << flattened_C, position;
    this->setEstimate(Eigen::VectorXd(estimate));
  }

  /**
   * @brief Set the estimate from a pose in SE(3).
   */
  void setFromPose(const Eigen::Matrix4d &pose) {
    Eigen::Matrix3d C = pose.block<3, 3>(0, 0);
    Eigen::Vector3d r = pose.block<3, 1>(0, 3);
    setFromAttitudeAndPosition(C, r);
  }

  /**
   * @brief Get the rotation estimate
   */
  Eigen::Matrix3d attitude() const {
    Eigen::Matrix<double, 12, 1> estimate = this->getEstimate();
    Eigen::Matrix<double, 9, 1> flattened_C = estimate.head<9>();
    return SO3::unflatten(flattened_C);
  }

  /**
   * @brief Get the position estimate
   */
  Eigen::Vector3d position() const {
    Eigen::Matrix<double, 12, 1> estimate = this->getEstimate();
    return estimate.tail<3>();
  }

  Eigen::Matrix4d pose() const {
    Eigen::Matrix3d C = attitude();
    Eigen::Vector3d r = position();
    return SE3::fromComponents(C, r);
  }

  // The plus operator
  virtual void plus(const double *x0, const double *Delta_Chi,
                    double *x0_plus_Delta) const override final {
    local_parameterization_ptr_->Plus(x0, Delta_Chi, x0_plus_Delta);
  }

  // Plus Jacobian
  virtual void plusJacobian(const double *x0,
                            double *jacobian) const override final {
    local_parameterization_ptr_->ComputeJacobian(x0, jacobian);
  }
};