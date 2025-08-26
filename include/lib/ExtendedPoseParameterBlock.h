#pragma once

#include "ParameterBlock.h"
#include "lie/LieDirection.h"
#include "lie/SE23.h"
#include "lie/SE3.h"
#include "lie/SO3.h"

#include "local_parameterizations/DecoupledExtendedPoseLocalParameterization.h"
#include "local_parameterizations/ExtendedPoseLocalParameterization.h"
#include "local_parameterizations/SE23LocalParameterization.h"

enum class ExtendedPoseRepresentation { SE23, Decoupled };

/**
 * @brief Parameter block for SE_2(3) poses, containing attitude, velocity, and
 * position.
 *
 * Similarly to PoseParameterBlock, this class represents an extended pose as
 * a 15D vector, where the first 9 elements represent the rotation in SO(3)
 * flattened form, the next 3 elements represent the velocity in R^3, and the
 * last 3 elements represent the position in R^3.
 *
 * Like the PoseParameterBlock, the ExtendedPoseParameterBlock allows for
 * the pose to be updated using either a left or right perturbation.
 */
class ExtendedPoseParameterBlock : public ParameterBlock<15, 9> {
public:
  explicit ExtendedPoseParameterBlock(
      ExtendedPoseRepresentation param_type = ExtendedPoseRepresentation::SE23,
      const std::string &name = "extended_pose_parameter_block",
      LieDirection direction = LieDirection::left)
      : ParameterBlock<15, 9>(name), direction_(direction) {
    switch (param_type) {
    case ExtendedPoseRepresentation::SE23:
      local_parameterization_ptr_ = new SE23LocalParameterization(direction);
      break;
    case ExtendedPoseRepresentation::Decoupled:
      local_parameterization_ptr_ =
          new DecoupledExtendedPoseLocalParameterization(direction);
      break;
    default:
      throw std::invalid_argument("Unsupported state representation type");
    }
  }

  /**
   * @brief Constructor directly from a nav state
   */
  ExtendedPoseParameterBlock(
      const Eigen::Matrix<double, 5, 5> &nav_state,
      ExtendedPoseRepresentation param_type = ExtendedPoseRepresentation::SE23,
      const std::string &name = "extended_pose_parameter_block",
      LieDirection direction = LieDirection::left)
      : ExtendedPoseParameterBlock(param_type, name, direction) {
    Eigen::Matrix3d C_ab = nav_state.block<3, 3>(0, 0);
    Eigen::Vector3d v = nav_state.block<3, 1>(0, 3);
    Eigen::Vector3d r = nav_state.block<3, 1>(0, 4);
    setFromComponents(C_ab, v, r);
  }

  /**
   * @brief Set the estimate from rotation and position.
   */

  void setFromComponents(const Eigen::Matrix3d &attitude,
                         const Eigen::Vector3d &velocity,
                         const Eigen::Vector3d &position) {
    Eigen::Matrix<double, 9, 1> flattened_C = SO3::flatten(attitude);
    Eigen::Matrix<double, 15, 1> estimate;
    estimate << flattened_C, velocity, position;
    this->setEstimate(Eigen::VectorXd(estimate));
  }

  /**
   * @brief Get the rotation estimate
   */
  Eigen::Matrix3d attitude() const {
    Eigen::Matrix<double, 15, 1> estimate = this->getEstimate();
    Eigen::Matrix<double, 9, 1> flattened_C = estimate.head<9>();
    return SO3::unflatten(flattened_C);
  }

  Eigen::Vector3d velocity() const {
    Eigen::Matrix<double, 15, 1> estimate = this->getEstimate();
    return estimate.segment<3>(9); // Velocity is stored in the next 3 elements
  }

  /**
   * @brief Get the position estimate
   */
  Eigen::Vector3d position() const {
    Eigen::Matrix<double, 15, 1> estimate = this->getEstimate();
    return estimate.tail<3>();
  }

  Eigen::Matrix<double, 5, 5> extendedPose() const {
    Eigen::Matrix3d C = attitude();
    Eigen::Vector3d v = velocity();
    Eigen::Vector3d r = position();
    return SE23::fromComponents(C, v, r);
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

  LieDirection direction() const { return direction_; }

protected:
  LieDirection direction_;
};