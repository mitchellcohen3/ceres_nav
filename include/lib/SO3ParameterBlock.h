#pragma once

#include "ParameterBlock.h"
#include "lie/LieDirection.h"
#include "lie/SO3.h"

#include "local_parameterizations/SO3LocalParameterization.h"

/**
 * @brief Parameter block for SO(3) rotations.
 *
 * The global state is represented as a 9D vector, representing the flattened
 * rotation matrix.
 *
 * The SO3LocalParameterization allows for the rotation to be updated using
 * either a left or right perturbation model, written as
 *      C = Exp(delta_xi) * C_bar (left perturbation)
 *      C = C_bar * Exp(delta_xi) (right perturbation)
 */
class SO3ParameterBlock : public ParameterBlock<9, 3> {
public:
  explicit SO3ParameterBlock(const std::string &name = "so3_param_block",
                             LieDirection direction = LieDirection::left)
      : ParameterBlock<9, 3>(name) {
    local_parameterization_ptr_ = new SO3LocalParameterization(direction);
  }

  // Construct directly from a DCM
  SO3ParameterBlock(const Eigen::Matrix3d &C,
                    const std::string &name = "so3_param_block",
                    LieDirection direction = LieDirection::left)
      : SO3ParameterBlock(name, direction) {
    setFromMatrix(C);
  }

  void setFromMatrix(const Eigen::Matrix3d &C) {
    Eigen::Matrix<double, 9, 1> flattened_C = SO3::flatten(C);
    this->setEstimate(Eigen::VectorXd(flattened_C));
  }

  Eigen::Matrix3d attitude() const {
    return SO3::unflatten(this->getEstimate());
  }

  // The plus operator
  virtual void plus(const double *x, const double *delta_xi,
                    double *x_plus_delta_xi) const override final {
    local_parameterization_ptr_->Plus(x, delta_xi, x_plus_delta_xi);
  }

  // Plus Jacobian
  virtual void plusJacobian(const double *x0,
                            double *jacobian) const override final {
    local_parameterization_ptr_->ComputeJacobian(x0, jacobian);
  }
};