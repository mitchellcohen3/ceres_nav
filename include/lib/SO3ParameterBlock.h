#pragma once

#include "ParameterBlock.h"
#include "lie/LieDirection.h"
#include "lie/SO3.h"

#include "local_parameterizations/SO3LocalParameterization.h"

namespace ceres_nav {

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
      : ParameterBlock<9, 3>(name), direction_(direction) {
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

  virtual void minus(const double *y_, const double *x_,
                     double *y_minus_x_) const override final {
    Eigen::Map<const Eigen::Matrix<double, 9, 1>> y(y_);
    Eigen::Map<const Eigen::Matrix<double, 9, 1>> x(x_);
    Eigen::Map<Eigen::Matrix<double, 3, 1>> y_minus_x(y_minus_x_);

    Eigen::Matrix3d Y = SO3::unflatten(y);
    Eigen::Matrix3d X = SO3::unflatten(x);
    y_minus_x = SO3::minus(Y, X, direction_);
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

protected:
  LieDirection direction_;
};

} // namespace ceres_nav