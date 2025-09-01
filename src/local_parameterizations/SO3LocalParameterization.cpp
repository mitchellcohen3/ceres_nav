#include "local_parameterizations/SO3LocalParameterization.h"
#include "lie/SE3.h"
#include "lie/SO3.h"

/**
 * Defines the plus operator for elements of SO(3).
 * 
 * Here, 
 */
bool SO3LocalParameterization::Plus(const double *x, const double *delta,
                                     double *x_plus_delta) const {
  // Map inputs to Eigen types
  Eigen::Map<const Eigen::Matrix<double, 9, 1>> x_raw(x);
  Eigen::Map<const Eigen::Matrix<double, 3, 1>> delta_xi_raw(delta);
  Eigen::Map<Eigen::Matrix<double, 9, 1>> x_plus_delta_raw(x_plus_delta);

  // Update the DCM using either the left or right perturbation
  Eigen::Matrix3d C = SO3::unflatten(x_raw);
  Eigen::Matrix<double, 3, 3> C_new;
  if (_direction == LieDirection::left) {
    C_new = SO3::expMap(delta_xi_raw) * C;
  } else if (_direction == LieDirection::right) {
    C_new = C * SO3::expMap(delta_xi_raw);
  } else {
    std::cerr << "Invalid Lie direction" << std::endl;
  }

  // Store updated result
  x_plus_delta_raw.block<9, 1>(0, 0) = SO3::flatten(C_new);
  return true;
}

/*
 * This function computes the Jacobian of the global parameterization w.r.t the
 * local parameterization. Within each cost function, the user is expected to
 * supply the Jacobian of the residual with respect to the global
 * parameterization. Interally, Ceres chain rules these together to supply the
 * Jacobian of the residual with respect to the local parameterization.
 *
 */
bool SO3LocalParameterization::ComputeJacobian(const double *x, double *jacobian) const {
  Eigen::Map<Eigen::Matrix<double, 9, 3, Eigen::RowMajor>> j(jacobian);
  j = getEigenJacobian();
  return true;
}

Eigen::Matrix<double, 9, 3> SO3LocalParameterization::getEigenJacobian() const {
  Eigen::Matrix<double, 9, 3> jac = Eigen::Matrix<double, 9, 3>::Zero();
  jac.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
  return jac;
}
