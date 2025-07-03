#include "local_parameterizations/ExtendedPoseLocalParameterization.h"
#include "lie/SE23.h"
#include "lie/SO3.h"

/**
 * ExtendedPoseLocalParameterization::Plus defines the update rule for elements
 * of SE_2(3). This function defines how to increment parameters x, given a
 * small increment delta. The size of x is the GlobalSize of the parameter block
 * (10), while the size of delta is the LocalSize of the parameter block (9).
 * The variable x_plus_delta encodes how the global parameterization changes due
 * to an increment delta and it's size is the GlobalSize.
 */
bool ExtendedPoseLocalParameterization::Plus(const double *x,
                                             const double *delta,
                                             double *x_plus_delta) const {
  Eigen::Map<const Eigen::Matrix<double, 9, 1>> delta_xi_raw(delta); // SE23
  Eigen::Map<Eigen::Matrix<double, 15, 1>> x_plus_delta_raw(
      x_plus_delta); // Ceres

  Eigen::Matrix<double, 5, 5> X = SE23::fromCeresParameters(x);

  // Update the state
  Eigen::Matrix<double, 5, 5> X_new;
  if (_direction == LieDirection::left){
    X_new = SE23::expMap(delta_xi_raw) * X;
  } else if (_direction == LieDirection::right){
    X_new = X * SE23::expMap(delta_xi_raw);
  } else {
    std::cerr << "Invalid Lie direction" << std::endl;
  }
  Eigen::Matrix3d C_new;
  Eigen::Vector3d v_new;
  Eigen::Vector3d r_new;
  SE23::toComponents(X_new, C_new, v_new, r_new);

  //  Store updated result
  x_plus_delta_raw.block<9, 1>(0, 0) = SO3::flatten(C_new);
  x_plus_delta_raw.block<3, 1>(9, 0) = v_new;
  x_plus_delta_raw.block<3, 1>(12, 0) = r_new;

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
//
bool ExtendedPoseLocalParameterization::ComputeJacobian(
    const double *x, double *jacobian) const {
  Eigen::Map<Eigen::Matrix<double, 15, 9, Eigen::RowMajor>> j(jacobian);
  j = getEigenJacobian();

  return true;
}

Eigen::Matrix<double, 15, 9>
ExtendedPoseLocalParameterization::getEigenJacobian() const {
  Eigen::Matrix<double, 15, 9> jac;
  jac.setZero();
  jac.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
  jac.block<6, 6>(9, 3) = Eigen::Matrix<double, 6, 6>::Identity();

  return jac;
}
