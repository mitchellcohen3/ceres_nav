#include "local_parameterizations/PoseLocalParameterization.h"
#include "lie/SE3.h"
#include "lie/SO3.h"

/**
 * ExtendedPoseLocalParameterization::Plus defines the update rule for elements
 * of SE_2(3). This function defines how to increment parameters x, given a
 * small increment delta. The size of x is the GlobalSize of the parameter block
 * (10), while the size of delta is the LocalSize of the parameter block (9).
 * The variable x_plus_delta encodes how the global parameterization changes due
 * to an increment delta and it's size is the GlobalSize.
 */
bool PoseLocalParameterization::Plus(const double *x, const double *delta,
                                     double *x_plus_delta) const {
  Eigen::Map<const Eigen::Matrix<double, 6, 1>> delta_xi_raw(delta);
  Eigen::Map<Eigen::Matrix<double, 12, 1>> x_plus_delta_raw(x_plus_delta);

  Eigen::Matrix<double, 4, 4> X = SE3::fromCeresParameters(x);

  // Perform update
  Eigen::Matrix<double, 4, 4> X_new;
  if (_direction == LieDirection::left) {
    X_new = SE3::expMap(delta_xi_raw) * X;
  } else if (_direction == LieDirection::right) {
    X_new = X * SE3::expMap(delta_xi_raw);
  } else {
    std::cerr << "Invalid Lie direction" << std::endl;
  }
  Eigen::Matrix3d C_new;
  Eigen::Vector3d r_new;
  SE3::toComponents(X_new, C_new, r_new);

  // Store updated result
  x_plus_delta_raw.block<9, 1>(0, 0) = SO3::flatten(C_new);
  x_plus_delta_raw.block<3, 1>(9, 0) = r_new;

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
bool PoseLocalParameterization::ComputeJacobian(const double *x, double *jacobian) const {
  Eigen::Map<Eigen::Matrix<double, 12, 6, Eigen::RowMajor>> j(jacobian);
  j = getEigenJacobian();

  // std::cout << j << std::endl;
  return true;
}

Eigen::Matrix<double, 12, 6> PoseLocalParameterization::getEigenJacobian() const {
  Eigen::Matrix<double, 12, 6> jac;
  jac.setZero();
  jac.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
  jac.block<3, 3>(9, 3) = Eigen::Matrix<double, 3, 3>::Identity();

  return jac;
}
