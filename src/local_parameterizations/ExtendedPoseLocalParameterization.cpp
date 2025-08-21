#include "local_parameterizations/ExtendedPoseLocalParameterization.h"

/*
 * This function computes the Jacobian of the global parameterization w.r.t the
 * local parameterization. Within each cost function, the user is expected to
 * supply the Jacobian of the residual with respect to the global
 * parameterization. Interally, Ceres chain rules these together to supply the
 * Jacobian of the residual with respect to the local parameterization.
 *
 */
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
