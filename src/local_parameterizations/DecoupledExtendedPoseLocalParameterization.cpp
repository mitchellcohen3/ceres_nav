#include "local_parameterizations/DecoupledExtendedPoseLocalParameterization.h"
#include "lie/SE23.h"
#include "lie/SO3.h"

namespace ceres_nav {

/**
 * ExtendedPoseLocalParameterization::Plus defines the update rule for elements
 * of SE_2(3). This function defines how to increment parameters x, given a
 * small increment delta. The size of x is the GlobalSize of the parameter block
 * (10), while the size of delta is the LocalSize of the parameter block (9).
 * The variable x_plus_delta encodes how the global parameterization changes due
 * to an increment delta and it's size is the GlobalSize.
 */
bool DecoupledExtendedPoseLocalParameterization::Plus(
    const double *x, const double *delta, double *x_plus_delta) const {
  Eigen::Map<const Eigen::Matrix<double, 9, 1>> delta_xi_raw(delta); // SE23
  Eigen::Map<Eigen::Matrix<double, 15, 1>> x_plus_delta_raw(
      x_plus_delta); // Ceres

  Eigen::Matrix<double, 5, 5> X = SE23::fromCeresParameters(x);

  // Update the state
  Eigen::Matrix3d C_new;
  Eigen::Vector3d v_new;
  Eigen::Vector3d r_new;

  // Update the DCM with the correct perturbation
  if (_direction == LieDirection::left) {
    C_new = SO3::expMap(delta_xi_raw.block<3, 1>(0, 0)) * X.block<3, 3>(0, 0);
  }
  else if (_direction == LieDirection::right) {
    C_new = X.block<3, 3>(0, 0) * SO3::expMap(delta_xi_raw.block<3, 1>(0, 0));
  } else {
    std::cerr << "Invalid Lie direction" << std::endl;
  }

  // Update the velocity and position
  v_new = X.block<3, 1>(0, 3) + delta_xi_raw.block<3, 1>(3, 0);
  r_new = X.block<3, 1>(0, 4) + delta_xi_raw.block<3, 1>(6, 0);

  // Return the updated state
  x_plus_delta_raw.block<9, 1>(0, 0) = SO3::flatten(C_new);
  x_plus_delta_raw.block<3, 1>(9, 0) = v_new;
  x_plus_delta_raw.block<3, 1>(12, 0) = r_new;
  return true;
}
} // namespace ceres_nav
