#pragma once

#include <Eigen/Dense>
#include <cmath>

#include "lie/LieDirection.h"

namespace ceres_nav {

class SO3 {
public:
  static Eigen::Matrix3d cross(const Eigen::Vector3d &x);
  static Eigen::Vector3d vee(const Eigen::Matrix3d &element_so3);
  static Eigen::Matrix3d expMap(const Eigen::Vector3d &phi);
  static Eigen::Vector3d logMap(const Eigen::Matrix3d &element_so3);
  static Eigen::Matrix3d leftJacobian(const Eigen::Vector3d &phi);
  static Eigen::Matrix3d rightJacobian(const Eigen::Vector3d &phi);
  static Eigen::Matrix3d leftJacobianInverse(const Eigen::Vector3d &phi);
  static Eigen::Matrix3d rightJacobianInverse(const Eigen::Vector3d &phi);
  static Eigen::Matrix<double, 9, 1> flatten(const Eigen::Matrix3d C);
  static Eigen::Matrix3d unflatten(const Eigen::Matrix<double, 9, 1> vec_C);
  static Eigen::Vector3d toEuler(const Eigen::Matrix3d &C);

  /**
   * @brief Computes the minus operator between two SO3 elements,
   * for both left and right Lie direction.
   */
  static Eigen::Vector3d minus(const Eigen::Matrix3d &Y,
                               const Eigen::Matrix3d &X,
                               const LieDirection &direction);
};

} // namespace ceres_nav