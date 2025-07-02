#pragma once
#include <Eigen/Dense>
#include <cmath>

#include "lie/LieDirection.h"

class SE3 {
public:
  static constexpr float small_angle_tol = 1e-7;

  static Eigen::Matrix<double, 4, 4> expMap(const Eigen::Matrix<double, 6, 1> &x);
  static Eigen::Matrix<double, 6, 1> logMap(const Eigen::MatrixXd &X);
  static Eigen::MatrixXd fromComponents(const Eigen::Matrix3d &C, const Eigen::Vector3d &r);

  static Eigen::Matrix<double, 4, 4> fromCeresParameters(double const *parameters);
  static void toComponents(const Eigen::Matrix<double, 4, 4> &X, Eigen::Matrix3d &C,
                           Eigen::Vector3d &r);
  static Eigen::Matrix<double, 4, 4> inverse(const Eigen::Matrix<double, 4, 4> &X);
  static Eigen::Matrix<double, 6, 1> minus(const Eigen::Matrix<double, 4, 4> &X1,
                                           const Eigen::Matrix<double, 4, 4> &X2);
  static Eigen::Matrix<double, 6, 6> leftJacobian(const Eigen::VectorXd &x);
  static Eigen::Matrix<double, 6, 6> leftJacobianInverse(const Eigen::VectorXd &x);
  static Eigen::Matrix<double, 6, 6> rightJacobian(const Eigen::VectorXd &X);
  static Eigen::Matrix<double, 6, 6> rightJacobianInverse(const Eigen::VectorXd &x);
  static Eigen::Matrix3d leftJacobianQMatrix(const Eigen::Vector3d &phi,
                                             const Eigen::Vector3d &xi_r);

  static Eigen::Matrix<double, 6, 6> adjoint(const Eigen::MatrixXd &X);

  static Eigen::Matrix<double, 6, 1> minus(const Eigen::Matrix<double, 4, 4> &Y,
                                           const Eigen::Matrix<double, 4, 4> &X,
                                           LieDirection direction);
};