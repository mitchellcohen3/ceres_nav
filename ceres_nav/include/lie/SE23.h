#pragma once

#include <Eigen/Dense>
#include <cmath>
#include "lie/LieDirection.h"

class SE23 {
public:
  static constexpr float small_angle_tol = 1e-7;

  static Eigen::Matrix<double, 5, 5>
  expMap(const Eigen::Matrix<double, 9, 1> &x);
  static Eigen::Matrix<double, 9, 1> logMap(const Eigen::MatrixXd &X);
  static Eigen::MatrixXd fromComponents(const Eigen::Matrix3d &C,
                                        const Eigen::Vector3d &v,
                                        const Eigen::Vector3d &r);
  static Eigen::Matrix<double, 5, 5>
  fromCeresParameters(double const *parameters);
  static void toComponents(const Eigen::Matrix<double, 5, 5> &X,
                           Eigen::Matrix3d &C, Eigen::Vector3d &v,
                           Eigen::Vector3d &r);
  static Eigen::Matrix<double, 5, 5>
  inverse(const Eigen::Matrix<double, 5, 5> &X);
  static Eigen::Matrix<double, 9, 9> leftJacobian(const Eigen::VectorXd &x);
  static Eigen::Matrix<double, 9, 9> rightJacobian(const Eigen::VectorXd &X);
  static Eigen::Matrix<double, 9, 9>
  leftJacobianInverse(const Eigen::VectorXd &x);
  static Eigen::Matrix<double, 9, 9>
  rightJacobianInverse(const Eigen::VectorXd &x);

  static Eigen::Matrix3d leftJacobianQMatrix(const Eigen::Vector3d &phi,
                                             const Eigen::Vector3d &xi_r);

  static Eigen::Matrix<double, 9, 9> adjoint(const Eigen::MatrixXd &X);
  static Eigen::Matrix<double, 15, 1>
  toCeresParameters(Eigen::Matrix<double, 5, 5> X);

  static Eigen::Matrix<double, 9, 1> minus(const Eigen::Matrix<double, 5, 5> &Y,
                                           const Eigen::Matrix<double, 5, 5> &X,
                                           LieDirection direction);
};