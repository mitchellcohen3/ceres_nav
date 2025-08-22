#include "lie/SE23.h"
#include "lie/SO3.h"
#include <Eigen/Dense>
#include <iostream>

Eigen::Matrix<double, 5, 5> SE23::expMap(const Eigen::Matrix<double, 9, 1> &x) {
  Eigen::Matrix<double, 5, 5> X = Eigen::Matrix<double, 5, 5>::Identity();
  Eigen::Matrix3d R{SO3::expMap(x.block<3, 1>(0, 0))};
  Eigen::Vector3d xi_v{x.block<3, 1>(3, 0)};
  Eigen::Vector3d xi_r{x.block<3, 1>(6, 0)};
  Eigen::Matrix3d J{SO3::leftJacobian(x.block<3, 1>(0, 0))};
  X.block<3, 3>(0, 0) = R;
  X.block<3, 1>(0, 3) = J * xi_v;
  X.block<3, 1>(0, 4) = J * xi_r;
  return X;
}

Eigen::Matrix<double, 9, 1> SE23::logMap(const Eigen::MatrixXd &X) {
  Eigen::Matrix3d C;
  Eigen::Vector3d v;
  Eigen::Vector3d r;
  SE23::toComponents(X, C, v, r);

  Eigen::Vector3d phi = SO3::logMap(C);
  Eigen::Matrix3d J_left_inv = SO3::leftJacobianInverse(phi);

  Eigen::Matrix<double, 9, 1> xi;
  xi.block<3, 1>(0, 0) = phi;
  xi.block<3, 1>(3, 0) = J_left_inv * v;
  xi.block<3, 1>(6, 0) = J_left_inv * r;
  return xi;
}

Eigen::MatrixXd SE23::fromComponents(const Eigen::Matrix3d &C,
                                     const Eigen::Vector3d &v,
                                     const Eigen::Vector3d &r) {
  // Form an element of SE23 from individual components
  Eigen::Matrix<double, 5, 5> element_SE23;
  element_SE23.setIdentity();
  element_SE23.block<3, 3>(0, 0) = C;
  element_SE23.block<3, 1>(0, 3) = v;
  element_SE23.block<3, 1>(0, 4) = r;

  return element_SE23;
}

Eigen::Matrix<double, 5, 5>
SE23::fromCeresParameters(double const *parameters) {
  Eigen::Map<const Eigen::Matrix<double, 15, 1>> x_raw(parameters);
  Eigen::Matrix3d C = SO3::unflatten(x_raw.block<9, 1>(0, 0));
  Eigen::Vector3d v = x_raw.block<3, 1>(9, 0);
  Eigen::Vector3d r = x_raw.block<3, 1>(12, 0);
  Eigen::Matrix<double, 5, 5> element_SE23 = SE23::fromComponents(C, v, r);
  return element_SE23;
}

Eigen::Matrix<double, 15, 1>
SE23::toCeresParameters(Eigen::Matrix<double, 5, 5> X) {
  Eigen::Matrix<double, 15, 1> vec;
  Eigen::Matrix3d C;
  Eigen::Vector3d v;
  Eigen::Vector3d r;
  SE23::toComponents(X, C, v, r);

  vec.block<9, 1>(0, 0) = SO3::flatten(C);
  vec.block<3, 1>(9, 0) = v;
  vec.block<3, 1>(12, 0) = r;
  return vec;
}

void SE23::toComponents(const Eigen::Matrix<double, 5, 5> &X,
                        Eigen::Matrix3d &C, Eigen::Vector3d &v,
                        Eigen::Vector3d &r) {
  C = X.block<3, 3>(0, 0);
  v = X.block<3, 1>(0, 3);
  r = X.block<3, 1>(0, 4);
}

Eigen::Matrix<double, 5, 5>
SE23::inverse(const Eigen::Matrix<double, 5, 5> &X) {
  Eigen::Matrix<double, 5, 5> Xinv = Eigen::Matrix<double, 5, 5>::Identity();
  Eigen::Matrix3d C;
  Eigen::Vector3d v;
  Eigen::Vector3d r;
  SE23::toComponents(X, C, v, r);
  Eigen::Matrix3d Cinv = C.transpose();
  Xinv.block<3, 3>(0, 0) = Cinv;
  Xinv.block<3, 1>(0, 3) = -Cinv * v;
  Xinv.block<3, 1>(0, 4) = -Cinv * r;
  return Xinv;
}
Eigen::Matrix<double, 9, 9> SE23::leftJacobian(const Eigen::VectorXd &x) {

  Eigen::Vector3d phi = x.segment<3>(0);
  if (phi.norm() < SE23::small_angle_tol) {
    return Eigen::Matrix<double, 9, 9>::Identity();
  } else {
    Eigen::Vector3d xi_v = x.segment<3>(3);
    Eigen::Vector3d xi_r = x.segment<3>(6);
    Eigen::Matrix3d Jso3 = SO3::leftJacobian(phi);
    Eigen::Matrix3d Q_v{leftJacobianQMatrix(phi, xi_v)};
    Eigen::Matrix3d Q_r{leftJacobianQMatrix(phi, xi_r)};
    // Create left Jacobian
    Eigen::Matrix<double, 9, 9> J = Eigen::Matrix<double, 9, 9>::Identity();
    J.block<3, 3>(0, 0) = Jso3;
    J.block<3, 3>(3, 3) = Jso3;
    J.block<3, 3>(6, 6) = Jso3;
    J.block<3, 3>(3, 0) = Q_v;
    J.block<3, 3>(6, 0) = Q_r;
    return J;
  }
}

Eigen::Matrix<double, 9, 9>
SE23::leftJacobianInverse(const Eigen::VectorXd &x) {
  // Check if rotation component is small
  Eigen::Vector3d phi{x.block<3, 1>(0, 0)};
  if (x.block<3, 1>(0, 0).norm() < SE23::small_angle_tol) {
    return Eigen::Matrix<double, 9, 9>::Identity();
  } else {
    Eigen::Vector3d xi_v{x.block<3, 1>(3, 0)};
    Eigen::Vector3d xi_r{x.block<3, 1>(6, 0)};
    Eigen::Matrix3d Jinv{SO3::leftJacobianInverse(x.block<3, 1>(0, 0))};
    Eigen::Matrix3d Q_v{leftJacobianQMatrix(phi, xi_v)};
    Eigen::Matrix3d Q_r{leftJacobianQMatrix(phi, xi_r)};

    Eigen::Matrix<double, 9, 9> J = Eigen::Matrix<double, 9, 9>::Zero();
    J.block<3, 3>(0, 0) = Jinv;
    J.block<3, 3>(3, 3) = Jinv;
    J.block<3, 3>(6, 6) = Jinv;
    J.block<3, 3>(3, 0) = -Jinv * Q_v * Jinv;
    J.block<3, 3>(6, 0) = -Jinv * Q_r * Jinv;
    return J;
  }
}

Eigen::Matrix<double, 9, 9>
SE23::rightJacobianInverse(const Eigen::VectorXd &x) {
  return leftJacobianInverse(-x);
}

Eigen::Matrix<double, 9, 9> SE23::rightJacobian(const Eigen::VectorXd &x) {
  return SE23::leftJacobian(-x);
}

Eigen::Matrix<double, 9, 9> SE23::adjoint(const Eigen::MatrixXd &X) {
  Eigen::Matrix<double, 9, 9> Xadj{Eigen::Matrix<double, 9, 9>::Zero()};
  Eigen::Matrix3d C;
  Eigen::Vector3d v;
  Eigen::Vector3d r;
  SE23::toComponents(X, C, v, r);
  Xadj.block<3, 3>(0, 0) = C;
  Xadj.block<3, 3>(3, 3) = C;
  Xadj.block<3, 3>(6, 6) = C;
  Xadj.block<3, 3>(3, 0) = SO3::cross(v) * C;
  Xadj.block<3, 3>(6, 0) = SO3::cross(r) * C;
  return Xadj;
}

Eigen::Matrix3d SE23::leftJacobianQMatrix(const Eigen::Vector3d &phi,
                                          const Eigen::Vector3d &xi_r) {
  Eigen::Matrix3d rx{SO3::cross(xi_r)};
  Eigen::Matrix3d px{SO3::cross(phi)};

  double ph{phi.norm()};

  double ph2{ph * ph};
  double ph3{ph2 * ph};
  double ph4{ph3 * ph};
  double ph5{ph4 * ph};

  double cph{cos(ph)};
  double sph{sin(ph)};

  double m1{0.5};
  double m2{(ph - sph) / ph3};
  double m3{(0.5 * ph2 + cph - 1.0) / ph4};
  double m4{(ph - 1.5 * sph + 0.5 * ph * cph) / ph5};

  Eigen::Matrix3d pxrx{px * rx};
  Eigen::Matrix3d rxpx{rx * px};
  Eigen::Matrix3d pxrxpx{pxrx * px};

  Eigen::Matrix3d t1{rx};
  Eigen::Matrix3d t2{pxrx + rxpx + pxrxpx};
  Eigen::Matrix3d t3{px * pxrx + rxpx * px - 3.0 * pxrxpx};
  Eigen::Matrix3d t4{pxrxpx * px + px * pxrxpx};

  return m1 * t1 + m2 * t2 + m3 * t3 + m4 * t4;
};

Eigen::Matrix<double, 9, 1> SE23::minus(const Eigen::Matrix<double, 5, 5> &Y,
                                        const Eigen::Matrix<double, 5, 5> &X,
                                        LieDirection direction) {
  if (direction == LieDirection::left) {
    return SE23::logMap(Y * SE23::inverse(X));
  } else if (direction == LieDirection::right) {
    return SE23::logMap(SE23::inverse(X) * Y);
  } else {
    std::cerr << "Invalid Lie direction" << std::endl;
    return Eigen::Matrix<double, 9, 1>::Zero();
  }
}