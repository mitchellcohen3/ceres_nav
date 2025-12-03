#include "lie/SE3.h"
#include "lie/SE23.h"
#include "lie/SO3.h"
#include <Eigen/Dense>

#include <glog/logging.h>

namespace ceres_nav {

Eigen::Matrix4d SE3::wedge(const Eigen::Matrix<double, 6, 1> &xi) {
  Eigen::Matrix4d X;
  // clang-format off
  X << SO3::cross(xi.block<3, 1>(0, 0)), xi.block<3, 1>(3, 0), 
    0, 0, 0, 0;
  // clang-format on
  return X;
}

Eigen::Matrix<double, 4, 4> SE3::expMap(const Eigen::Matrix<double, 6, 1> &x) {
  Eigen::Matrix<double, 4, 4> X = Eigen::Matrix<double, 4, 4>::Identity();
  Eigen::Matrix3d R{SO3::expMap(x.block<3, 1>(0, 0))};
  Eigen::Vector3d xi_r{x.block<3, 1>(3, 0)};
  Eigen::Matrix3d J{SO3::leftJacobian(x.block<3, 1>(0, 0))};
  X.block<3, 3>(0, 0) = R;
  X.block<3, 1>(0, 3) = J * xi_r;
  return X;
}

Eigen::Matrix<double, 6, 1> SE3::logMap(const Eigen::MatrixXd &X) {
  Eigen::Matrix3d C;
  Eigen::Vector3d r;
  SE3::toComponents(X, C, r);

  Eigen::Vector3d phi = SO3::logMap(C);
  Eigen::Matrix3d J_left_inv = SO3::leftJacobianInverse(phi);

  Eigen::Matrix<double, 6, 1> xi;
  xi.block<3, 1>(0, 0) = phi;
  xi.block<3, 1>(3, 0) = J_left_inv * r;
  return xi;
}

Eigen::MatrixXd SE3::fromComponents(const Eigen::Matrix3d &C,
                                    const Eigen::Vector3d &r) {

  // Form an element of SE3 from individual components
  Eigen::Matrix<double, 4, 4> element_SE3;
  element_SE3.setIdentity();
  element_SE3.block<3, 3>(0, 0) = C;
  element_SE3.block<3, 1>(0, 3) = r;

  return element_SE3;
}

Eigen::Matrix<double, 4, 4> SE3::fromCeresParameters(double const *parameters) {
  Eigen::Map<const Eigen::Matrix<double, 12, 1>> x_raw(parameters);
  Eigen::Matrix3d C = SO3::unflatten(x_raw.head(9));
  Eigen::Vector3d r = x_raw.tail(3);
  return SE3::fromComponents(C, r);
}

void SE3::toComponents(const Eigen::Matrix<double, 4, 4> &X, Eigen::Matrix3d &C,
                       Eigen::Vector3d &r) {
  C = X.block<3, 3>(0, 0);
  r = X.block<3, 1>(0, 3);
}

Eigen::Matrix<double, 4, 4> SE3::inverse(const Eigen::Matrix<double, 4, 4> &X) {
  Eigen::Matrix4d Xinv = Eigen::Matrix<double, 4, 4>::Identity();
  Eigen::Matrix3d C;
  Eigen::Vector3d r;
  SE3::toComponents(X, C, r);
  Eigen::Matrix3d Cinv = C.transpose();
  Xinv.block<3, 3>(0, 0) = Cinv;
  Xinv.block<3, 1>(0, 3) = -Cinv * r;
  return Xinv;
}
Eigen::Matrix<double, 6, 6> SE3::leftJacobian(const Eigen::VectorXd &x) {

  Eigen::Vector3d phi = x.segment<3>(0);
  if (phi.norm() < SE23::small_angle_tol) {
    return Eigen::Matrix<double, 6, 6>::Identity();
  } else {
    Eigen::Vector3d xi_r = x.segment<3>(3);
    Eigen::Matrix3d Jso3 = SO3::leftJacobian(phi);
    Eigen::Matrix3d Q_r{leftJacobianQMatrix(phi, xi_r)};
    // Create left Jacobian
    Eigen::Matrix<double, 6, 6> J = Eigen::Matrix<double, 6, 6>::Identity();
    J.block<3, 3>(0, 0) = Jso3;
    J.block<3, 3>(3, 0) = Q_r;
    J.block<3, 3>(3, 3) = Jso3;
    return J;
  }
}

Eigen::Matrix<double, 6, 6> SE3::leftJacobianInverse(const Eigen::VectorXd &x)
{
    // Check if rotation component is small
    if (x.block<3, 1>(0, 0).norm() < SE3::small_angle_tol) {
      return Eigen::Matrix<double, 6, 6>::Identity();
    } else {
      Eigen::Matrix<double, 6, 6> J = Eigen::Matrix<double, 6, 6>::Zero();
      Eigen::Matrix3d Jinv{SO3::leftJacobianInverse(x.block<3, 1>(0, 0))};
      Eigen::Matrix3d Q{
          SE3::leftJacobianQMatrix(x.block<3, 1>(0, 0), x.block<3, 1>(3, 0))};
      J.block<3, 3>(0, 0) = Jinv;
      J.block<3, 3>(3, 0) = -Jinv * Q * Jinv;
      J.block<3, 3>(3, 3) = Jinv;
      return J;
    }
}

Eigen::Matrix<double, 6, 1> SE3::minus(const Eigen::Matrix<double, 4, 4> &Y,
                                       const Eigen::Matrix<double, 4, 4> &X) {
  // Return Y \ominus X, left perturbation
  return SE3::logMap(Y * SE3::inverse(X));
}

Eigen::Matrix<double, 6, 6> SE3::rightJacobian(const Eigen::VectorXd &x) {
  return SE3::leftJacobian(-x);
}

Eigen::Matrix<double, 6, 6> SE3::rightJacobianInverse(const Eigen::VectorXd &x)
{
  return SE3::leftJacobianInverse(-x);
}



Eigen::Matrix<double, 6, 6> SE3::adjoint(const Eigen::MatrixXd &X) {
  Eigen::Matrix<double, 6, 6> Xadj{Eigen::Matrix<double, 6, 6>::Zero()};
  Eigen::Matrix3d C;
  Eigen::Vector3d r;
  SE3::toComponents(X, C, r);
  Xadj.block<3, 3>(0, 0) = C;
  Xadj.block<3, 3>(3, 3) = C;
  Xadj.block<3, 3>(3, 0) = SO3::cross(r) * C;
  return Xadj;
}

Eigen::Matrix3d SE3::leftJacobianQMatrix(const Eigen::Vector3d &phi,
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

Eigen::Matrix<double, 6, 1> SE3::minus(const Eigen::Matrix<double, 4, 4> &Y,
                                        const Eigen::Matrix<double, 4, 4> &X,
                                        LieDirection direction) {
  if (direction == LieDirection::left) {
    return SE3::logMap(Y * SE3::inverse(X));
  } else if (direction == LieDirection::right) {
    LOG(INFO) << "Calling right minus";
    return SE3::logMap(SE3::inverse(X) * Y);
  } else {
    LOG(ERROR) << "Direction not supported!";
    return Eigen::Matrix<double, 6, 1>::Zero();
  }
}

} // namespace ceres_nav