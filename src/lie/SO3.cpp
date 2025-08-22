#include "lie/SO3.h"
// #include "utils/utility.h"
#include <iostream>

Eigen::Matrix3d SO3::cross(const Eigen::Vector3d &x) {
  Eigen::Matrix3d X;
  // clang-format off
    X <<     0, -x(2),  x(1),
          x(2),     0, -x(0), 
         -x(1),  x(0),     0;
  // clang-format on
  return X;
};

Eigen::Vector3d SO3::vee(const Eigen::Matrix3d &element_so3) {
  Eigen::Vector3d phi;
  phi.x() = element_so3(2, 1);
  phi.y() = element_so3(0, 2);
  phi.z() = element_so3(1, 0);
  return phi;
}

Eigen::Matrix<double, 9, 1> SO3::flatten(const Eigen::Matrix3d C) {
  return Eigen::Map<const Eigen::Matrix<double, 9, 1>>(C.data());
}
Eigen::Matrix3d SO3::unflatten(const Eigen::Matrix<double, 9, 1> vec_C) {
  return Eigen::Map<const Eigen::Matrix3d>(vec_C.data());
}

Eigen::Matrix3d SO3::expMap(const Eigen::Vector3d &phi) {
  Eigen::Matrix3d phi_cross = SO3::cross(phi);
  double angle = sqrt(phi.transpose() * phi);

  double A, B;
  double small_angle_tol = 1e-7;

  // If angle is small, use Taylor series expansion
  if (angle <= small_angle_tol) {
    double t2 = angle * angle;

    A = 1.0 -
        t2 / 6.0 * (1.0 - t2 / 20.0 * (1.0 - t2 / 42.0)); // eq. (155) from Eade
    B = 1.0 / 2.0 *
        (1.0 -
         t2 / 12.0 *
             (1.0 - t2 / 30.0 * (1.0 - t2 / 56.0))); // eq. (157) from Eade
  } else {
    // Compute the coefficients from equation (103) from "Lie Groups for
    // Computer Vision"
    A = sin(angle) / angle;
    B = (1.0 - cos(angle)) / (angle * angle);
  }

  return Eigen::Matrix3d::Identity() + A * phi_cross +
         B * (phi_cross * phi_cross);
}

Eigen::Vector3d SO3::logMap(const Eigen::Matrix3d &x) {
  Eigen::Vector3d xi;
  double cos_theta = 0.5 * x.trace() - 0.5;

  double small_angle_tol = 1e-7;
  // Clip cos(angle) to its proper domain to avoid nans from rounding errors
  if (cos_theta >= 1.0) {
    cos_theta = 0.99999999999999;
  } else if (cos_theta <= -1.0) {
    cos_theta = -0.99999999999999;
  }
  double theta = acos(cos_theta);
  if (theta < small_angle_tol) {
    // xi = x.block<3, 1>(0, 2);
    xi = SO3::vee(x - Eigen::Matrix3d::Identity());
  } else if (abs(M_PI - theta) < small_angle_tol) {
    std::cout << "Warning: angle is close to pi in SO3::logMap" << std::endl;
    xi = theta * SO3::vee(x - x.transpose()) /
         SO3::vee(x - x.transpose()).norm();
  } else {
    xi = SO3::vee((theta / (2 * sin(theta))) * (x - x.transpose()));
  }
  return xi;
}

Eigen::Matrix3d SO3::leftJacobian(const Eigen::Vector3d &phi) {
  // Computes the left Jacobian for SO(3)
  double angle = phi.norm();
  double A;
  double B;

  double small_angle_tol = 1e-7;

  if (angle <= small_angle_tol) {
    // ROS_WARN("Small angle in compute J left");

    // std::cout << "Small angle in compute J left" << std::endl;
    double t2 = angle * angle;
    A = 1.0 / 2.0 *
        (1.0 -
         t2 / 12.0 * (1.0 - t2 / 30.0 * (1.0 - t2 / 56.0))); // eq (157) of Eade
    B = 1.0 / 6.0 *
        (1.0 -
         t2 / 20.0 * (1.0 - t2 / 42.0 * (1.0 - t2 / 72.0))); // eq (159) of Eade
  } else {
    // Compute coefficients from eq (124) of Eade.
    A = (1.0 - cos(angle)) / (angle * angle);
    B = (angle - sin(angle)) / (angle * angle * angle);
  }

  Eigen::Matrix3d cross_phi = SO3::cross(phi);

  return Eigen::Matrix3d::Identity() + A * cross_phi +
         B * cross_phi * cross_phi;
}

/**
 * leftJacobianInv computes the inverse left Jacobian for SO(3).
 * The implementation corresponds to Section 9.3, equation (125) of "Lie Groups
 * for Computer Vision", by Ethan Eade. When the angle is small, use Taylor
 * Series expansion given in Section 11, eq 163 of Eade.
 *
 * Note that a different form of the inverse Left Jacobian is sometimes found in
 * literature, for example, equation (146) from "A Micro Lie Theory For State
 * Estimation in Robotics" by Joan Sola. The form found from equation (146) of
 * the paper by Joan Sola is numerically equivalent to the form implemented
 * here.
 */
Eigen::Matrix3d SO3::leftJacobianInverse(const Eigen::Vector3d &phi) {
  double angle = phi.norm();
  double A;

  double small_angle_tol = 1e-7;

  if (angle <= small_angle_tol) {
    // ROS_WARN("Small angle in compute J left inverse");

    double t2 = angle * angle;
    A = 1.0 / 12.0 *
        (1.0 +
         t2 / 60.0 * (1.0 + t2 / 42.0 * (1.0 + t2 / 40.0))); // eq (163) in Eade
  } else {
    A = 1.0 / (angle * angle) *
        (1.0 - (angle * sin(angle) / (2.0 * (1.0 - cos(angle)))));
  }

  Eigen::Matrix3d cross_phi = SO3::cross(phi);

  return Eigen::Matrix3d::Identity() - 0.5 * cross_phi +
         A * cross_phi * cross_phi;
}

/**
 * computeJRight computes the right Jacobian of SO(3).
 * Here, the relationship between the left and the right Jacobian from
 * section 7.1.5 of "State Estimation for Robotics" by T. Barfoot is used,
 * which says that J^r (phi) = J^left (-phi).
 */
Eigen::Matrix3d SO3::rightJacobian(const Eigen::Vector3d &phi) {
  // Equation (7.87) of "State Estimation for Robotics".
  return SO3::leftJacobian(-phi);
}

/**
 * computeJRightInv computes the inverse right Jacobian of SO(3).
 * The relationship between the left and right inverse JAcobians is once again
 * used.
 */
Eigen::Matrix3d SO3::rightJacobianInverse(const Eigen::Vector3d &phi) {
  return SO3::leftJacobianInverse(-phi);
}

Eigen::Vector3d SO3::toEuler(const Eigen::Matrix3d &C) {
  // double pitch = atan2(-C(2, 0), sqrt(C(0, 0) * C(0, 0) + C(1, 0) * C(1,
  // 0))); double roll; double yaw;

  // if (isDoubleWithinTolerance(pitch, 0.0, 1e-5)) {
  //   yaw = 0.0;
  //   roll = atan2(C(0, 1), C(1, 1));
  // } else if (isDoubleWithinTolerance(pitch, -M_PI / 2.0, 1e-5)) {
  //   yaw = 0.0;
  //   roll = -atan2(C(0, 1), C(1, 1));
  // } else {
  //   double sec_pitch = 1 / cos(pitch);
  //   yaw = atan2(C(1, 0) * sec_pitch, C(0, 0) * sec_pitch);
  //   roll = atan2(C(2, 1) * sec_pitch, C(2, 2) * sec_pitch);
  // }

  // Eigen::Vector3d euler;
  // euler << roll, pitch, yaw;
  // return euler;
  std::cout << "Not implemented" << std::endl;
}

Eigen::Vector3d SO3::minus(const Eigen::Matrix3d &Y, const Eigen::Matrix3d &X,
                           const LieDirection &direction) {
  Eigen::Matrix3d delta_C;
  if (direction == LieDirection::left) {
    delta_C = Y * X.transpose();
  } else if (direction == LieDirection::right) {
    delta_C = X.transpose() * Y;
  } else {
    std::cerr << "Invalid Lie direction" << std::endl;
  }

  return SO3::logMap(delta_C);
}

/*
template <typename Derived>
Eigen::Matrix<typename Derived::Scalar, 3, 3> SO3::cross(const
Eigen::MatrixBase<Derived> &q)
{
    Eigen::Matrix<typename Derived::Scalar, 3, 3> ans;
    ans << typename Derived::Scalar(0), -q(2), q(1),
        q(2), typename Derived::Scalar(0), -q(0),
        -q(1), q(0), typename Derived::Scalar(0);
    return ans;
}
*/
