#pragma once

#include "lib/ExtendedPoseParameterBlock.h"
#include "lie/LieDirection.h"
#include <Eigen/Dense>
#include <vector>

namespace ceres_nav {

// Two options for mean discretization of the RMI are available -
//  1. ConstantAccel: here, we assume that the
// global acceleration is constant over the integration interval.
//  2. ConstantMeas: here, we assume that the IMU measurements themselves
//  are constant over the integration interval.
enum class IMUDiscretizationMethod { ConstantAccel, ConstantMeas };

class IMUIncrementOptions {
public:
  /**
   * @brief Constructor for IMUIncrementOptions
   */
  IMUIncrementOptions(double sigma_gyro_ct = 0.01, double sigma_accel_ct = 0.01,
                      double sigma_gyro_bias_ct = 0.001,
                      double sigma_accel_bias_ct = 0.001,
                      const Eigen::Vector3d &gravity = Eigen::Vector3d(0, 0,
                                                                       -9.81),
                      const LieDirection &direction = LieDirection::left,
                      const ExtendedPoseRepresentation &pose_rep =
                          ExtendedPoseRepresentation::SE23,
                      const IMUDiscretizationMethod &discretization =
                          IMUDiscretizationMethod::ConstantMeas)
      : direction(direction), pose_rep(pose_rep), sigma_gyro_ct(sigma_gyro_ct),
        sigma_accel_ct(sigma_accel_ct), sigma_gyro_bias_ct(sigma_gyro_bias_ct),
        sigma_accel_bias_ct(sigma_accel_bias_ct), gravity(gravity),
        discretization(discretization){};

  // State representation options
  LieDirection direction;
  ExtendedPoseRepresentation pose_rep;

  // Noise covariances
  double sigma_gyro_ct;
  double sigma_accel_ct;
  double sigma_gyro_bias_ct;
  double sigma_accel_bias_ct;

  Eigen::Vector3d gravity;

  IMUDiscretizationMethod discretization;

  /**
   * @brief Returns the continuous-time noise covariance matrix Q_ct
   * based on the noise parameters.
   */
  Eigen::Matrix<double, 12, 12> continuousTimeNoiseCovariance() const {
    Eigen::Matrix<double, 12, 12> Q_ct =
        Eigen::Matrix<double, 12, 12>::Identity();
    Q_ct.block<3, 3>(0, 0) *= sigma_gyro_ct * sigma_gyro_ct;
    Q_ct.block<3, 3>(3, 3) *= sigma_accel_ct * sigma_accel_ct;
    Q_ct.block<3, 3>(6, 6) *= sigma_gyro_bias_ct * sigma_gyro_bias_ct;
    Q_ct.block<3, 3>(9, 9) *= sigma_accel_bias_ct * sigma_accel_bias_ct;
    return Q_ct;
  }
};

class IMUIncrement {
public:
  /**
   * @brief Constructor for IMUIncrement
   */
  IMUIncrement(const std::shared_ptr<IMUIncrementOptions> &options,
               const Eigen::Vector3d &init_gyro_bias,
               const Eigen::Vector3d &init_accel_bias);

  /**
   * @brief Propagates the RMI forward in time given te new IMU measurements.
   * Updates the mean RMI, covariance, and Jacobians.
   */
  void propagate(double dt, const Eigen::Vector3d &gyro,
                 const Eigen::Vector3d &acc);

  void reset(double new_start_stamp,
             const Eigen::Vector3d &new_gyro_bias = Eigen::Vector3d::Zero(),
             const Eigen::Vector3d &new_accel_bias = Eigen::Vector3d::Zero());

  Eigen::Matrix<double, 5, 5> getDeltaX();

  // Repropagate all relevant quantities from a new initial bias
  void repropagate(const Eigen::Vector3d &init_gyro_bias,
                   const Eigen::Vector3d &init_accel_bias);

  /**
   * @brief Gets the propagated covariance matrix on the RMI
   */
  Eigen::Matrix<double, 15, 15> covariance() const { return covariance_; }

  /**
   * @brief Gets the mean RMI, an element of SE_2(3)
   */
  Eigen::Matrix<double, 5, 5> meanRMI() const { return Upsilon_ij_; }

  // Get subcomponents of the mean RMI
  Eigen::Matrix3d delta_Cij() const { return Upsilon_ij_.block<3, 3>(0, 0); }
  Eigen::Vector3d delta_vij() const { return Upsilon_ij_.block<3, 1>(3, 0); }
  Eigen::Vector3d delta_rij() const { return Upsilon_ij_.block<3, 1>(4, 0); }

  /**
   * @brief Returns the full Jacobian
   */
  Eigen::Matrix<double, 15, 15> jacobian() const { return jacobian_; }

  /**
   * @brief returns the bias Jacobian
   */
  Eigen::Matrix<double, 9, 6> biasJacobian() const {
    return jacobian_.block<9, 6>(0, 9);
  }

  // Get the RMI propagation time
  double deltaT() const { return end_stamp - start_stamp; }

  double startTime() const { return start_stamp; }
  double endTime() const { return end_stamp; }

  std::shared_ptr<IMUIncrementOptions> options() const { return options_; }

  Eigen::Vector3d gyroBias() const { return gyro_bias; }
  Eigen::Vector3d accelBias() const { return accel_bias; }

protected:
  /**
   * @brief Propagates forward the covariance and bias Jacobian.
   */
  void propagateCovarianceAndBiasJacobian(double dt,
                                          const Eigen::Vector3d &omega,
                                          const Eigen::Vector3d &accel);

  // Helper functions to compute the continuous-time A and L matrices
  void computeContinuousTimeJacobiansSE23(const Eigen::Matrix3d &C,
                                          const Eigen::Vector3d &v,
                                          const Eigen::Vector3d &r,
                                          const Eigen::Vector3d &omega,
                                          const Eigen::Vector3d &accel,
                                          Eigen::Matrix<double, 15, 15> &A_ct,
                                          Eigen::Matrix<double, 15, 12> &L_ct);

  void computeContinuousTimeJacobiansDecoupled(
      const Eigen::Matrix3d &C, const Eigen::Vector3d &v,
      const Eigen::Vector3d &r, const Eigen::Vector3d &omega,
      const Eigen::Vector3d &accel, Eigen::Matrix<double, 15, 15> &A_ct,
      Eigen::Matrix<double, 15, 12> &L_ct);

  /**
   * @brief The options for our IMUIncrement
   */
  std::shared_ptr<IMUIncrementOptions> options_;

  // The mean value of the RMI, an element of SE_2(3)
  Eigen::Matrix<double, 5, 5> Upsilon_ij_;

  // Covariance and Jacobian that we'll propagate forward at each iteration
  Eigen::Matrix<double, 15, 15> covariance_;
  Eigen::Matrix<double, 15, 15> jacobian_;

  // Initial estimates gyro and accel bias
  Eigen::Vector3d gyro_bias;
  Eigen::Vector3d accel_bias;

  // Timestamp information
  double start_stamp;
  double end_stamp;

  std::vector<double> dt_buf;
  std::vector<Eigen::Vector3d> acc_buf;
  std::vector<Eigen::Vector3d> gyr_buf;
};

/// Helper functions for preintegration
/**
 * @brief Creates the Phi matrix for a given dt and extended pose T
 */
Eigen::Matrix<double, 5, 5> phiMat(double dt,
                                   const Eigen::Matrix<double, 5, 5> &T);

/// Creates the Upsilon matrix
Eigen::Matrix<double, 5, 5> upsilonMat(double dt, const Eigen::Vector3d &omega,
                                       const Eigen::Vector3d &accel,
                                       IMUDiscretizationMethod discretization);

/// Creates the Psi matrix
Eigen::Matrix3d psiMat(const Eigen::Vector3d &omega);

} // namespace ceres_nav
