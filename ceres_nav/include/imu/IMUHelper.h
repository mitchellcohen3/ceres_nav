#pragma once

/*
 * Some helper functions for IMU preintegration.
*/

#include <Eigen/Dense>
#include <vector>

class IMUIncrement;
class IMU;

Eigen::Matrix<double, 5, 5> createGMatrix(const Eigen::Vector3d &gravity,
                                          double dt);
                                          
Eigen::Matrix3d createNMatrix(const Eigen::Vector3d &phi_vec);
Eigen::Matrix<double, 5, 5> createUMatrix(const Eigen::Vector3d &omega,
                                          const Eigen::Vector3d &accel,
                                          double dt);

Eigen::Matrix<double, 5, 5> inverseIE3(const Eigen::Matrix<double, 5, 5> &X);
Eigen::Matrix<double, 5, 5> createUMatrixInv(const Eigen::Vector3d &omega,
                                             const Eigen::Vector3d &accel,
                                             double dt);

Eigen::Matrix<double, 9, 6> createLMatrix(const Eigen::Vector3d &unbiased_gyro,
                                          const Eigen::Vector3d &unbiased_accel,
                                          double dt);
Eigen::Matrix<double, 9, 9> adjointIE3(const Eigen::Matrix<double, 5, 5> &X);

bool preintegrateIMUMeasurements(IMUIncrement &rmi,
                                 const std::vector<IMU> &imu_meas_vec);
std::vector<IMU> getIMUBetweenTimes(const double &stamp_i,
                                    const double &stamp_j,
                                    const std::vector<IMU> &imu_meas_vec);
bool preintegrateBetweenTimes(IMUIncrement &rmi, const double &stamp_i,
                              const double &stamp_j,
                              const std::vector<IMU> &imu_meas_vec);