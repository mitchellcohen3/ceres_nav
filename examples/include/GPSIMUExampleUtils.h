#pragma once

#include <Eigen/Dense>
#include <string>
#include <vector>

#include <fstream>
#include <sstream>

#include "lie/SE23.h"
#include "imu/IMUHelper.h"

#include <glog/logging.h>

/**
 * @brief A holder for IMU measurements.
 */
class IMUMessage {
public:
  Eigen::Vector3d gyro;
  Eigen::Vector3d accel;
  double timestamp;

  IMUMessage(const Eigen::Vector3d &g, const Eigen::Vector3d &a, double t)
      : gyro(g), accel(a), timestamp(t) {}
};

/**
 * @brief A holder for GPS measurements.
 */
class GPSMessage {
public:
  Eigen::Vector3d measurement;
  double timestamp;

  GPSMessage(const Eigen::Vector3d &m, double t)
      : measurement(m), timestamp(t) {}
};

/**
 * @brief A container for IMU states, with convient accessors for attitude,
 * velocity, position, gyro bias, and accel bias.
 */
class IMUState {
public:
  IMUState(const Eigen::Matrix<double, 5, 5> &nav_state,
           const Eigen::Vector3d &gyro_bias, const Eigen::Vector3d &accel_bias,
           double stamp)
      : nav_state_(nav_state), gyro_bias_(gyro_bias), accel_bias_(accel_bias),
        timestamp_(stamp) {}

  Eigen::Matrix<double, 5, 5> navState() const { return nav_state_; }
  Eigen::Matrix3d attitude() const { return nav_state_.block<3, 3>(0, 0); }
  Eigen::Vector3d velocity() const { return nav_state_.block<3, 1>(0, 3); }
  Eigen::Vector3d position() const { return nav_state_.block<3, 1>(0, 4); }

  Eigen::Vector3d gyroBias() const { return gyro_bias_; }
  Eigen::Vector3d accelBias() const { return accel_bias_; }

  Eigen::Matrix<double, 6, 1> bias() const {
    Eigen::Matrix<double, 6, 1> bias;
    bias.block<3, 1>(0, 0) = gyro_bias_;
    bias.block<3, 1>(3, 0) = accel_bias_;
    return bias;
  }

  double timestamp() const { return timestamp_; }

  void setNavState(const Eigen::Matrix<double, 5, 5> &nav_state) {
    nav_state_ = nav_state;
  }

  void setStamp(double stamp) {
    timestamp_ = stamp;
  }

  Eigen::Matrix<double, 17, 1> toVector() const {
    Eigen::Matrix<double, 17, 1> vec;
    vec(0) = timestamp_;
    vec.block<3, 1>(1, 0) = position();
    Eigen::Quaterniond q(attitude());
    vec(4, 0) = q.w();
    vec(5, 0) = q.x();
    vec(6, 0) = q.y();
    vec(7, 0) = q.z();
    vec.block<3, 1>(8, 0) = velocity();
    vec.block<3, 1>(11, 0) = gyro_bias_;
    vec.block<3, 1>(14, 0) = accel_bias_;
    return vec;
  }

protected:
  double timestamp_;
  Eigen::Matrix<double, 5, 5> nav_state_;
  Eigen::Vector3d gyro_bias_;
  Eigen::Vector3d accel_bias_;
};

/**
 * @brief Loads IMU data from a file.
 *
 * The IMU data file is assumed to be a CSV where each row contains:
 *   timestamp, gyro_x, gyro_y, gyro_z, accel_x, accel_y, accel_z
 */
std::vector<IMUMessage> loadIMUData(const std::string &fname) {
  std::vector<IMUMessage> imu_data;
  std::ifstream file(fname);

  if (!file.is_open()) {
    LOG(ERROR) << "Failed to open IMU data file: " << fname;
    return imu_data;
  }

  std::string line;
  while (std::getline(file, line)) {
    std::istringstream ss(line);
    std::string value;
    std::vector<double> values;

    // Split the line by the comma
    while (std::getline(ss, value, ',')) {
      values.push_back(std::stod(value));
    }

    // Expecting 7 values
    if (values.size() != 7) {
      LOG(ERROR) << "Invalid IMU data format in line: " << line;
      continue;
    }

    // Create an IMU Message
    double stamp = values[0];
    Eigen::Vector3d gyro{values[1], values[2], values[3]};
    Eigen::Vector3d accel{values[4], values[5], values[6]};
    imu_data.push_back(IMUMessage(gyro, accel, stamp));
  }

  return imu_data;
}

/**
 * @brief Loads GPS data from a file.
 *
 * The GPS data file is assumed to be a CSV where each row contains:
 *   timestamp, pox_x, pos_y, pos_z
 */
std::vector<GPSMessage> loadGPSData(const std::string &fname) {
  std::vector<GPSMessage> data;
  std::ifstream file(fname);

  if (!file.is_open()) {
    LOG(ERROR) << "Failed to open IMU data file: " << fname;
    return data;
  }

  std::string line;
  while (std::getline(file, line)) {
    std::istringstream ss(line);
    std::string value;
    std::vector<double> values;

    // Split the line by the comma
    while (std::getline(ss, value, ',')) {
      values.push_back(std::stod(value));
    }

    // Expecting 7 values
    if (values.size() != 4) {
      LOG(ERROR) << "Invalid GPS data format in line: " << line;
      continue;
    }

    // Create a GPS Message
    double stamp = values[0];
    Eigen::Vector3d meas{values[1], values[2], values[3]};
    data.push_back(GPSMessage(meas, stamp));
  }

  return data;
}

/**
 * @brief Loads IMU states from an ASL format file.
 *
 * The file is a CSV where each row contains:
 *
 */
std::vector<IMUState> loadIMUStates(const std::string &fname) {
  std::vector<IMUState> imu_states;
  std::ifstream file(fname);
  if (!file.is_open()) {
    LOG(ERROR) << "Failed to open the IMU states file: " << fname;
    return imu_states;
  }

  std::string line;
  while (std::getline(file, line)) {
    std::istringstream ss(line);
    std::string value;
    std::vector<double> values;

    while (std::getline(ss, value, ',')) {
      values.push_back(std::stod(value));
    }

    if (values.size() != 17) {
      // LOG(ERROR) << "Invalid IMU state format in line: " << line;
      LOG(ERROR) << "Expected 17 values, got " << values.size();
      continue;
    }

    // Convert to IMUState type
    double stamp = values[0];
    Eigen::Vector3d position{values[1], values[2], values[3]};
    Eigen::Quaterniond attitude{values[4], values[5], values[6], values[7]};
    Eigen::Matrix3d C_ab = attitude.toRotationMatrix();
    Eigen::Vector3d velocity{values[8], values[9], values[10]};
    Eigen::Vector3d gyro_bias{values[11], values[12], values[13]};
    Eigen::Vector3d accel_bias{values[14], values[15], values[16]};

    Eigen::Matrix<double, 5, 5> nav_state =
        ceres_nav::SE23::fromComponents(C_ab, velocity, position);
    imu_states.push_back(IMUState(nav_state, gyro_bias, accel_bias, stamp));
  }

  return imu_states;
}

/**
 * @brief propagates and IMU state forward using the IMU measurements.
*/
void propagateIMUState(IMUState &state, const IMUMessage &imu_msg, const Eigen::Vector3d &gravity, double dt) {
  Eigen::Vector3d unbiased_gyro = imu_msg.gyro - state.gyroBias();
  Eigen::Vector3d unbiased_accel = imu_msg.accel - state.accelBias();

  Eigen::Matrix<double, 5, 5> G = ceres_nav::createGMatrix(gravity, dt);
  Eigen::Matrix<double, 5, 5> U =
      ceres_nav::createUMatrix(unbiased_gyro, unbiased_accel, dt);

  Eigen::Matrix<double, 5, 5> prev_extended_pose = state.navState();
  Eigen::Matrix<double, 5, 5> next_extended_pose = G * prev_extended_pose * U;

  double new_stamp = state.timestamp() + dt;
  state.setStamp(new_stamp);
  // Update the value for the extended pose
  state.setNavState(next_extended_pose);
}
