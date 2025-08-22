#pragma once

#include <Eigen/Dense>

#include "factors/AbsolutePositionFactor.h"
#include "factors/IMUPreintegrationFactor.h"
#include "factors/IMUPriorFactor.h"

#include "imu/IMUIncrement.h"
#include "lib/ExtendedPoseParameterBlock.h"
#include "lib/FactorGraph.h"
#include "lib/ParameterBlock.h"

#include "GPSIMUExampleUtils.h"
#include "utils/Utils.h"

/**
 * @brief A collection of utilities to add specific states and factors to the
 * graph
 */

namespace factor_graph_utils {

struct ProblemKeys {
  std::string nav_state_key = "nav_state";
  std::string bias_state_key = "gyro_bias";
};

void addIMUState(
    ceres_nav::FactorGraph &graph, const IMUState &imu_state,
    const LieDirection direction,
    ExtendedPoseRepresentation state_rep = ExtendedPoseRepresentation::SE23,
    ProblemKeys keys = ProblemKeys()) {
  // Create a new ExtendedPoseParameterBlock for the IMU state
  std::shared_ptr<ExtendedPoseParameterBlock> nav_state_block =
      std::make_shared<ExtendedPoseParameterBlock>(imu_state.navState(),
                                                   state_rep, "extended_pose", direction);
  std::shared_ptr<ParameterBlock<6>> bias_block =
      std::make_shared<ParameterBlock<6>>(imu_state.bias());

  graph.addState(keys.nav_state_key, imu_state.timestamp(), nav_state_block);
  graph.addState(keys.bias_state_key, imu_state.timestamp(), bias_block);
};

void addPriorFactor(ceres_nav::FactorGraph &graph,
                    const IMUState prior_imu_state,
                    const Eigen::Matrix<double, 15, 15> &prior_covariance,
                    LieDirection direction,
                    ExtendedPoseRepresentation state_rep, ProblemKeys keys) {
  std::vector<ceres_nav::StateID> state_ids = {
      ceres_nav::StateID(keys.nav_state_key, prior_imu_state.timestamp()),
      ceres_nav::StateID(keys.bias_state_key, prior_imu_state.timestamp())};

  auto *factor =
      new IMUPriorFactor(prior_imu_state.navState(), prior_imu_state.bias(),
                         prior_covariance, direction);
  graph.addFactor(state_ids, factor, prior_imu_state.timestamp());
}

void addPreintegrationFactor(ceres_nav::FactorGraph &graph,
                             const IMUIncrement &imu_increment,
                             LieDirection &direction,
                             ExtendedPoseRepresentation state_rep,
                             ProblemKeys keys = ProblemKeys()) {
  double start_stamp = imu_increment.start_stamp;
  double end_stamp = imu_increment.end_stamp;
  std::vector<ceres_nav::StateID> state_ids = {
      ceres_nav::StateID(keys.nav_state_key, start_stamp),
      ceres_nav::StateID(keys.bias_state_key, start_stamp),
      ceres_nav::StateID(keys.nav_state_key, end_stamp),
      ceres_nav::StateID(keys.bias_state_key, end_stamp)};

  auto *factor = new IMUPreintegrationFactor(imu_increment, false, direction, state_rep);
  graph.addFactor(state_ids, factor, start_stamp);
}

/**
 * @brief Adds a GPS factor to the graph
 */
void addGPSFactor(
    ceres_nav::FactorGraph &graph, const GPSMessage &gps_message,
    const LieDirection &direction, const Eigen::Matrix3d &covariance,
    ExtendedPoseRepresentation state_rep = ExtendedPoseRepresentation::SE23,
    ProblemKeys keys = ProblemKeys()) {
  Eigen::Matrix3d sqrt_info = computeSquareRootInformation(covariance);

  std::vector<ceres_nav::StateID> state_ids = {
      ceres_nav::StateID(keys.nav_state_key, gps_message.timestamp)};
  auto factor =
      new AbsolutePositionFactor(gps_message.measurement, direction, sqrt_info, state_rep);
  graph.addFactor(state_ids, factor, gps_message.timestamp);
}

IMUState getIMUState(ceres_nav::FactorGraph &graph, double timestamp,
                     ProblemKeys keys = ProblemKeys()) {
  std::shared_ptr<ExtendedPoseParameterBlock> nav_state =
      graph.getStates().getState<ExtendedPoseParameterBlock>(keys.nav_state_key,
                                                             timestamp);
  std::shared_ptr<ParameterBlock<6>> bias =
      graph.getStates().getState<ParameterBlock<6>>(keys.bias_state_key,
                                                    timestamp);
  if (!nav_state || !bias) {
    throw std::runtime_error("IMU state not found for timestamp: " +
                             std::to_string(timestamp));
  }
  IMUState imu_state(nav_state->extendedPose(), bias->getEstimate().head<3>(),
                     bias->getEstimate().tail<3>(), timestamp);
  return imu_state;
}

Eigen::Matrix<double, 15, 15>
computeIMUCovariance(ceres_nav::FactorGraph &graph, double timestamp,
                     ProblemKeys keys) {
  bool success_ext_pose =
      graph.computeCovariance(keys.nav_state_key, timestamp);
  bool success_bias = graph.computeCovariance(keys.bias_state_key, timestamp);

  if (!success_ext_pose || !success_bias) {
    return Eigen::Matrix<double, 15, 15>::Identity();
  }

  // Assemble covariance and return
  Eigen::Matrix<double, 15, 15> covariance =
      Eigen::Matrix<double, 15, 15>::Zero();
  covariance.block<9, 9>(0, 0) =
      graph.getStates()
          .getState<ExtendedPoseParameterBlock>(keys.nav_state_key, timestamp)
          ->getCovariance();
  covariance.block<6, 6>(9, 9) =
      graph.getStates()
          .getState<ParameterBlock<6>>(keys.bias_state_key, timestamp)
          ->getCovariance();
  return covariance;
}

void marginalizeIMUState(ceres_nav::FactorGraph &graph, double timestamp_marg,
                         ProblemKeys keys) {
  std::vector<ceres_nav::StateID> state_ids_marg = {
      ceres_nav::StateID(keys.nav_state_key, timestamp_marg),
      ceres_nav::StateID(keys.bias_state_key, timestamp_marg)};

  graph.marginalizeStates(state_ids_marg);
}

} // namespace factor_graph_utils