#include <Eigen/Dense>

#include "lib/ExtendedPoseParameterBlock.h"
#include "lib/FactorGraph.h"
#include "lib/ParameterBlock.h"
#include "lib/StateCollection.h"

#include "lie/LieDirection.h"

#include "factors/AbsolutePositionFactor.h"
#include "factors/RelativePoseFactor.h"
#include "factors/RelativeLandmarkFactor.h"
#include "factors/IMUPreintegrationFactor.h"

#include <catch2/catch_test_macros.hpp>

using namespace ceres_nav;

TEST_CASE("Test Add/Remove states from FactorGraph") {
  FactorGraph factor_graph;
  std::shared_ptr<ParameterBlock<3>> state =
      std::make_shared<ParameterBlock<3>>(Eigen::Vector3d(1.0, 2.0, 3.0));

  factor_graph.addState("x", 0.0, state);
  REQUIRE(factor_graph.getStates().hasState("x", 0.0));
  REQUIRE(factor_graph.numParameterBlocks() == 1);

  // Get the state pointer for this state
  std::vector<StateID> state_ids = {StateID("x", 0.0)};
  std::vector<double *> estimate_ptrs;
  factor_graph.getStatePointers(state_ids, estimate_ptrs);
  REQUIRE(estimate_ptrs.size() == 1);
  REQUIRE(estimate_ptrs[0] == state->estimatePointer());

  factor_graph.removeState("x", 0.0);
  REQUIRE(!factor_graph.getStates().hasState("x", 0.0));
  REQUIRE(factor_graph.numParameterBlocks() == 0);

  // Now, add a non-timestamped state
  std::shared_ptr<ParameterBlock<3>> static_state =
      std::make_shared<ParameterBlock<3>>(Eigen::Vector3d(4.0, 5.0, 6.0));
  factor_graph.addState(StateID("static_state"), static_state);
  REQUIRE(factor_graph.getStates().hasStaticState("static_state"));
  REQUIRE(factor_graph.numParameterBlocks() == 1);

  // Remove the state
  factor_graph.removeState(StateID("static_state"));
  REQUIRE(!factor_graph.getStates().hasStaticState("static_state"));
  REQUIRE(factor_graph.numParameterBlocks() == 0);
}

TEST_CASE("Test setting states constant") {
  FactorGraph factor_graph;
  std::shared_ptr<ParameterBlock<3>> state =
      std::make_shared<ParameterBlock<3>>(Eigen::Vector3d(1.0, 2.0, 3.0));
  factor_graph.addState("x", 0.0, state);
  factor_graph.setConstant("x", 0.0);
  REQUIRE(factor_graph.isConstant("x", 0.0));

  factor_graph.setVariable("x", 0.0);
  REQUIRE(!factor_graph.isConstant("x", 0.0));
}

TEST_CASE("Test adding a factor") {
  FactorGraph factor_graph;
  ceres::CostFunction *cost_function = new AbsolutePositionFactor(
      Eigen::Vector3d(1.0, 2.0, 3.0), LieDirection::right,
      Eigen::Matrix3d::Identity());

  std::vector<StateID> state_ids = {StateID("x", 0.0)};
  bool success = factor_graph.addFactor(state_ids, cost_function, 0.0);
  REQUIRE(!success); // Should fail because state does not exist

  // Add in the state to the factor graph
  auto state =
      std::make_shared<ExtendedPoseParameterBlock>();
  factor_graph.addState("x", 0.0, state);

  // Try adding the factor again
  REQUIRE(factor_graph.addFactor(state_ids, cost_function, 0.0));
  REQUIRE(factor_graph.numResidualBlocks() == 1);
}

/**
 * This test case creates a simple factor graph with three IMU states and one landmark. 
 * Preintegration factors connect the IMU states, and a landmark factor connects the last IMU state to the landmark.
 */
TEST_CASE("Test marginalization functions") {
  FactorGraph factor_graph;

  // Create and add IMU states
  auto X0 = std::make_shared<ExtendedPoseParameterBlock>();
  auto b0 = std::make_shared<ParameterBlock<6>>(Eigen::VectorXd::Zero(6));
  auto X1 = std::make_shared<ExtendedPoseParameterBlock>();
  auto b1 = std::make_shared<ParameterBlock<6>>(Eigen::VectorXd::Zero(6));

  // Add states to the factor graph
  factor_graph.addState("X", 0.0, X0);
  factor_graph.addState("b", 0.0, b0);
  factor_graph.addState("X", 1.0, X1);
  factor_graph.addState("b", 1.0, b1);

  // Create and add IMU preintegration factor between X0, b0 and X1, b1
  IMUIncrement rmi0(
      Eigen::Matrix<double, 12, 12>::Identity(), Eigen::Vector3d::Zero(),
      Eigen::Vector3d::Zero(), 0.0, Eigen::Vector3d(0, 0, -9.81),
      LieDirection::right);
  IMUIncrement rmi1(
      Eigen::Matrix<double, 12, 12>::Identity(), Eigen::Vector3d::Zero(),
      Eigen::Vector3d::Zero(), 1.0, Eigen::Vector3d(0, 0, -9.81),
      LieDirection::right);
  
  auto preintegration_factor = new IMUPreintegrationFactor(rmi0, false);
  std::vector<StateID> state_ids = {
    StateID("X", 0.0), StateID("b", 0.0),
    StateID("X", 1.0), StateID("b", 1.0)
  };

  factor_graph.addFactor(state_ids, preintegration_factor, 1.0);
  REQUIRE(factor_graph.numResidualBlocks() == 1);
  REQUIRE(factor_graph.numParameterBlocks() == 4);

  // Check the connected factors to 
}