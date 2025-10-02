#include <Eigen/Dense>

#include "lib/ExtendedPoseParameterBlock.h"
#include "lib/FactorGraph.h"
#include "lib/ParameterBlock.h"
#include "lib/StateCollection.h"

#include "lie/LieDirection.h"

#include "factors/AbsolutePositionFactor.h"
#include "factors/IMUPreintegrationFactor.h"
#include "factors/RelativeLandmarkFactor.h"
#include "factors/RelativePoseFactor.h"
#include "factors/MarginalizationPrior.h"

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
  auto state = std::make_shared<ExtendedPoseParameterBlock>();
  factor_graph.addState("x", 0.0, state);

  // Try adding the factor again
  REQUIRE(factor_graph.addFactor(state_ids, cost_function, 0.0));
  REQUIRE(factor_graph.numResidualBlocks() == 1);
}

/**
 * This test case creates a simple factor graph with two IMU states and one
 * landmark. Preintegration factors connect the IMU states, and landmark
 * factor connects the IMU states to the landmark. 
 * 
 * The test then calls getMarkovBlanketInfo() to get the information for states connected
 * to the first IMU state, and checks that the correct states and factors are returned.
 */
TEST_CASE("Test MarkovBlanketInfo") {
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
  IMUIncrement rmi0(Eigen::Matrix<double, 12, 12>::Identity(),
                    Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), 0.0,
                    Eigen::Vector3d(0, 0, -9.81), LieDirection::right);
  IMUIncrement rmi1(Eigen::Matrix<double, 12, 12>::Identity(),
                    Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), 1.0,
                    Eigen::Vector3d(0, 0, -9.81), LieDirection::right);

  ceres::CostFunction *preintegration_factor =
      new IMUPreintegrationFactor(rmi0, false);
  std::vector<StateID> state_ids = {StateID("X", 0.0), StateID("b", 0.0),
                                    StateID("X", 1.0), StateID("b", 1.0)};

  factor_graph.addFactor(state_ids, preintegration_factor, 1.0);
  REQUIRE(factor_graph.numResidualBlocks() == 1);
  REQUIRE(factor_graph.numParameterBlocks() == 4);

  // Add a landmark to the problem
  auto landmark =
      std::make_shared<ParameterBlock<3>>(Eigen::Vector3d(5.0, 0.0, 0.0));
  factor_graph.addState(StateID("landmark"), landmark);
  REQUIRE(factor_graph.numParameterBlocks() == 5);

  // Add a landmark factor connecting 0 and the landmark
  Eigen::Matrix3d landmark_cov = 0.01 * Eigen::Matrix3d::Identity();
  ceres::CostFunction *landmark_factor = new RelativeLandmarkFactor(
      Eigen::Vector3d(0.1, 0.2, 0.3), landmark_cov, 1.0, LieDirection::right,
      ExtendedPoseRepresentation::SE23);
  std::vector<StateID> landmark_factor_state_ids = {StateID("X", 0.0),
                                                    StateID("landmark")};
  factor_graph.addFactor(landmark_factor_state_ids, landmark_factor, 0.0);

  // Add another landmark factor connecting X1 and the landmark
  ceres::CostFunction *landmark_factor2 = new RelativeLandmarkFactor(
      Eigen::Vector3d(0.1, 0.2, 0.3), landmark_cov, 2.0, LieDirection::right,
      ExtendedPoseRepresentation::SE23);
  std::vector<StateID> landmark_factor2_state_ids = {StateID("X", 1.0),
                                                     StateID("landmark")};
  factor_graph.addFactor(landmark_factor2_state_ids, landmark_factor2, 1.0);

  // Now, say we want to marginalize out X0 and b0. Get the connected state
  // and factor information for this example factor graph
  std::vector<StateID> states_m = {StateID("X", 0.0), StateID("b", 0.0)};
  std::vector<ParameterBlockInfo> connected_states;
  std::vector<ceres::ResidualBlockId> factors_m;
  std::vector<ceres::ResidualBlockId> factors_r;
  bool success = factor_graph.getMarkovBlanketInfo(
      states_m, connected_states, factors_m, factors_r);

  REQUIRE(success);

  // The connected states should be X1, b1, and landmark
  REQUIRE(connected_states.size() == 3);
  for (auto const &state : connected_states) {
    StateID state_id = state.state_id;
    REQUIRE((state_id == StateID("X", 1.0) || state_id == StateID("b", 1.0) ||
             state_id == StateID("landmark")));
  }

  // factors_m should contain the preintegration factor and
  // the landmark factor connecting X0 and the landmark
  REQUIRE(factors_m.size() == 2);
  for (auto const &factor_id : factors_m) {
    auto cost_function_ptr = factor_graph.getCostFunction(factor_id);
    REQUIRE((cost_function_ptr == preintegration_factor ||
             cost_function_ptr == landmark_factor));
  }

  // Factors r should contain the other landmark factor
  REQUIRE(factors_r.size() == 1);
  auto cost_function_ptr = factor_graph.getCostFunction(factors_r[0]);
  REQUIRE(cost_function_ptr == landmark_factor2);
}