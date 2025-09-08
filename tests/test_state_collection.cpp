#include <Eigen/Dense>

#include "lib/ExtendedPoseParameterBlock.h"
#include "lib/FactorGraph.h"
#include "lib/ParameterBlock.h"
#include "lib/PoseParameterBlock.h"
#include "lib/StateCollection.h"

#include "lie/SE23.h"
#include "lie/SE3.h"
#include "lie/SO3.h"

#include "lib/StateId.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

using namespace ceres_nav;

TEST_CASE("Test Add/Remove Operations") {
  StateCollection state_collection;

  std::shared_ptr<ParameterBlock<3>> state =
      std::make_shared<ParameterBlock<3>>(Eigen::Vector3d(1.0, 2.0, 3.0));

  state_collection.addState("x", 0.0, state);
  state_collection.addState("x", 1.0, state);
  state_collection.addState("x", 2.0, state);

  REQUIRE(state_collection.hasState("x", 0.0));
  REQUIRE(state_collection.hasState("x", 1.0));
  REQUIRE(state_collection.hasState("x", 2.0));
  REQUIRE(!state_collection.hasState("x", 3.0));
  REQUIRE(state_collection.getNumStateTypes() == 1);
  REQUIRE(state_collection.getNumStatesForType("x") == 3);

  // Now, try removing a state
  state_collection.removeState("x", 1.0);
  REQUIRE(state_collection.getNumStatesForType("x") == 2);
  REQUIRE(!state_collection.hasState("x", 1.0));

  // Remove the other two states
  state_collection.removeState("x", 0.0);
  state_collection.removeState("x", 2.0);
  REQUIRE(state_collection.getNumStatesForType("x") == 0);
}

TEST_CASE("Test Multiple State Types") {
  StateCollection state_collection;
  Eigen::Vector3d x0 = Eigen::Vector3d(1.0, 2.0, 3.0);
  Eigen::Matrix<double, 4, 1> x1 = {1.0, 2.0, 3.0, 4.0};

  std::shared_ptr<ParameterBlock<3>> state =
      std::make_shared<ParameterBlock<3>>(x0);
  std::shared_ptr<ParameterBlock<4>> state_2 =
      std::make_shared<ParameterBlock<4>>(x1);

  // Add the states to the collection
  state_collection.addState("x1", 0.0, state);
  state_collection.addState("x2", 0.0, state_2);

  // Try retrieving the states
  std::shared_ptr<ParameterBlockBase> retrieved_state =
      state_collection.getState("x1", 0.0);
  std::shared_ptr<ParameterBlockBase> retrieved_state_2 =
      state_collection.getState("x2", 0.0);

  REQUIRE(retrieved_state != nullptr);
  REQUIRE(retrieved_state_2 != nullptr);

  // Check the estimates
  Eigen::Vector3d estimate_x0 = retrieved_state->getEstimate();
  Eigen::Vector4d estimate_x1 = retrieved_state_2->getEstimate();
  REQUIRE(estimate_x0.isApprox(x0));
  REQUIRE(estimate_x1.isApprox(x1));

  // Try adding a PoseParameterBlock
  Eigen::Matrix4d pose =
      SE3::fromComponents(SO3::expMap(Eigen::Vector3d(0.1, 0.2, 0.3)),
                          Eigen::Vector3d(1.0, 2.0, 3.0));
  std::shared_ptr<PoseParameterBlock> pose_state =
      std::make_shared<PoseParameterBlock>(pose);

  state_collection.addState("pose", 0.0, pose_state);

  // Retrieve the pose state using the method that downcasts to
  // PoseParameterBlock
  std::shared_ptr<PoseParameterBlock> retrieved_pose_state =
      state_collection.getState<PoseParameterBlock>("pose", 0.0);
  REQUIRE(retrieved_pose_state != nullptr);
  REQUIRE(retrieved_pose_state->pose().isApprox(pose));
}

TEST_CASE("Test Timestamp Operations") {
  StateCollection state_collection;

  Eigen::Vector3d x(1.0, 2.0, 3.0);
  // Add states with timestamps and then try to retrieve them
  double dt = 0.02;
  double cur_stamp = 102.523;
  int num_states = 100;
  std::vector<double> timestamps;
  std::vector<Eigen::Vector3d> state_values;
  for (int i = 0; i < num_states; ++i) {

    // Update the state
    x(0) = x(0) + 0.1 * i;
    x(1) = x(1) + 0.2 * i;
    x(2) = x(2) + 0.3 * i;

    std::shared_ptr<ParameterBlock<3>> state =
        std::make_shared<ParameterBlock<3>>(x);
    state_collection.addState("x", cur_stamp, state);

    std::shared_ptr<ParameterBlockBase> retrieved_state =
        state_collection.getState("x", cur_stamp);
    REQUIRE(retrieved_state != nullptr);

    state_values.push_back(x);
    timestamps.push_back(cur_stamp);
    cur_stamp += dt;
  }

  // Test the oldest/newest state retrieval
  double oldest_stamp, latest_stamp;
  REQUIRE(state_collection.getOldestStamp("x", oldest_stamp));
  REQUIRE(state_collection.getLatestStamp("x", latest_stamp));

  // Check that the oldest and latest timestamps approximately match
  REQUIRE_THAT(oldest_stamp,
               Catch::Matchers::WithinAbs(timestamps.front(), 1e-5));
  REQUIRE_THAT(latest_stamp,
               Catch::Matchers::WithinAbs(timestamps.back(), 1e-5));

  // Get the timestamps for the states
  std::vector<double> timestamps_retrieved;
  REQUIRE(state_collection.getTimesForState("x", timestamps_retrieved));
  REQUIRE(timestamps_retrieved.size() == num_states);

  // Get the oldest and latest states from the collection
  std::shared_ptr<ParameterBlock<3>> oldest_state =
      state_collection.getOldestState<ParameterBlock<3>>("x");
  std::shared_ptr<ParameterBlock<3>> latest_state =
      state_collection.getLatestState<ParameterBlock<3>>("x");

  REQUIRE(oldest_state != nullptr);
  REQUIRE(latest_state != nullptr);
  REQUIRE(oldest_state->getEstimate().isApprox(state_values.front()));
  REQUIRE(latest_state->getEstimate().isApprox(state_values.back()));
}

TEST_CASE("Test Static States") {
    StateCollection state_collection;

    auto state_1 = std::make_shared<ParameterBlock<3>>(Eigen::Vector3d(1.0, 2.0, 3.0));
    auto state_2 = std::make_shared<ParameterBlock<3>>(Eigen::Vector3d(4.0, 5.0, 6.0));

    state_collection.addStaticState("x", state_1);
    state_collection.addState("y", 0.0, state_2);

    REQUIRE(state_collection.hasStaticState("x"));
    REQUIRE(!state_collection.hasStaticState("y"));
    REQUIRE(state_collection.hasStateType("x"));
    REQUIRE(state_collection.hasState("y", 0.0));

    auto static_state = state_collection.getStaticState<ParameterBlock<3>>("x");
    REQUIRE(static_state != nullptr);
    REQUIRE(static_state->getEstimate().isApprox(Eigen::Vector3d(1.0, 2.0, 3.0)));

    // Try getting a state by it's estimate pointer
    double* state_ptr = state_1->estimatePointer();
    auto found_state = state_collection.getStateByEstimatePointer(state_ptr);
    REQUIRE(found_state != nullptr);
    REQUIRE(found_state->getEstimate().isApprox(Eigen::Vector3d(1.0, 2.0, 3.0)));

    double* state_ptr_2 = state_2->estimatePointer();
    auto found_state_2 = state_collection.getStateByEstimatePointer(state_ptr_2);
    REQUIRE(found_state_2 != nullptr);
    REQUIRE(found_state_2->getEstimate().isApprox(Eigen::Vector3d(4.0, 5.0, 6.0)));

    // Test retrieving a state by StateID
    StateID static_id("x");
    auto retrieved_static_state = state_collection.getState(static_id);
    REQUIRE(retrieved_static_state != nullptr);
    REQUIRE(retrieved_static_state->estimatePointer() == state_ptr);

    // Remove the static state
    state_collection.removeStaticState("x");
    REQUIRE(!state_collection.hasStaticState("x"));
}

// TEST_CASE("Test StateID") {
//     StateID id1("x", 0.1);
//     StateID id2("y");

//     REQUIRE(!id1.isStatic());
//     REQUIRE(id2.isStatic());
// }