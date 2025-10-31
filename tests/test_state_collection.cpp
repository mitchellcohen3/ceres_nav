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

TEST_CASE("State ID Comparisons") {
  SECTION("Test timestamp rounding and equality") {
    StateID id1("pose", 1.2344, 1e-3);
    StateID id2("pose", 1.2338, 1e-3);

    REQUIRE(id1.timestamp().value() == 1.234);
    REQUIRE(id2.timestamp().value() == 1.234);
    REQUIRE(id1 == id2);
  }

  SECTION("Different keys are not equal") {
    StateID id3("landmark");
    StateID id4("landmark", 0.0);
    REQUIRE(id3 != id4);
  }

  SECTION("Change default precision") {
    StateID::setDefaultPrecision(1e-9);
    double test_time = 2.123456789;
    StateID id5("x", test_time);
    REQUIRE(id5.timestamp().value() == test_time);
    StateID::setDefaultPrecision(1e-6);
  }

  SECTION("StateID as map key") {
    std::map<StateID, int> state_map;

    StateID id1("x", 10.0);
    StateID id2("x", 20.0);
    StateID id3("x", 30.0);
    StateID id4("y");
    StateID id5("z");

    state_map[id1] = 1;
    state_map[id2] = 2;
    state_map[id3] = 3;
    state_map[id4] = 4;
    state_map[id5] = 5;
    REQUIRE(state_map.size() == 5);

    REQUIRE(state_map[id1] == 1);
    REQUIRE(state_map[id2] == 2);
    REQUIRE(state_map[id3] == 3);
    REQUIRE(state_map[id4] == 4);
    REQUIRE(state_map[id5] == 5);
  }
}

TEST_CASE("Add time-varying parameter block", "[StateCollection]") {
  StateCollection collection;
  auto block =
      std::make_shared<ParameterBlock<3>>(Eigen::Vector3d(1.0, 2.0, 3.0));
  StateID state_id("velocity", 10.0);
  REQUIRE(collection.addState(state_id, block));
  REQUIRE(collection.hasState(state_id));
  REQUIRE(collection.size() == 1);
  REQUIRE(collection.getNumberOfStatesForType("velocity") == 1);

  auto retrieved_block = collection.getState(state_id);
  REQUIRE(retrieved_block == block);

  // Try adding the same state again - should fail
  REQUIRE(!collection.addState(state_id, block));

  // Now, remove the state
  REQUIRE(collection.removeState(state_id));
  REQUIRE(!collection.hasState(state_id));
  REQUIRE(collection.size() == 0);
  REQUIRE(collection.getState(state_id) == nullptr);
  REQUIRE(collection.getNumberOfStatesForType("velocity") == 0);

  // Add in two states
  StateID state_id2("velocity", 12.0);
  REQUIRE(collection.addState(state_id, block));
  REQUIRE(collection.addState(state_id2, block));
  REQUIRE(collection.size() == 2);
  REQUIRE(collection.getNumberOfStatesForType("velocity") == 2);
}

TEST_CASE("Add static states", "[StateCollection]") {
  StateCollection collection;
  auto block =
      std::make_shared<ParameterBlock<3>>(Eigen::Vector3d(4.0, 5.0, 6.0));
  StateID id("landmark");
  REQUIRE(collection.addState(id, block));
  REQUIRE(collection.hasState(id));
  REQUIRE(collection.size() == 1);
  auto retrieved_block = collection.getState(id);
  REQUIRE(retrieved_block == block);

  // Try adding the same static state again - should fail
  REQUIRE(!collection.addState(id, block));
  // Now, remove the static state
  REQUIRE(collection.removeState(id));
  REQUIRE(!collection.hasState(id));
  REQUIRE(collection.size() == 0);
  REQUIRE(collection.getState(id) == nullptr);
}

TEST_CASE("Test multiple state types", "[StateCollection]") {
  StateCollection collection;
  auto block1 =
      std::make_shared<ParameterBlock<3>>(Eigen::Vector3d(1.0, 2.0, 3.0));
  auto block2 =
      std::make_shared<ParameterBlock<4>>(Eigen::Vector4d(4.0, 5.0, 6.0, 7.0));
  StateID id1("x", 0.0);
  StateID id2("y", 0.0);

  REQUIRE(collection.addState(id1, block1));
  REQUIRE(collection.addState(id2, block2));

  // Try retrieving both states
  auto retrieved_block1 = collection.getState(id1);
  auto retrieved_block2 = collection.getState(id2);

  REQUIRE(retrieved_block1 == block1);
  REQUIRE(retrieved_block2 == block2);
}

TEST_CASE("Test Timestamp Operations") {
  StateCollection collection;

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
    StateID state_id("x", cur_stamp);
    collection.addState(state_id, state);
    REQUIRE(collection.getState(state_id) != nullptr);
    REQUIRE(collection.getState(state_id)->getEstimate().isApprox(x));

    state_values.push_back(x);
    timestamps.push_back(cur_stamp);
    cur_stamp += dt;
  }

  // Test the oldest/newest state retrieval
  double oldest_stamp, latest_stamp;
  REQUIRE(collection.getOldestStamp("x", oldest_stamp));
  REQUIRE(collection.getLatestStamp("x", latest_stamp));

  // Check that the oldest and latest timestamps approximately match
  REQUIRE_THAT(oldest_stamp,
               Catch::Matchers::WithinAbs(timestamps.front(), 1e-5));
  REQUIRE_THAT(latest_stamp,
               Catch::Matchers::WithinAbs(timestamps.back(), 1e-5));

  // Get the timestamps for the states
  std::vector<double> timestamps_retrieved;
  REQUIRE(collection.getTimesForState("x", timestamps_retrieved));
  REQUIRE(timestamps_retrieved.size() == num_states);

  // Get the oldest and latest states from the collection
  std::shared_ptr<ParameterBlock<3>> oldest_state =
      collection.getOldestState<ParameterBlock<3>>("x");
  std::shared_ptr<ParameterBlock<3>> latest_state =
      collection.getLatestState<ParameterBlock<3>>("x");

  REQUIRE(oldest_state != nullptr);
  REQUIRE(latest_state != nullptr);
  REQUIRE(oldest_state->getEstimate().isApprox(state_values.front()));
  REQUIRE(latest_state->getEstimate().isApprox(state_values.back()));
}

TEST_CASE("Test estimate pointer functions", "[StateCollection]") {
  StateCollection collection;

  Eigen::Vector3d x(1.0, 2.0, 3.0);
  StateID state_id("x", 100.0);
  std::shared_ptr<ParameterBlock<3>> state =
      std::make_shared<ParameterBlock<3>>(x);
  collection.addState(state_id, state);

  auto retrieved_state =
      collection.getStateByEstimatePointer(state->estimatePointer());
  REQUIRE(retrieved_state != nullptr);
  REQUIRE(retrieved_state->getEstimate().isApprox(x));

  // Now, try to get the StateID by the estimate pointer
  StateID retrieved_id;
  REQUIRE(collection.getStateIDByEstimatePointer(state->estimatePointer(),
                                                 retrieved_id));
  REQUIRE(retrieved_id == state_id);
}