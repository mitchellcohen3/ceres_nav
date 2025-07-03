#include <Eigen/Dense>

#include "lib/ExtendedPoseParameterBlock.h"
#include "lib/FactorGraph.h"
#include "lib/ParameterBlock.h"
#include "lib/StateCollection.h"

#include "lie/SE23.h"
#include "lie/SO3.h"


#include <catch2/catch_test_macros.hpp>

TEST_CASE("Test FactorGraph") {
  StateCollection state_collection;

  // Create a new parameter block
  std::shared_ptr<ParameterBlock<3>> state =
      std::make_shared<ParameterBlock<3>>();
  state->setEstimate(Eigen::Vector3d(1.0, 2.0, 3.0));

  state_collection.addState("x", 0.0, state);
  state_collection.addState("x", 1.0, state);
  state_collection.addState("x", 2.0, state);

  REQUIRE(state_collection.hasState("x", 0.0));
  REQUIRE(state_collection.hasState("x", 1.0));
  REQUIRE(state_collection.hasState("x", 2.0));
  REQUIRE(!state_collection.hasState("x", 3.0));
  REQUIRE(state_collection.getNumStateTypes() == 1);
  REQUIRE(state_collection.getNumStatesForType("x") == 3);

  // Remove a state
  state_collection.removeState("x", 1.0);
  REQUIRE(state_collection.getNumStatesForType("x") == 2);
  REQUIRE(!state_collection.hasState("x", 1.0));
  // Remove the other two states
  state_collection.removeState("x", 0.0);
  state_collection.removeState("x", 2.0);
  REQUIRE(state_collection.getNumStatesForType("x") == 0);

  // Add another state
  std::shared_ptr<ParameterBlock<3>> state_2 = std::make_shared<ParameterBlock<3>>("y");
  state_2->setEstimate(Eigen::Vector3d(1.0, 2.0, 3.0));  
  state_collection.addState("y", 0, state_2);
  REQUIRE(state_collection.getNumStateTypes() == 1);
  REQUIRE(state_collection.getNumStatesForType("y") == 1);
  REQUIRE(state_collection.hasState("y", 0.0));
  std::shared_ptr<ParameterBlockBase> state_new = state_collection.getState("y", 0.0);

  // Downcast to the correct type
  auto derived_state = std::dynamic_pointer_cast<ParameterBlock<3, 3>>(state);
  REQUIRE(derived_state != nullptr);
  Eigen::Vector3d estimate = derived_state->getEstimate();
  REQUIRE(estimate.isApprox(Eigen::Vector3d(1.0, 2.0, 3.0)));
}

// TEST_CASE("Test Factor Graph") {
//   ceres_nav::FactorGraph factor_graph;
//   factor_graph.addState(
//       "x", 0.0,
//       std::make_shared<ParameterBlock<3, 3>>(Eigen::Vector3d(1.0, 2.0, 3.0)));
//   factor_graph.addState(
//       "x", 1.0,
//       std::make_shared<ParameterBlock<3, 3>>(Eigen::Vector3d(4.0, 5.0, 6.0)));
//   factor_graph.addState(
//       "x", 2.0,
//       std::make_shared<ParameterBlock<3, 3>>(Eigen::Vector3d(7.0, 8.0, 9.0)));

//   // Get the state data
//   Eigen::Vector3d estimate =
//       factor_graph.getStates().getState("x", 2.0)->getEstimate();
//   REQUIRE(estimate.isApprox(Eigen::Vector3d(7.0, 8.0, 9.0)));
// }