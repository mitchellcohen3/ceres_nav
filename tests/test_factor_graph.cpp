#include <Eigen/Dense>

#include "lib/ExtendedPoseParameterBlock.h"
#include "lib/FactorGraph.h"
#include "lib/ParameterBlock.h"
#include "lib/StateCollection.h"

#include "lie/SE23.h"
#include "lie/SO3.h"

#include <catch2/catch_test_macros.hpp>

TEST_CASE("Test Add/Remove states from FactorGraph") {
  ceres_nav::FactorGraph factor_graph;
  std::shared_ptr<ParameterBlock<3>> state =
      std::make_shared<ParameterBlock<3>>(Eigen::Vector3d(1.0, 2.0, 3.0));
  
  factor_graph.addState(
      "x", 0.0, state);
  REQUIRE(factor_graph.getStates().hasState("x", 0.0));
  REQUIRE(factor_graph.numParameterBlocks() == 1);

  // Get the state pointer for this state
  std::vector<ceres_nav::StateID> state_ids = {ceres_nav::StateID("x", 0.0)};
  std::vector<double *> estimate_ptrs;
  factor_graph.getStatePointers(state_ids, estimate_ptrs);
  REQUIRE(estimate_ptrs.size() == 1);
  REQUIRE(estimate_ptrs[0] == state->estimatePointer());

  factor_graph.removeState("x", 0.0);
  REQUIRE(!factor_graph.getStates().hasState("x", 0.0));
  REQUIRE(factor_graph.numParameterBlocks() == 0);
}

TEST_CASE("Test setting states constant") {
  ceres_nav::FactorGraph factor_graph;
  std::shared_ptr<ParameterBlock<3>> state =
      std::make_shared<ParameterBlock<3>>(Eigen::Vector3d(1.0, 2.0, 3.0));
  factor_graph.addState("x", 0.0, state);
  factor_graph.setConstant("x", 0.0);
  REQUIRE(factor_graph.isConstant("x", 0.0));

  factor_graph.setVariable("x", 0.0);
  REQUIRE(!factor_graph.isConstant("x", 0.0));
}