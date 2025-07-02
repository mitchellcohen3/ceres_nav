#include "lib/FactorGraph.h"
#include "lib/Covariance.h"

namespace ceres_swf {

FactorGraph::FactorGraph() : problem_(default_problem_options_) {}

void FactorGraph::addState(const std::string &name, double timestamp,
                           std::shared_ptr<ParameterBlockBase> state) {
  states_.addState(name, timestamp, state);

  // Get the mean pointer
  double *estimate_ptr = state->estimatePointer();
  int size = state->dimension();

  // Check to ensure that the size actually matches
  //   int size_2 = static_cast<int>(state->getEstimate().size());

  ceres::LocalParameterization *local_parameterization_ptr =
      state->getLocalParameterizationPointer();

  if (local_parameterization_ptr != nullptr) {
    problem_.AddParameterBlock(estimate_ptr, state->dimension(),
                               local_parameterization_ptr);
  } else {
    problem_.AddParameterBlock(estimate_ptr, size);
  }
}

// Add a factor to the problem
void FactorGraph::addFactor(const std::vector<StateID> &state_ids,
                            ceres::CostFunction *cost_function, double stamp,
                            ceres::LossFunction *loss_function) {
  // Build vector of state pointers
  std::vector<double *> state_ptrs;
  for (auto &state_id : state_ids) {
    state_ptrs.push_back(
        states_.getState(state_id.ID, state_id.timestamp)->estimatePointer());
  }

  // Add the residual block to the problem
  ceres::ResidualBlockId residual_id =
      problem_.AddResidualBlock(cost_function, loss_function, state_ptrs);

  // Add to map
  residual_blocks_to_cost_function_map.insert({residual_id, cost_function});

  // TODO: store the graph structure
}

void FactorGraph::solve() {
  /** check if config is valid */
  std::string OptionsError;
  if (!solver_options_.IsValid(&OptionsError)) {
    // PRINT_ERROR("The given solver options are wrong: ", OptionsError);
    std::cout << "The given solver options are wrong: " << OptionsError
              << std::endl;
  } else {
    /** call ceres to solve the optimization problem */
    ceres::Solve(solver_options_, &problem_, &summary_);
    last_solver_duration = summary_.total_time_in_seconds;
    total_solver_duration += summary_.total_time_in_seconds;
    num_solver_iterations += summary_.iterations.size();
  }
}

void FactorGraph::solve(ceres::Solver::Options Options) {
  solver_options_ = Options;
  this->solve();
}

bool FactorGraph::getStatePointers(const std::vector<StateID> &state_ids,
                                   std::vector<double *> &state_ptrs) {
  // Get the pointers to the states
  for (auto &state_id : state_ids) {
    if (!states_.hasState(state_id.ID, state_id.timestamp)) {
      return false;
    }

    // Ensure that this is actually in the Ceres problem
    double *state_ptr =
        states_.getState(state_id.ID, state_id.timestamp)->estimatePointer();
    if (!problem_.HasParameterBlock(state_ptr)) {
      return false;
    }
    state_ptrs.push_back(state_ptr);
  }

  return true;
}

bool FactorGraph::getConnectedFactorIDs(
    const std::vector<double *> &state_ptrs,
    std::vector<ceres::ResidualBlockId> &factors) {

  // Loop through the state pointers and get all
  // connected residuals
  for (double *const state_ptr : state_ptrs) {
    if (!problem_.HasParameterBlock(state_ptr)) {
      // PRINT_ERROR("State doesn't exist in the graph!");
      return false;
    }

    // Query residual IDs
    std::vector<ceres::ResidualBlockId> residuals;
    problem_.GetResidualBlocksForParameterBlock(state_ptr, &residuals);

    // For each factor, check if it's already in the vector
    for (auto const &residual : residuals) {
      if (std::find(factors.begin(), factors.end(), residual) ==
          factors.end()) {
        factors.push_back(residual);
      }
    }
  }

  return true;
}

bool FactorGraph::getConnectedStatePointers(
    const std::vector<ceres::ResidualBlockId> &factors,
    std::vector<double *> &state_ptrs) {
  //
  for (auto const &factor : factors) {
    std::vector<double *> cur_state_pointers;

    // Get the states that are connected to a certain factor
    problem_.GetParameterBlocksForResidualBlock(factor, &cur_state_pointers);
    // Add only the unique states
    for (auto const &state_ptr : cur_state_pointers) {
      if (std::find(state_ptrs.begin(), state_ptrs.end(), state_ptr) ==
          state_ptrs.end()) {
        state_ptrs.push_back(state_ptr);
      }
    }
  }

  if (state_ptrs.empty()) {
    return false;
  }

  return true;
}

void FactorGraph::removeState(const std::string &name, double timestamp) {
  if (states_.hasState(name, timestamp)) {
    // Remove from the Ceres problem
    problem_.RemoveParameterBlock(
        states_.getState(name, timestamp)->estimatePointer());
    // Remove from the StateCollection
    states_.removeState(name, timestamp);
  } else {
    // PRINT_ERROR("State doesn't exist at: ", Timestamp, " Type: ", Name);
    std::cout << "State doesn't exist at: " << timestamp << " Type: " << name
              << std::endl;
  }
}

bool FactorGraph::computeCovariance(const std::string &key, double timestamp) {
  const bool success = calculateCovariance(problem_, states_, key, timestamp);
  return success;
}

bool getMarkovBlanketInfo(const std::vector<StateID> &States_m,
                          std::vector<double *> &ConnectedStatePtrs,
                          std::vector<ceres::ResidualBlockId> &Factors_m,
                          std::vector<ceres::ResidualBlockId> &Factors_r,
                          std::vector<StateID> &ConnectedStateIDs);

} // namespace ceres_swf