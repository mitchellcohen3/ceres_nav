#include "lib/FactorGraph.h"
#include "lib/Covariance.h"
#include "lib/Marginalization.h"

#include "utils/VectorMath.h"
#include "utils/VectorTypes.h"

#include "factors/MarginalizationPrior.h"
#include "utils/Timer.h"

#include <glog/logging.h>
#include <thread>

namespace ceres_nav {

FactorGraph::FactorGraph() : problem_(default_problem_options_) {}

FactorGraph::FactorGraph(ceres::Solver::Options solver_options)
    : problem_(default_problem_options_), solver_options_(solver_options) {}

void FactorGraph::addState(const std::string &name, double timestamp,
                           std::shared_ptr<ParameterBlockBase> state) {
  states_.addState(name, timestamp, state);

  // Get the mean pointer
  double *estimate_ptr = state->estimatePointer();
  int size = state->dimension();

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
      LOG(ERROR) << "State not found in collection: " << state_id.ID
                 << " at timestamp: " << state_id.timestamp;
      return false;
    }

    // Ensure that this is actually in the Ceres problem
    double *state_ptr =
        states_.getState(state_id.ID, state_id.timestamp)->estimatePointer();
    if (!problem_.HasParameterBlock(state_ptr)) {
      LOG(ERROR) << "State not found in Ceres problem: " << state_id.ID
                 << " at timestamp: " << state_id.timestamp;
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

    // TODO: remove relevant residuals from the cost function map
    // double state_ptr = states_.getState(name, timestamp)->estimatePointer();

    // Remove from the Ceres problem
    problem_.RemoveParameterBlock(
        states_.getState(name, timestamp)->estimatePointer());
    // Remove from the StateCollection
    states_.removeState(name, timestamp);
  } else {
    LOG(ERROR) << "State not found in collection: " << name
               << " at timestamp: " << timestamp;
  }
}

void FactorGraph::setConstant(const std::string &name, double timestamp) {
  if (states_.hasState(name, timestamp)) {
    problem_.SetParameterBlockConstant(
        states_.getState(name, timestamp)->estimatePointer());
  } else {
    LOG(ERROR) << "State not found in collection: " << name
               << " at timestamp: " << timestamp;
  }
}

void FactorGraph::setVariable(const std::string &name, double timestamp) {
  if (states_.hasState(name, timestamp)) {
    problem_.SetParameterBlockVariable(
        states_.getState(name, timestamp)->estimatePointer());
  } else {
    LOG(ERROR) << "State not found in collection: " << name
               << " at timestamp: " << timestamp;
  }
}

bool FactorGraph::computeCovariance(const std::string &key, double timestamp) {
  const bool success = calculateCovariance(problem_, states_, key, timestamp);
  return success;
}

bool FactorGraph::marginalizeStates(std::vector<StateID> states_m) {
  Timer marg_timer;
  marg_timer.tic();

  Timer get_info_timer;
  get_info_timer.tic();

  if (states_m.empty()) {
    LOG(ERROR) << "No states to marginalize.";
    return false;
  }

  std::vector<double *> state_ptrs_m;
  if (!getStatePointers(states_m, state_ptrs_m)) {
    return false;
  }
  int marginal_size = 0;
  for (const double *state_ptr : state_ptrs_m) {
    marginal_size += problem_.ParameterBlockLocalSize(state_ptr);
  }

  // Get the connected states
  std::vector<double *> connected_state_ptrs;
  std::vector<ceres::ResidualBlockId> connected_factors;
  std::vector<int> global_size;
  std::vector<int> local_size;
  std::vector<StateID> state_ids;
  std::vector<const ceres::LocalParameterization *> local_param_ptrs;
  getMarginalizationInfo(state_ptrs_m, connected_state_ptrs, connected_factors,
                         global_size, local_size, state_ids, local_param_ptrs);
  double info_duration = get_info_timer.toc() * 1e-3;
  marginalization_timing_stats_["marginalization_info_duration"] =
      info_duration;

  if (!connected_state_ptrs.empty()) {
    // Evaluate the sub-problem involving marginalized states and their
    // connected states and factors
    Timer marg_eval_timer;
    marg_eval_timer.tic();

    ceres::Problem::EvaluateOptions options;
    options.apply_loss_function = true;
    options.num_threads = static_cast<int>(std::thread::hardware_concurrency());

    /** set which states should be evaluated (base + the connected ones) */
    std::vector<double *> combined_states;
    combined_states.insert(combined_states.end(), state_ptrs_m.begin(),
                           state_ptrs_m.end());
    combined_states.insert(combined_states.end(), connected_state_ptrs.begin(),
                           connected_state_ptrs.end());
    options.parameter_blocks = combined_states;

    // Set the resiudal blocks to evaluate
    options.residual_blocks = connected_factors;

    ceres::CRSMatrix JacobianCRS;
    std::vector<double> ResidualVec;
    problem_.Evaluate(options, nullptr, &ResidualVec, nullptr, &JacobianCRS);

    marg_eval_timer.toc();
    double marg_eval_duration = marg_eval_timer.toc() * 1e-3;
    marginalization_timing_stats_["marginalization_evaluation_duration"] =
        marg_eval_duration;

    // Map the error to a vector
    ceres_nav::Vector residual_vec =
        Eigen::Map<ceres_nav::Vector>(ResidualVec.data(), ResidualVec.size());

    // Convert the Jacobian to an Eigen Matrix
    ceres_nav::Matrix jacobian;
    ceres_nav::CRSToMatrix(JacobianCRS, jacobian);

    Timer schur_compl_timer;
    schur_compl_timer.tic();
    // Compute the marginalziation
    ceres_nav::Matrix jacobian_marg;
    ceres_nav::Vector residual_marg;
    ceres_nav::Marginalize(residual_vec, jacobian, residual_marg, jacobian_marg,
                           marginal_size, 1.0);
    schur_compl_timer.toc();
    double schur_compl_duration = schur_compl_timer.toc() * 1e-3;
    marginalization_timing_stats_["marginalization_schur_complement_duration"] =
        schur_compl_duration;

    // Store the original states
    std::vector<ceres_nav::Vector> original_states;
    for (int n = 0; n < connected_state_ptrs.size(); n++) {
      ceres_nav::VectorRef<double, Eigen::Dynamic> state_vec(
          connected_state_ptrs.at(n), global_size.at(n));
      original_states.emplace_back(state_vec);
    }

    // Add a marginal prior
    ceres_nav::MarginalizationPrior *marg_prior =
        new ceres_nav::MarginalizationPrior(
            local_size, global_size, original_states, local_param_ptrs,
            jacobian_marg, residual_marg, state_ids);
    ceres::ResidualBlockId marginalization_id =
        problem_.AddResidualBlock(marg_prior, nullptr, connected_state_ptrs);

    residual_blocks_to_cost_function_map.insert(
        {marginalization_id, marg_prior});
  } else {
    LOG(WARNING) << "No connected states found for marginalization. "
                    "Marginalized states will get deleted directly.";
  }

  // Remove marginalized states in reverse order
  std::reverse(states_m.begin(), states_m.end());
  for (const StateID &state : states_m) {
    removeState(state.ID, state.timestamp);
  }

  // Get the marginalization time
  marginalization_duration = marg_timer.toc() * 1e-3;
  return true;
}

void FactorGraph::getMarginalizationInfo(
    const std::vector<double *> state_ptrs_m,
    std::vector<double *> &connected_state_ptrs,
    std::vector<ceres::ResidualBlockId> &connected_factors,
    std::vector<int> &state_sizes, std::vector<int> &local_sizes,
    std::vector<StateID> &state_ids,
    std::vector<const ceres::LocalParameterization *> &local_param_ptrs) const {

  // Sets to store unique state pointers
  std::set<double *> base_state_ptrs;
  std::set<ceres::ResidualBlockId> connected_factor_ids;
  for (double *const state_ptr : state_ptrs_m) {
    if (!problem_.HasParameterBlock(state_ptr)) {
      LOG(ERROR) << "State pointer not found in Ceres problem.";
      return;
    }

    if (base_state_ptrs.count(state_ptr) != 0) {
      LOG(ERROR) << "Duplicate state pointer for marginalized states!";
      return;
    }

    base_state_ptrs.emplace(state_ptr);
    // Get the connected factors
    std::vector<ceres::ResidualBlockId> factors;
    problem_.GetResidualBlocksForParameterBlock(state_ptr, &factors);
    for (ceres::ResidualBlockId factor : factors) {
      if (connected_factor_ids.count(factor) == 0) {
        connected_factor_ids.emplace(factor);
      }
    }
  }

  // Now that we have all the connected factors, get all states involed in those
  // factors
  std::set<double *> connected_state_ptrs_set;
  for (const ceres::ResidualBlockId factor : connected_factor_ids) {
    // Get the states that are connected to this factor
    std::vector<double *> cur_state_ptrs;
    problem_.GetParameterBlocksForResidualBlock(factor, &cur_state_ptrs);

    // For each of these states, check if they were already in the
    // base state pointers or connected state pointers
    // Only add unique states
    for (double *state_ptr : cur_state_ptrs) {
      if (connected_state_ptrs_set.count(state_ptr) == 0 &&
          base_state_ptrs.count(state_ptr) == 0) {
        connected_state_ptrs_set.emplace(state_ptr);
      }
    }

    // Copy into the connected factors vector
    connected_factors.emplace_back(factor);
  }

  // Copy into output vectors
  for (double *const state_ptr : connected_state_ptrs_set) {
    connected_state_ptrs.emplace_back(state_ptr);
    state_sizes.emplace_back(problem_.ParameterBlockSize(state_ptr));
    local_sizes.emplace_back(problem_.ParameterBlockLocalSize(state_ptr));

    // Get the local parameterization pointer if it exists
    const ceres::LocalParameterization *local_param_ptr =
        problem_.GetParameterization(state_ptr);
    local_param_ptrs.emplace_back(local_param_ptr);
  }
}

bool getMarkovBlanketInfo(const std::vector<StateID> &states_m,
                          std::vector<double *> &connected_state_ptrs,
                          std::vector<ceres::ResidualBlockId> &factors_m,
                          std::vector<ceres::ResidualBlockId> &factors_r,
                          std::vector<StateID> &ConnectedStateIDs);

std::vector<ceres::CostFunction *> FactorGraph::getCostFunctionPtrs() {
  std::vector<ceres::CostFunction *> cost_functions;
  std::vector<ceres::ResidualBlockId> residual_blocks;
  problem_.GetResidualBlocks(&residual_blocks);

  // Extract cost functions from map
  for (auto const &residual_block : residual_blocks) {
    // Check if this is actually in the map
    if (residual_blocks_to_cost_function_map.find(residual_block) ==
        residual_blocks_to_cost_function_map.end()) {
      LOG(ERROR)
          << "Residual block ID not found in the map! Residual block ID: "
          << residual_block;
      return std::vector<ceres::CostFunction *>();
    }
    cost_functions.push_back(
        residual_blocks_to_cost_function_map.at(residual_block));
  }
  return cost_functions;
}

} // namespace ceres_nav