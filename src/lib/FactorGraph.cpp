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

void FactorGraph::addState(const StateID &state_id,
                           std::shared_ptr<ParameterBlockBase> state) {
  if (state_id.isStatic()) {
    states_.addStaticState(state_id.ID, state);
  } else {
    states_.addState(state_id.ID, state_id.timestamp.value(), state);
  }

  // Common parameter block addition logic
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
bool FactorGraph::addFactor(const std::vector<StateID> &state_ids,
                            ceres::CostFunction *cost_function, double stamp,
                            ceres::LossFunction *loss_function) {
  // Build vector of state pointers
  std::vector<double *> state_ptrs;
  for (auto &state_id : state_ids) {
    std::shared_ptr<ParameterBlockBase> state;
    // For static states, we don't have a timestamp
    if (state_id.isStatic()) {
      state = states_.getStaticState(state_id.ID);
      if (!state) {
        LOG(ERROR)
            << "Trying to add a factor with static state that does not exist: "
            << state_id.ID;
        return false;
      }
    } else {
      // For timestamped states
      state = states_.getState(state_id.ID, state_id.timestamp.value());
      if (!state) {
        LOG(ERROR) << "Trying to add a factor with state that does not exist: "
                   << state_id.ID
                   << " at timestamp: " << state_id.timestamp.value();
        return false;
      }
    }

    state_ptrs.push_back(state->estimatePointer());
  }

  // Add the residual block to the problem
  ceres::ResidualBlockId residual_id =
      problem_.AddResidualBlock(cost_function, loss_function, state_ptrs);

  // Add to map
  residual_blocks_to_cost_function_map.insert({residual_id, cost_function});
  return true;
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
                                   std::vector<double *> &state_ptrs) const {
  // Get the pointers to the states
  for (auto &state_id : state_ids) {
    std::shared_ptr<ParameterBlockBase> state;

    if (state_id.isStatic()) {
      state = states_.getStaticState(state_id.ID);
      if (!state) {
        LOG(ERROR) << "State not found in collection: " << state_id.ID;
        return false;
      }
    } else {
      state = states_.getState(state_id.ID, state_id.timestamp.value());
      if (!state) {
        LOG(ERROR) << "State not found in collection: " << state_id.ID
                   << " at timestamp: " << state_id.timestamp.value();
        return false;
      }
    }

    double *state_ptr = state->estimatePointer();

    if (!problem_.HasParameterBlock(state_ptr)) {
      LOG(ERROR) << "State not found in Ceres problem: " << state_id.ID
                 << " at timestamp: " << state_id.timestamp.value();
      return false;
    }
    state_ptrs.push_back(state_ptr);
  }

  return true;
}

bool FactorGraph::getConnectedFactorIDs(
    const std::vector<double *> &state_ptrs,
    std::vector<ceres::ResidualBlockId> &factors) const {

  // Loop through the state pointers and get all
  // connected residuals
  for (double *const state_ptr : state_ptrs) {
    if (!problem_.HasParameterBlock(state_ptr)) {
      LOG(ERROR) << "State pointer not found in Ceres problem.";
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

  if (!states_.hasState(name, timestamp)) {
    LOG(ERROR) << "State not found in collection: " << name
               << " at timestamp: " << timestamp;
    return;
  }

  // Remove relevant residuals from the cost function map
  double *state_ptr = states_.getState(name, timestamp)->estimatePointer();
  std::vector<ceres::ResidualBlockId> residual_ids;
  problem_.GetResidualBlocksForParameterBlock(state_ptr, &residual_ids);

  for (auto const &residual : residual_ids) {
    if (residual_blocks_to_cost_function_map.find(residual) !=
        residual_blocks_to_cost_function_map.end()) {
      residual_blocks_to_cost_function_map.erase(residual);
    } else {
      LOG(WARNING) << "Residual block not found in map: " << residual;
    }
  }

  // Remove from the Ceres problem
  problem_.RemoveParameterBlock(
      states_.getState(name, timestamp)->estimatePointer());

  // Remove from the StateCollection
  states_.removeState(name, timestamp);
}

void FactorGraph::removeState(const StateID &state_id) {
  std::shared_ptr<ParameterBlockBase> state;

  if (state_id.isStatic()) {
    state = states_.getStaticState(state_id.ID);
  } else {
    state = states_.getState(state_id.ID, state_id.timestamp.value());
  }

  if (!state) {
    LOG(ERROR) << "State not found in collection: " << state_id.ID;
    if (!state_id.isStatic()) {
      LOG(ERROR) << " at timestamp: " << state_id.timestamp.value();
    }
    return;
  }

  // Remove relevant residuals from the cost function map
  std::vector<ceres::ResidualBlockId> residual_ids;
  problem_.GetResidualBlocksForParameterBlock(state->estimatePointer(),
                                              &residual_ids);
  for (auto const &residual : residual_ids) {
    if (residual_blocks_to_cost_function_map.find(residual) !=
        residual_blocks_to_cost_function_map.end()) {
      residual_blocks_to_cost_function_map.erase(residual);
    } else {
      LOG(WARNING) << "Residual block not found in map: " << residual;
    }
  }

  // Remove from the Ceres problem
  problem_.RemoveParameterBlock(state->estimatePointer());

  // Remove from the StateCollection
  if (state_id.isStatic()) {
    states_.removeStaticState(state_id.ID);
  } else {
    states_.removeState(state_id.ID, state_id.timestamp.value());
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

bool FactorGraph::isConstant(const std::string &name, double timestamp) {
  if (states_.hasState(name, timestamp)) {
    return problem_.IsParameterBlockConstant(
        states_.getState(name, timestamp)->estimatePointer());
  } else {
    LOG(ERROR) << "State not found in collection: " << name
               << " at timestamp: " << timestamp;
    return false;
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

  // Reset the last marginalization info
  // last_marginalization_info_.reset();
  last_marginalization_info_.marginalized_state_ids = states_m;

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
    LOG(ERROR) << "Failed to get state pointers for marginalization.";
    return false;
  }

  // Compute the total degrees of freedom of the state to be marginalized
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
  // LOG(INFO) << "Getting marginalization info for states: ";
  getMarginalizationInfo(state_ptrs_m, connected_state_ptrs, connected_factors,
                         global_size, local_size, state_ids, local_param_ptrs);
  double info_duration = get_info_timer.toc() * 1e-3;
  marginalization_timing_stats_["marginalization_info_duration"] =
      info_duration;

  // LOG(INFO) << "Done!" << " Marginalization info duration: "
  //           << info_duration << " seconds.";
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
    // LOG(INFO) << "Evaluating marginalization sub-problem with "
    //           << combined_states.size() << " connected states and "
    //           << connected_factors.size() << " factors.";
    // Evaluate the problem
    problem_.Evaluate(options, nullptr, &ResidualVec, nullptr, &JacobianCRS);
    marg_eval_timer.toc();
    // LOG(INFO) << "Done! Marginalization evaluation duration: "
    //           << marg_eval_timer.toc() * 1e-3 << " seconds.";
    double marg_eval_duration = marg_eval_timer.toc() * 1e-3;
    marginalization_timing_stats_["marginalization_evaluation_duration"] =
        marg_eval_duration;

    // LOG(INFO) << "Logging error to vector...";
    // Map the error to a vector
    ceres_nav::Vector residual_vec =
        Eigen::Map<ceres_nav::Vector>(ResidualVec.data(), ResidualVec.size());
    // LOG(INFO) << "Done! Error vector size: " << residual_vec.size();
    // Convert the Jacobian to an Eigen Matrix
    ceres_nav::Matrix jacobian;
    ceres_nav::CRSToMatrix(JacobianCRS, jacobian);

    // LOG(INFO) << "Performing schur complement...";
    Timer schur_compl_timer;
    schur_compl_timer.tic();
    // Compute the marginalziation
    ceres_nav::Matrix jacobian_marg;
    ceres_nav::Vector residual_marg;
    ceres_nav::Marginalize(residual_vec, jacobian, residual_marg, jacobian_marg,
                           marginal_size, 1.0);
    schur_compl_timer.toc();
    // LOG(INFO) << "Done! Schur complement duration: "
    //           << schur_compl_timer.toc() * 1e-3 << " seconds.";
    double schur_compl_duration = schur_compl_timer.toc() * 1e-3;
    marginalization_timing_stats_["marginalization_schur_complement_duration"] =
        schur_compl_duration;

    // Store a copy of the original states
    std::vector<Eigen::VectorXd> original_states;
    for (int n = 0; n < connected_state_ptrs.size(); n++) {
      Eigen::Map<Eigen::VectorXd> state_vec(connected_state_ptrs.at(n),
                                            global_size.at(n));
      Eigen::VectorXd state_copy = state_vec;
      original_states.emplace_back(state_copy);
    }

    // Construct the marginal prior
    std::vector<ParameterBlockInfo> parameter_blocks_prior;
    for (int n = 0; n < connected_state_ptrs.size(); n++) {
      ParameterBlockInfo param_block_info;
      param_block_info.param_ptr =
          states_.getStateByEstimatePointer(connected_state_ptrs.at(n));
      param_block_info.linearization_point = original_states.at(n);
      param_block_info.state_id = state_ids.at(n);
      parameter_blocks_prior.emplace_back(param_block_info);
    }

    ceres_nav::MarginalizationPrior *marg_prior = new MarginalizationPrior(
        parameter_blocks_prior, jacobian_marg, residual_marg);

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
  for (const StateID &state_id : states_m) {
    removeState(state_id);
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
    std::vector<StateID> &connected_state_ids,
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

  // Now that we have all the connected factors, get all states involed in
  // those factors
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
    StateID state_id;
    if (states_.getStateIDByEstimatePointer(state_ptr, state_id)) {
      connected_state_ids.emplace_back(state_id);
    } else {
      LOG(ERROR) << "Failed to get StateID for state pointer: " << state_ptr;
    }
    // Get the local parameterization pointer if it exists
    const ceres::LocalParameterization *local_param_ptr =
        problem_.GetParameterization(state_ptr);
    local_param_ptrs.emplace_back(local_param_ptr);
  }
}

bool FactorGraph::getMarkovBlanketInfo(
    const std::vector<StateID> &states_m,
    std::vector<ParameterBlockInfo> &connected_states,
    std::vector<ceres::ResidualBlockId> &factors_m,
    std::vector<ceres::ResidualBlockId> &factors_r) const {
  // Get the state pointers for the states in states_m
  std::vector<double *> state_ptrs_m;
  if (!getStatePointers(states_m, state_ptrs_m)) {
    LOG(ERROR) << "Failed to get state pointers for Markov blanket info.";
    return false;
  }

  // Get the connected states and factors
  std::vector<int> global_sizes;
  std::vector<int> local_sizes;
  std::vector<const ceres::LocalParameterization *> local_param_ptrs;

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

  // Now that we have all the connected factors, get all states involed in
  // those factors
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
    StateID state_id;
    if (states_.getStateIDByEstimatePointer(state_ptr, state_id)) {
      connected_state_ids.emplace_back(state_id);
    } else {
      LOG(ERROR) << "Failed to get StateID for state pointer: " << state_ptr;
    }
    // Get the local parameterization pointer if it exists
    const ceres::LocalParameterization *local_param_ptr =
        problem_.GetParameterization(state_ptr);
    local_param_ptrs.emplace_back(local_param_ptr);
  }
  getMarginalizationInfo(state_ptrs_m, connected_state_ptrs, factors_m,
                         global_sizes, local_sizes, connected_state_ids,
                         local_param_ptrs);

  // Next, get any factors that are connected to states in
  // connected_state_ptrs
  std::vector<ceres::ResidualBlockId> connected_factors;
  if (!getConnectedFactorIDs(connected_state_ptrs, connected_factors)) {
    LOG(ERROR) << "Failed to get connected factor IDs for Markov blanket info.";
    return false;
  }

  // Now, we need to find the factors that are not in factors_m
  for (const ceres::ResidualBlockId &factor : connected_factors) {
    if (std::find(factors_m.begin(), factors_m.end(), factor) ==
        factors_m.end()) {
      // This factor is not in factors_m, so add it to factors_r
      factors_r.push_back(factor);
    }
  }
  return true;
}

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

ceres::CostFunction *
FactorGraph::getCostFunction(const ceres::ResidualBlockId &residual_id) const {
  auto it = residual_blocks_to_cost_function_map.find(residual_id);
  if (it != residual_blocks_to_cost_function_map.end()) {
    return it->second;
  }
  LOG(ERROR) << "Residual block ID not found in the map: " << residual_id;
  return nullptr;
}

Eigen::MatrixXd FactorGraph::evaluateJacobian(bool include_fixed_parameters) {
  std::vector<double> residuals;
  std::vector<double *> all_param_blocks;
  ceres::CRSMatrix jacobian_crs;
  problem_.GetParameterBlocks(&all_param_blocks);

  std::vector<double *> parameter_blocks;
  for (double *block : all_param_blocks) {
    // Skip constant blocks if not including fixed parameters
    if (problem_.IsParameterBlockConstant(block) && !include_fixed_parameters) {
      continue;
    }
    parameter_blocks.push_back(block);
  }

  ceres::Problem::EvaluateOptions options;
  options.apply_loss_function = true;
  options.num_threads = static_cast<int>(std::thread::hardware_concurrency());
  options.parameter_blocks = parameter_blocks;

  problem_.Evaluate(options, nullptr, &residuals, nullptr, &jacobian_crs);

  // Convert the CRS matrix to an Eigen matrix
  ceres_nav::Matrix jacobian;
  ceres_nav::CRSToMatrix(jacobian_crs, jacobian);

  return jacobian;
}

Eigen::MatrixXd
FactorGraph::evaluateJacobian(const std::vector<StateID> &states) {
  std::vector<double *> state_ptrs;
  if (!getStatePointers(states, state_ptrs)) {
    LOG(ERROR) << "Failed to get state pointers for Jacobian evaluation.";
    return Eigen::MatrixXd();
  }

  std::vector<double> residuals;
  ceres::CRSMatrix jacobian_crs;

  ceres::Problem::EvaluateOptions options;
  options.apply_loss_function = true;
  options.num_threads = static_cast<int>(std::thread::hardware_concurrency());
  options.parameter_blocks = state_ptrs;
  // options.residual_blocks =

  problem_.Evaluate(options, nullptr, &residuals, nullptr, &jacobian_crs);

  // Convert the CRS matrix to an Eigen matrix
  ceres_nav::Matrix jacobian;
  ceres_nav::CRSToMatrix(jacobian_crs, jacobian);

  return jacobian;
}

} // namespace ceres_nav