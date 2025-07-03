#pragma once

#include "ParameterBlockBase.h"
#include "StateCollection.h"
#include "StateId.h"
#include <ceres/ceres.h>

/**
 * @brief Main class that contains all the information about an optimization
 * problem. Contains methods for interacting wit the problem (i.e, adding
 * states, etc.)
 * The main difference between this and libRSF::FactorGraph is that
 * to use this, the user already creates ParameterBlock objects rather than
 */

namespace ceres_swf {

class FactorGraph {
public:
  using StatePtr = std::shared_ptr<ParameterBlockBase>;

  // This stores the actual Ceres problem.
  FactorGraph();

  void addState(const std::string &name, double timestamp,
                std::shared_ptr<ParameterBlockBase> state);

  // Add a factor to the problem
  void addFactor(const std::vector<StateID> &state_ids,
                 ceres::CostFunction *cost_function, double stamp,
                 ceres::LossFunction *loss_function = nullptr);

  // Solve the underlying optimization problem
  void solve();
  void solve(ceres::Solver::Options Options);

  /** Get information about the internal Ceres problem. */
  bool getStatePointers(const std::vector<StateID> &StateIDs,
                        std::vector<double *> &state_ptrs);
  bool getConnectedFactorIDs(const std::vector<double *> &StatePointers,
                             std::vector<ceres::ResidualBlockId> &factors);
  bool
  getConnectedStatePointers(const std::vector<ceres::ResidualBlockId> &factors,
                            std::vector<double *> &StatePointers);
  // Simpl calls FactorGraphStructure.getMarginalizationInfo()
  bool getMarkovBlanketInfo(const std::vector<StateID> &States_m,
                            std::vector<double *> &ConnectedStatePtrs,
                            std::vector<ceres::ResidualBlockId> &Factors_m,
                            std::vector<ceres::ResidualBlockId> &Factors_r,
                            std::vector<StateID> &ConnectedStateIDs);

  // Remove states from the problem
  void removeState(const std::string &name, double timestamp);

  // Marginalize states from the problem
  bool marginalizeStates(const std::vector<StateID> &state_ids);

  // Get the cost functions for a set of residual blocks
  bool getCostFunctionPointersForResidualBlocks(
      const std::vector<ceres::ResidualBlockId> &residual_ids,
      std::vector<ceres::CostFunction *> &cost_functions) const;

  /// Getters
  ceres::Problem &getProblem() { return problem_; }
  ceres::Solver::Summary &getSummary() { return summary_; }
  StateCollection &getStates() { return states_; }
  ceres::Solver::Options &getSolverOptions() { return solver_options_; }

  double getLastSolverDuration() { return last_solver_duration; }
  double getTotalSolverDuration() { return total_solver_duration; }
  int getNumSolverIterations() { return num_solver_iterations; }
  int numParameterBlocks() { return problem_.NumParameterBlocks(); }
  int numResidualBlocks() { return problem_.NumResidualBlocks(); }

  /**
   * @brief Computes the covariance of a state with a given name at
   * a particular timestamp.
   */
  bool computeCovariance(const std::string &name, double timestamp);

  /**
   * @brief Gets the marginalization information for a set of states.
   */
  void getMarginalizationInfo(const std::vector<double *> state_ptrs_m,
                              std::vector<double *> &connected_state_ptrs,
                              std::vector<ceres::ResidualBlockId> &factors_m,
                              std::vector<int> &state_sizes,
                              std::vector<int> &local_sizes,
                              std::vector<StateID> &state_ids) const;

protected:
  // The collection of states
  StateCollection states_;

  const ceres::Problem::Options default_problem_options_ = {
      ceres::Ownership::TAKE_OWNERSHIP, // cost_function_ownership
      ceres::Ownership::TAKE_OWNERSHIP, // loss_function_ownership
      ceres::Ownership::TAKE_OWNERSHIP, // local_parameterization_ownership
      true,                             // enable_fast_removal
      false,                            // disable_all_safety_checks
      nullptr,                          // context
      nullptr                           // evaluation_callback
  };

  ceres::Problem problem_;
  ceres::Solver::Summary summary_;
  ceres::Solver::Options solver_options_;

  // A mapping between residual block IDs and the cost function
  std::unordered_map<ceres::ResidualBlockId, ceres::CostFunction *>
      residual_blocks_to_cost_function_map;

  double last_solver_duration = 0.0;
  double total_solver_duration = 0.0;
  int num_solver_iterations = 0;
};

} // namespace ceres_swf