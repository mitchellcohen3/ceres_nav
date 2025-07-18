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

namespace ceres_nav {

class FactorGraph {
public:
  using StatePtr = std::shared_ptr<ParameterBlockBase>;

  /**
   * @brief Default constructor for the FactorGraph class.
   */
  FactorGraph();

  /**
   * @brief Constructor that allows the user to specify Ceres solver options.
   */
  FactorGraph(ceres::Solver::Options solver_options);

  /**
   * @brief Adds a state to the problem with a particular name and
   * timestamp
   */
  void addState(const std::string &name, double timestamp,
                std::shared_ptr<ParameterBlockBase> state);

  /**
   * @brief Adds a factor to the problem.
   * @param state_ids A vector of StateID objects that the factor is connected
   * to
   * @param  cost_function The Ceres cost function to add
   * @param stamp The timestamp for the factor
   * @param loss_function An optional Ceres loss function for the factor.
   */
  void addFactor(const std::vector<StateID> &state_ids,
                 ceres::CostFunction *cost_function, double stamp,
                 ceres::LossFunction *loss_function = nullptr);

  /**
   * @brief Solves the optimization problem using the current solver options.
   */
  void solve();

  /**
   * @brief Solves the optimization problem using the provided solver options.
   */
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

  /**
   * @brief Sets a state as constant in the optimization problem.
   */
  void setConstant(const std::string &name, double timestamp);

  /**
   * @brief Checks if a state is constant in the optimization problem.
   */
  bool isConstant(const std::string &name, double timestamp);

  /**
   * @brief Sets a state as variable in the optimization problem.
   */
  void setVariable(const std::string &name, double timestamp);

  // Marginalize states from the problem
  bool marginalizeStates(std::vector<StateID> state_ids);

  // Get the cost functions for a set of residual blocks
  bool getCostFunctionPointersForResidualBlocks(
      const std::vector<ceres::ResidualBlockId> &residual_ids,
      std::vector<ceres::CostFunction *> &cost_functions) const;

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
                              std::vector<StateID> &state_ids,
                              std::vector<const ceres::LocalParameterization *>
                                  &local_param_ptrs) const;

  /// Getters
  const ceres::Problem &getProblem() { return problem_; }
  const StateCollection &getStates() { return states_; }
  ceres::Solver::Summary &getSolverSummary() { return summary_; }
  ceres::Solver::Options &getSolverOptions() { return solver_options_; }

  std::vector<ceres::CostFunction *> getCostFunctionPtrs();

  double getLastSolverDuration() { return last_solver_duration; }
  double getTotalSolverDuration() { return total_solver_duration; }
  double getLastMarginalizationDuration() { return marginalization_duration; }

  int getNumSolverIterations() { return num_solver_iterations; }
  int numParameterBlocks() { return problem_.NumParameterBlocks(); }
  int numResidualBlocks() { return problem_.NumResidualBlocks(); }

  void setSolverOptions(const ceres::Solver::Options &options) {
    solver_options_ = options;
  }

  std::map<std::string, double> getMarginalizationTimingStats() const {
    return marginalization_timing_stats_;
  }

  /**
   * @brief Evaluates the full Jacobian of the problem and returns the
   * result as an Eigen matrix.
   */
  Eigen::MatrixXd evaluateJacobian(bool include_fixed_parameters = false);

  /**
   * @brief Evaluate the Jacobian for a specific set of state IDs.
   * Here, the ordering of the state IDs is important, as it determines the
   * ordering of the columns in the Jacobian.
   */
  Eigen::MatrixXd evaluateJacobian(const std::vector<StateID> &state_ids);

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

  // A mapping between residual block IDs and the cost function pointer
  std::unordered_map<ceres::ResidualBlockId, ceres::CostFunction *>
      residual_blocks_to_cost_function_map;

  // Store info about the last computation times
  double last_solver_duration = 0.0;
  double total_solver_duration = 0.0;
  int num_solver_iterations = 0;
  double marginalization_duration = 0.0;

  std::map<std::string, double> marginalization_timing_stats_;
};

} // namespace ceres_nav