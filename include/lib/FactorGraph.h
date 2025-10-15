#pragma once

#include "ParameterBlockBase.h"
#include "StateCollection.h"
#include "StateId.h"
#include "factors/MarginalizationPrior.h"
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
   * @brief Adds a state to the problem with a particular StateID
   */
  void addState(const StateID &state_id,
                std::shared_ptr<ParameterBlockBase> state);

  /**
   * @brief Adds a factor to the problem.
   * @param state_ids A vector of StateID objects that the factor is connected
   * to
   * @param  cost_function The Ceres cost function to add
   * @param stamp The timestamp for the factor
   * @param loss_function An optional Ceres loss function for the factor.
   */
  bool addFactor(const std::vector<StateID> &state_ids,
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

  /**
   * @brief Gets the pointers to the states corresponding to the provided state
   * IDs.
   * @param StateIDs The state IDs to get the pointers for.
   * @param state_ptrs Output vector of state pointers.
   */
  bool getStatePointers(const std::vector<StateID> &StateIDs,
                        std::vector<double *> &state_ptrs) const;

  /**
   * @brief Gets the information about the connected states and factors
   * to the states in states_m, states that we'd like to marginalize out.
   *
   * @param states_m The states to get the Markov blanket information for.
   * @param Output vector containing ParameterBlockInfo objects for the states
   * connected to states_m via factors
   * @param factors_m Output vector of residual block IDs for the factors
   * connected to the states to be marginalized out.
   * @param factors_r Output vector of residual block IDs for the factors
   * involved with the connected states, that are not in factors_m.
   */
  bool
  getMarkovBlanketInfo(const std::vector<StateID> &states_m,
                       std::vector<ParameterBlockInfo> &connected_states,
                       std::vector<ceres::ResidualBlockId> &factors_m,
                       std::vector<ceres::ResidualBlockId> &factors_r) const;

  /**
   * @brief Removes a timestamped state from the problem.
   */
  void removeState(const std::string &name, double timestamp);

  /**
   * @brief Removes a state from the problem given a StateID.
   */
  void removeState(const StateID &state_id);

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

  /**
   * @brief Marginalizes out a set of states from the problem
   */
  bool marginalizeStates(std::vector<StateID> state_ids);

  /**
   * @brief A version of marginalizeStates that also allows the user to
   * specify linearization points to evaluate the marginalization at for each
   * state ID. For a given state, if no linearization point is provided, the
   * current estimate is used.
   */
  bool marginalizeStates(
      std::vector<StateID> states_m,
      const std::map<StateID, Eigen::VectorXd> &linearization_points);

  /**
   * @brief Computes the covariance of a state with a given name at
   * a particular timestamp.
   */
  bool computeCovariance(const std::string &name, double timestamp);

  /**
   * @brief For a given ceres residual block, gets the state IDs that it is a
   * function of
   */
  bool getStateIDsForResidualBlock(const ceres::ResidualBlockId &residual_id,
                                   std::vector<StateID> &state_ids) const;

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

  /**
   * @brief Retrieves the cost function for a given residual block ID.
   */
  ceres::CostFunction *
  getCostFunction(const ceres::ResidualBlockId &residual_id) const;

  // Marginalization information
  struct LastMarginalizationInfo {
  public:
    std::vector<StateID> marginalized_state_ids;
    std::vector<ParameterBlockInfo> connected_states_info;
    std::vector<ceres::ResidualBlockId> factors_m;
    std::vector<ceres::ResidualBlockId> factors_r;
    MarginalizationPrior *last_marginalization_factor = nullptr;

    void print() {
      LOG(INFO) << "Last Marginalization Info:";
      LOG(INFO) << "Number of marginalized states: "
                << marginalized_state_ids.size();
      LOG(INFO) << "Number of connected states: "
                << connected_states_info.size();
      LOG(INFO) << "Number of factors marginalized (factors_m): "
                << factors_m.size();
      LOG(INFO) << "Number of remaining factors (factors_r): "
                << factors_r.size();

      // LOG(INFO) << "Marginalized States:";
      // for (const auto &state_id : marginalized_state_ids) {
      //   if (state_id.isStatic()) {
      //     LOG(INFO) << "  - " << state_id.ID << " (static)";
      //   } else {
      //     LOG(INFO) << "  - " << state_id.ID
      //               << " at timestamp: " << state_id.timestamp.value();
      //   }
      // }
    }
  };

  LastMarginalizationInfo getLastMarginalizationInfo() const {
    return last_marginalization_info_;
  }

protected:
  /**
   * @brief Given a vector of state pointers, finds all the factors that are
   * connected to those states, and returns their residual block IDs.
   */
  bool
  getConnectedFactorIDs(const std::vector<double *> &StatePointers,
                        std::vector<ceres::ResidualBlockId> &factors) const;

  /**
   * @brief Given a vector of factor residual block IDs, finds all the states
   * that are connected.
   */
  bool
  getConnectedStatePointers(const std::vector<ceres::ResidualBlockId> &factors,
                            std::vector<double *> &StatePointers);

  /**
   * @brief Given a vector of residual block IDs, gets the corresponding cost
   * function pointers.
   */
  bool getCostFunctionPointersForResidualBlocks(
      const std::vector<ceres::ResidualBlockId> &residual_ids,
      std::vector<ceres::CostFunction *> &cost_functions) const;

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

  // Store information about the last marginalization
  LastMarginalizationInfo last_marginalization_info_;
};

} // namespace ceres_nav