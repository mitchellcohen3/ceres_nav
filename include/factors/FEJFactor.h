#pragma once

#include <ceres/ceres.h>
#include <memory>
#include <vector>

/**
 * @brief Wrapper for Ceres cost functions that enables the use of FEJ 
 * (first estimate Jacobians), by allowing the Jacobians to be evaluated at different
 * linearization points than the current parameter estimates.
 * 
 * To use this class, inherit from it and implement the evaluateResiduals and 
 * evaluateJacobians methods.
 * 
 * To set a fixed linearization point for a parameter block, call 
 * setLinearizationPoint with the parameter block index and the desired
 * linearization point.
 */
template <int kNumResiduals, int... Ns>
class FEJFactor : public ceres::SizedCostFunction<kNumResiduals, Ns...> {
public:
  static constexpr int kNumParameterBlocks = sizeof...(Ns);
  static constexpr int kNumResidualsStatic = kNumResiduals;

  FEJFactor() {
    // Initialize the parameter block sizes array
    parameter_block_sizes_ = {Ns...};

    // Initialize the other vectors
    linearization_points_.resize(kNumParameterBlocks);
    is_linearization_point_set_.resize(kNumParameterBlocks, false);
    use_fixed_linearization_.resize(kNumParameterBlocks, false);
  }
  virtual ~FEJFactor() = default;

  /**
   * @brief For a given parameter block index, set a fixed linearization point.
   */
  bool setLinearizationPoint(int param_idx,
                             const Eigen::VectorXd linearization_point) {
    if (param_idx < 0 || param_idx >= kNumParameterBlocks) {
        LOG(ERROR) << "Parameter index out of range";
        return false;
    }

    // Validate that the linearization point size matches the expected parameter block size
    int expected_size = parameter_block_sizes_[param_idx];
    if (linearization_point.size() != expected_size) {
      LOG(ERROR) << "Linearization point size (" << linearization_point.size()
                 << ") does not match expected parameter block size ("
                 << expected_size << ")";
      return false;
    }

    is_linearization_point_set_[param_idx] = true;
    use_fixed_linearization_[param_idx] = true;
    linearization_points_[param_idx] = linearization_point;
    return true;
  }

  /**
   * @brief For a given parameter block index, get the currently set 
   * linearization point.
   */
  bool getLinearizationPoint(int param_idx,
                             Eigen::VectorXd &linearization_point) const {
    if (param_idx < 0 || param_idx >= kNumParameterBlocks) {
        LOG(ERROR) << "Parameter index out of range";
        return false;
    }

    if (!is_linearization_point_set_[param_idx]) {
      return false;
    }

    linearization_point = linearization_points_[param_idx];
    return true;
  }

  /**
   * @brief See if we've already set a linearization point for a given parameter block.
   */
  bool hasLinearizationPoint(int param_idx) const {
    if (param_idx < 0 || param_idx >= kNumParameterBlocks) {
      throw std::out_of_range("Parameter index out of range");
    }
    return is_linearization_point_set_[param_idx];
  }

  /**
   * @brief Disable the use of a fixed linearization point for a given parameter block.
   * The Jacobians will be evaluated at the current parameter estimate.
   */
  bool resetLinearizationPoint(int param_idx) {
    if (param_idx < 0 || param_idx >= kNumParameterBlocks) {
        LOG(ERROR) << "Parameter index out of range";
        return false;
    }

    is_linearization_point_set_[param_idx] = false;
    use_fixed_linearization_[param_idx] = false;
    linearization_points_[param_idx].resize(0);
    return true;
  }

  /**
   * @brief Wraps the Evaluate function to handle FEJ logic.
   * 
   * We evaluate the residuals at the current parameter estimates,
   * but the Jacobians at either the current estimates or the fixed 
   * estimates, if a linearization point has been set for that
   * parameter block.
   */
  bool Evaluate(double const *const *parameters, double *residuals,
                double **jacobians) const override {
    // Always evaluate the residuals at the current state
    if (!evaluateResiduals(parameters, residuals)) {
      return false;
    }

    if (!jacobians) {
        return true;
    }

    // Evaluate the Jacobians at the correct evaluation points
    std::vector<const double *> eval_points(kNumParameterBlocks);
    for (int i = 0; i < kNumParameterBlocks; i++) {
      if (use_fixed_linearization_[i]) {
        eval_points[i] = linearization_points_[i].data();
      } else {
        eval_points[i] = parameters[i];
      }
    }

    return evaluateJacobians(eval_points.data(), jacobians);
  }

protected:
  /**
   * @brief Implement this function to evaluate the residuals given the
   * parameters.
   */
  virtual bool evaluateResiduals(double const *const *parameters,
                                 double *residuals) const = 0;
  
  /**
   * @brief Implement this function to evaluate the Jacobians for a given set of parameters.
   */
  virtual bool evaluateJacobians(double const *const *parameters,
                                 double **jacobians) const = 0;

  // Linearization point management
  std::vector<Eigen::VectorXd> linearization_points_;
  // Sizes of each parameter block
  std::vector<int> parameter_block_sizes_;
  std::vector<bool> is_linearization_point_set_;
  std::vector<bool> use_fixed_linearization_;
};
