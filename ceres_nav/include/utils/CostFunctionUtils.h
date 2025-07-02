#pragma once

#include <Eigen/Dense>
#include <ceres/ceres.h>
#include <vector>

#include "lib/ParameterBlockBase.h"

namespace ceres_swf {

/**
 * @brief Utility functions for ceres::CostFunction evaluation and Jacobian
 * computation.
 */

// Either a central or forward numerical Jacobian method can be used.
enum class NumericalJacobianMethod { CENTRAL, FORWARD };

/**
 * @brief Evaluate a ceres::CostFunction at a given estimate of the parameter
 * blocks.
 */
bool evaluateCostFunction(
    const std::shared_ptr<ceres::CostFunction> &cost_function,
    const std::vector<std::shared_ptr<ParameterBlockBase>> &parameter_blocks,
    Eigen::VectorXd &residuals, std::vector<Eigen::MatrixXd> &jacobians);

/**
 * @brief Computes numerical Jacobians for a given cost function,
 * evaluated at the current estimates of the parameter blocks.
 */
std::vector<Eigen::MatrixXd> computeNumericalJacobians(
    const std::shared_ptr<ceres::CostFunction> &cost_function,
    const std::vector<std::shared_ptr<ParameterBlockBase>> &parameter_blocks,
    double delta = 1e-6,
    const NumericalJacobianMethod &method = NumericalJacobianMethod::CENTRAL);

bool checkNumericalJacobians(
    const std::shared_ptr<ceres::CostFunction> &cost_function,
    const std::vector<std::shared_ptr<ParameterBlockBase>> &parameter_blocks,
    const NumericalJacobianMethod &method = NumericalJacobianMethod::CENTRAL,
    double delta = 1e-6, bool print_jacobians = false);
} // namespace ceres_swf