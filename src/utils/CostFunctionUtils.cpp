#include "utils/CostFunctionUtils.h"

namespace ceres_nav {

bool evaluateCostFunction(
    const std::shared_ptr<ceres::CostFunction> &cost_function,
    const std::vector<std::shared_ptr<ParameterBlockBase>> &parameter_blocks,
    Eigen::VectorXd &residuals, std::vector<Eigen::MatrixXd> &jacobians) {

  // Resize our residuals
  residuals.resize(cost_function->num_residuals());
  residuals.setZero();

  // Containers for our Jacobians
  std::vector<
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      jacs;
  std::vector<
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      jacs_min;

  // Create our raw Jacobians
  double **raw_jacobians;
  raw_jacobians = new double *[parameter_blocks.size()];

  // Setup our analytical Jacobians
  std::vector<double *> param_pts;
  for (size_t i = 0; i < parameter_blocks.size(); ++i) {
    // Setup our analytical Jacobians
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> jac;
    jac.resize(cost_function->num_residuals(),
               parameter_blocks[i]->dimension());
    jac.setZero();

    // Add to vectors
    jacs.push_back(jac);
    raw_jacobians[i] = jacs[i].data();

    // Store our parameter pointers
    param_pts.push_back(parameter_blocks[i]->estimatePointer());
  }

  // Evaluate the cost function
  bool success = cost_function->Evaluate(param_pts.data(), residuals.data(),
                                         raw_jacobians);

  // Convert to minimal Jacobians
  for (size_t i = 0; i < parameter_blocks.size(); ++i) {
    // Create minimal Jacobian
    Eigen::MatrixXd min_jac;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        jacobian_temp;
    jacobian_temp.resize(parameter_blocks[i]->dimension(),
                         parameter_blocks[i]->minimalDimension());
    parameter_blocks[i]->plusJacobian(parameter_blocks[i]->estimatePointer(),
                                      jacobian_temp.data());
    min_jac = jacs[i] * jacobian_temp;
    jacobians.push_back(min_jac);
  }

  return success;
}

std::vector<Eigen::MatrixXd> computeNumericalJacobians(
    const std::shared_ptr<ceres::CostFunction> &cost_function,
    const std::vector<std::shared_ptr<ParameterBlockBase>> &parameter_blocks,
    double delta, const NumericalJacobianMethod &method) {
  // Prepare the container for our residual
  Eigen::VectorXd residuals(cost_function->num_residuals());
  residuals.setZero();

  // Perpare our parameter pointers and numerical Jacobians
  std::vector<Eigen::MatrixXd> jacs_num;
  std::vector<double *> parameters;
  for (size_t i = 0; i < parameter_blocks.size(); ++i) {
    parameters.push_back(parameter_blocks[i]->estimatePointer());
    Eigen::MatrixXd jac_num(residuals.size(),
                            parameter_blocks[i]->minimalDimension());
    jac_num.setZero();
    jacs_num.push_back(jac_num);
  }

  // Container for our Jacobians
  if (method == NumericalJacobianMethod::CENTRAL) {
    for (size_t i = 0; i < parameter_blocks.size(); ++i) {
      for (size_t j = 0; j < parameter_blocks[i]->minimalDimension(); ++j) {
        Eigen::VectorXd residuals_p(residuals.size());
        Eigen::VectorXd residuals_m(residuals.size());

        // Apply positive delta
        Eigen::VectorXd parameters_p(parameter_blocks[i]->dimension());
        Eigen::VectorXd parameters_m(parameter_blocks[i]->dimension());
        Eigen::VectorXd delta_xi(parameter_blocks[i]->minimalDimension());
        delta_xi.setZero();
        delta_xi[j] = delta;
        parameter_blocks[i]->plus(parameter_blocks[i]->estimatePointer(),
                                  delta_xi.data(), parameters_p.data());
        // Everwrite our parameters
        parameters[i] = parameters_p.data();
        cost_function->Evaluate(parameters.data(), residuals_p.data(), nullptr);
        // Reset parameters
        parameters[i] = parameter_blocks[i]->estimatePointer();
        // Apply negative delta
        delta_xi.setZero();
        delta_xi[j] = -delta;
        parameter_blocks[i]->plus(parameter_blocks[i]->estimatePointer(),
                                  delta_xi.data(), parameters_m.data());
        parameters[i] = parameters_m.data();
        cost_function->Evaluate(parameters.data(), residuals_m.data(), nullptr);
        // Reset parameters
        parameters[i] = parameter_blocks[i]->estimatePointer();

        // Calculate numeric difference using central difference scheme
        jacs_num[i].col(j) = (residuals_p - residuals_m) * 1.0 / (2.0 * delta);
      }
    }
  }

  return jacs_num;
}

bool checkNumericalJacobians(
    const std::shared_ptr<ceres::CostFunction> &cost_function,
    const std::vector<std::shared_ptr<ParameterBlockBase>> &parameter_blocks,
    const NumericalJacobianMethod &method,
    double delta, bool print_jacobians) {
  // Prepare the container for our residual and Jacobians
  Eigen::VectorXd residuals;
  residuals.resize(cost_function->num_residuals());
  std::vector<Eigen::MatrixXd> jacobians_analytical;

  // Evaluate the cost function and get analytical Jacobians
  bool success_eval = evaluateCostFunction(cost_function, parameter_blocks,
                                           residuals, jacobians_analytical);
  if (!success_eval) {
    return false;
  }

  // Compute the numerical Jacobians
  std::vector<Eigen::MatrixXd> jacobians_numerical =
      computeNumericalJacobians(cost_function, parameter_blocks, delta, method);

  // Ensure that the number of Jacobians matches
  if (jacobians_analytical.size() != jacobians_numerical.size()) {
    std::cerr << "Mismatch in number of analytical and numerical Jacobians."
              << std::endl;
    return false;
  }

  // Check the Jacobians for consistency
  bool is_correct = true;
  for (size_t i = 0; i < jacobians_analytical.size(); ++i) {
    if (!jacobians_analytical[i].isApprox(jacobians_numerical[i], 1e-4)) {
      is_correct = false;
    }

    if (print_jacobians) {
        std::cout << "Analytical Jacobian[" << i << "]: " << std::endl;
        std::cout << jacobians_analytical[i] << std::endl;
        std::cout << "Numerical Jacobian[" << i << "]: " << std::endl;
        std::cout << jacobians_numerical[i] << std::endl; 
    }
  }

  return is_correct;
}

}  // namespace ceres_nav