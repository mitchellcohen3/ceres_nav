#include <catch2/catch_test_macros.hpp>

#include <Eigen/Dense>

#include "utils/CostFunctionUtils.h"
#include "factors/FEJFactor.h"

// A simple implementation of FEJFactor for testing purposes
// The dimension of the residual is 2, with two parameter blocks of sizes 3 and 4
class TestFEJFactor : public FEJFactor<2, 3, 4> {
public:
  // Track how many times each method is called and with what parameters
  mutable std::vector<Eigen::VectorXd> residual_evaluation_points_;
  mutable std::vector<Eigen::VectorXd> jacobian_evaluation_points_;
  mutable int residual_call_count_ = 0;
  mutable int jacobian_call_count_ = 0;

protected:
  bool evaluateResiduals(double const *const *parameters,
                         double *residuals) const override {
    residual_call_count_++;
    // Store the evaluation points for verification
    residual_evaluation_points_.clear();

    residual_evaluation_points_.push_back(
        Eigen::VectorXd::Map(parameters[0], 3));
    residual_evaluation_points_.push_back(
        Eigen::VectorXd::Map(parameters[1], 4));

    // Simple residual: f(x,y) = [x[0] + y[0], x[1] + y[1]]
    residuals[0] = parameters[0][0] + parameters[1][0];
    residuals[1] = parameters[0][1] + parameters[1][1];
    return true;
  }

  bool evaluateJacobians(double const *const *parameters,
                         double **jacobians) const override {

    jacobian_call_count_++;

    // Store the evaluation points for verification
    jacobian_evaluation_points_.clear();
    jacobian_evaluation_points_.push_back(
        Eigen::VectorXd::Map(parameters[0], 3));
    jacobian_evaluation_points_.push_back(
        Eigen::VectorXd::Map(parameters[1], 4));

    if (jacobians[0] != nullptr) {
      // Jacobian w.r.t. first parameter block (3x2 matrix, stored row-major)
      jacobians[0][0] = 1.0;
      jacobians[0][1] = 0.0;
      jacobians[0][2] = 0.0;
      jacobians[0][3] = 0.0;
      jacobians[0][4] = 1.0;
      jacobians[0][5] = 0.0;
    }

    if (jacobians[1] != nullptr) {
      // Jacobian w.r.t. second parameter block (4x2 matrix, stored row-major)
      jacobians[1][0] = 1.0;
      jacobians[1][1] = 0.0;
      jacobians[1][2] = 0.0;
      jacobians[1][3] = 0.0;
      jacobians[1][4] = 0.0;
      jacobians[1][5] = 1.0;
      jacobians[1][6] = 0.0;
      jacobians[1][7] = 0.0;
    }

    return true;
  }
};

TEST_CASE("FEJCostFunction Basic Functionality", "[FEJCostFunction]") {
  TestFEJFactor factor;

  SECTION("Constructor initializes correctly") {
    REQUIRE(factor.kNumParameterBlocks == 2);
    REQUIRE(factor.kNumResidualsStatic == 2);
  }

  SECTION("Set and get linearization points") {
    // Set two linearization points
    Eigen::Vector3d lin_point_0(1.0, 2.0, 3.0);
    Eigen::Vector4d lin_point_1(4.0, 5.0, 6.0, 7.0);
    factor.setLinearizationPoint(0, lin_point_0);
    factor.setLinearizationPoint(1, lin_point_1);

    Eigen::VectorXd retrieved_point;
    REQUIRE(factor.getLinearizationPoint(0, retrieved_point));
    REQUIRE(retrieved_point.isApprox(lin_point_0));

    REQUIRE(factor.getLinearizationPoint(1, retrieved_point));
    REQUIRE(retrieved_point.isApprox(lin_point_1));
  }

  SECTION("Error handling for invalid parameter indices") {
    Eigen::Vector3d lin_point(1.0, 2.0, 3.0);

    REQUIRE(!factor.setLinearizationPoint(-1, lin_point));
    REQUIRE(!factor.setLinearizationPoint(2, lin_point));

    Eigen::VectorXd retrieved_point;
    REQUIRE(!factor.getLinearizationPoint(-1, retrieved_point));
    REQUIRE(!factor.getLinearizationPoint(2, retrieved_point));
  }

  SECTION("Get linearization point returns false when not set") {
    Eigen::VectorXd retrieved_point;
    REQUIRE(!factor.getLinearizationPoint(0, retrieved_point));
    REQUIRE(!factor.getLinearizationPoint(1, retrieved_point));
  }

  SECTION("Wrong size linearization point should return false") {
    Eigen::Vector2d wrong_size_point(1.0, 2.0);
    REQUIRE(!factor.setLinearizationPoint(0, wrong_size_point));
  }
}

TEST_CASE("FEJCostFunction Evaluate Method", "[FEJCostFunction]") {
  TestFEJFactor factor;

  // Set up test parameters
  std::vector<double> param0 = {1.0, 2.0, 3.0};
  std::vector<double> param1 = {4.0, 5.0, 6.0, 7.0};
  double const *parameters[2] = {param0.data(), param1.data()};

  std::vector<double> residuals(2);

  SECTION("Evaluate residuals only") {
    REQUIRE(factor.Evaluate(parameters, residuals.data(), nullptr));

    // Verify residuals were evaluated at current parameters
    REQUIRE(factor.residual_call_count_ == 1);
    REQUIRE(factor.jacobian_call_count_ == 0);
  }

  SECTION("Evaluate residuals and Jacobians without FEJ") {
    std::vector<double> jac0(6), jac1(8);
    double *jacobians[2] = {jac0.data(), jac1.data()};

    REQUIRE(factor.Evaluate(parameters, residuals.data(), jacobians));

    // Verify both residuals and Jacobians were computed
    REQUIRE(factor.residual_call_count_ == 1);
    REQUIRE(factor.jacobian_call_count_ == 1);

    // Verify Jacobians were evaluated at current parameters (same as residuals)
    REQUIRE(factor.jacobian_evaluation_points_[0] ==
            factor.residual_evaluation_points_[0]);
    REQUIRE(factor.jacobian_evaluation_points_[1] ==
            factor.residual_evaluation_points_[1]);
  }

  SECTION("Evaluate residuals and Jacobians with FEJ") {
    // Set linearization points
    Eigen::Vector3d lin_point_0(10.0, 20.0, 30.0);
    Eigen::Vector4d lin_point_1(40.0, 50.0, 60.0, 70.0);

    factor.setLinearizationPoint(0, lin_point_0);
    factor.setLinearizationPoint(1, lin_point_1);

    std::vector<double> jac0(6), jac1(8);
    double *jacobians[2] = {jac0.data(), jac1.data()};

    REQUIRE(factor.Evaluate(parameters, residuals.data(), jacobians));

    // Verify residuals were evaluated at current parameters
    REQUIRE(factor.residual_evaluation_points_[0].isApprox(Eigen::Vector3d(1.0, 2.0, 3.0)));
    REQUIRE(factor.residual_evaluation_points_[1].isApprox(Eigen::Vector4d(4.0, 5.0, 6.0, 7.0)));

    // Verify Jacobians were evaluated at linearization points
    REQUIRE(factor.jacobian_evaluation_points_[0].isApprox(Eigen::Vector3d(10.0, 20.0, 30.0)));
    REQUIRE(factor.jacobian_evaluation_points_[1].isApprox(Eigen::Vector4d(40.0, 50.0, 60.0, 70.0)));

    std::cout << factor.jacobian_evaluation_points_[0].transpose() << std::endl;
    std::cout << factor.jacobian_evaluation_points_[1].transpose() << std::endl;
  }

  SECTION("Evaluate with partial FEJ (only one parameter has linearization "
          "point)") {
    // Set linearization point only for parameter 0
    Eigen::Vector3d lin_point_0(10.0, 20.0, 30.0);
    factor.setLinearizationPoint(0, lin_point_0);

    std::vector<double> jac0(6), jac1(8);
    double *jacobians[2] = {jac0.data(), jac1.data()};

    REQUIRE(factor.Evaluate(parameters, residuals.data(), jacobians));

    // Verify Jacobians: param 0 at linearization point, param 1 at current
    // point
    REQUIRE(factor.jacobian_evaluation_points_[0].isApprox(Eigen::Vector3d(10.0, 20.0, 30.0)));
    REQUIRE(factor.jacobian_evaluation_points_[1].isApprox(Eigen::Vector4d(4.0, 5.0, 6.0, 7.0)));
  }

  SECTION("Test resetting linearization points") {
    // Set and then reset linearization point for parameter 0
    Eigen::Vector3d lin_point_0(10.0, 20.0, 30.0);
    factor.setLinearizationPoint(0, lin_point_0);
    REQUIRE(factor.hasLinearizationPoint(0));
    factor.resetLinearizationPoint(0);
    REQUIRE(!factor.hasLinearizationPoint(0));

    std::vector<double> jac0(6), jac1(8);
    double *jacobians[2] = {jac0.data(), jac1.data()};

    REQUIRE(factor.Evaluate(parameters, residuals.data(), jacobians));

    // Verify Jacobians - both should be at the current points now
    REQUIRE(factor.jacobian_evaluation_points_[0].isApprox(Eigen::Vector3d(1.0, 2.0, 3.0)));
    REQUIRE(factor.jacobian_evaluation_points_[1].isApprox(Eigen::Vector4d(4.0, 5.0, 6.0, 7.0)));
  }
}