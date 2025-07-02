/**
 * This file contains unit tests verifying the Jacobians of the factors
 * implemented in the library. Jacobians are checked against numerical Jacobians 
 * for both left and right perturbations of Lie group states.
 */

#include <catch2/catch_test_macros.hpp>
#include <iostream>

#include "lib/ParameterBlock.h"
#include "lib/ParameterBlockBase.h"
#include "lib/PoseParameterBlock.h"

#include "factors/AbsolutePositionFactor.h"
#include "factors/RelativeLandmarkFactor.h"
#include "factors/RelativePoseFactor.h"

#include "utils/CostFunctionUtils.h"

TEST_CASE("Test AbsolutePositionFactor Jacobians") {
  // Create an absolute position factor
  Eigen::Matrix3d attitude = SO3::expMap(Eigen::Vector3d(0.5, -0.2, 0.3));
  Eigen::Vector3d position(2.0, 2.0, 3.0);
  Eigen::Vector3d measurement =
      position + Eigen::Vector3d(0.1, 0.2, 0.3); // Simulated measurement
  Eigen::Matrix3d sqrt_info = Eigen::Matrix3d::Identity();

  std::vector<LieDirection> directions = {LieDirection::left,
                                          LieDirection::right};

  for (auto const &direction : directions) {
    std::cout << "Testing AbsolutePositionFactor with direction: "
              << (direction == LieDirection::left ? "left" : "right")
              << std::endl;

    std::shared_ptr<AbsolutePositionFactor> factor =
        std::make_shared<AbsolutePositionFactor>(measurement, direction,
                                                 sqrt_info, "SE3", false);
    REQUIRE(factor->num_residuals() == 3);

    // Evaluate the factor with a pose parameter block
    std::shared_ptr<PoseParameterBlock> pose_block =
        std::make_shared<PoseParameterBlock>("test_pose", direction);
    pose_block->setFromAttitudeAndPosition(attitude, position);

    std::vector<std::shared_ptr<ParameterBlockBase>> parameter_blocks;
    parameter_blocks.push_back(pose_block);

    Eigen::VectorXd residuals(factor->num_residuals());
    std::vector<Eigen::MatrixXd> jacobians;

    bool success = ceres_swf::evaluateCostFunction(factor, parameter_blocks,
                                                   residuals, jacobians);
    REQUIRE(success);
    // Check the residuals
    Eigen::Vector3d expected_residual = measurement - position;
    REQUIRE(residuals.isApprox(expected_residual, 1e-6));

    // Try numerically evaluating the Jacobians
    std::vector<Eigen::MatrixXd> numerical_jacobians =
        ceres_swf::computeNumericalJacobians(
            factor, parameter_blocks, 1e-6,
            ceres_swf::NumericalJacobianMethod::CENTRAL);
    REQUIRE(numerical_jacobians.size() == jacobians.size());

    // Check the Jacobians numerically
    bool is_correct = ceres_swf::checkNumericalJacobians(
        factor, parameter_blocks, ceres_swf::NumericalJacobianMethod::CENTRAL,
        1e-6, false);
    REQUIRE(is_correct);
  }
}

TEST_CASE("Test RelativeLandmarkFactorJacobians") {
  Eigen::Matrix3d attitude = SO3::expMap(Eigen::Vector3d(0.5, -0.2, 0.3));
  Eigen::Vector3d position(2.0, 2.0, 3.0);

  Eigen::Vector3d landmark_position = Eigen::Vector3d(1.0, 1.0, 1.0);

  Eigen::Vector3d meas = attitude.transpose() * (landmark_position - position);
  Eigen::Matrix3d sqrt_info = Eigen::Matrix3d::Identity();

  std::vector<LieDirection> directions = {LieDirection::left,
                                          LieDirection::right};
  for (auto const &direction : directions) {
    std::cout << "Testing RelativeLandmarkFactor with direction: "
              << (direction == LieDirection::left ? "left" : "right")
              << std::endl;
    std::shared_ptr<RelativeLandmarkFactor> factor =
        std::make_shared<RelativeLandmarkFactor>(meas, sqrt_info, 0.0, 1,
                                                 direction, "SE3");
    // Create parameter blocks to evaluate this factor at
    std::shared_ptr<PoseParameterBlock> pose_block =
        std::make_shared<PoseParameterBlock>("test_pose", direction);
    pose_block->setFromAttitudeAndPosition(attitude, position);
    std::shared_ptr<ParameterBlock<3>> landmark_block =
        std::make_shared<ParameterBlock<3>>("landmark_1");
    landmark_block->setEstimate(landmark_position);

    std::vector<std::shared_ptr<ParameterBlockBase>> parameter_blocks;
    parameter_blocks.push_back(pose_block);
    parameter_blocks.push_back(landmark_block);

    // Check the Jacobians numerically
    bool is_correct = ceres_swf::checkNumericalJacobians(
        factor, parameter_blocks, ceres_swf::NumericalJacobianMethod::CENTRAL,
        1e-6, false);
    REQUIRE(is_correct);
  }
}

TEST_CASE("Test RelativePoseFactor Jacobians") {
  Eigen::Matrix4d T_i = SE3::expMap(Eigen::VectorXd::Random(6));
  Eigen::Matrix4d T_j = SE3::expMap(Eigen::VectorXd::Random(6));
  Eigen::Matrix4d relative_pose_meas =
      T_i.inverse() * T_j * SE3::expMap(Eigen::VectorXd::Random(6));

  std::vector<LieDirection> directions = {LieDirection::left,
                                          LieDirection::right};

  for (auto const &direction : directions) {
    std::cout << "Testing RelativePoseFactor with direction: "
              << (direction == LieDirection::left ? "left" : "right")
              << std::endl;
    Eigen::Matrix<double, 6, 6> sqrt_info =
        Eigen::Matrix<double, 6, 6>::Identity();

    std::shared_ptr<RelativePoseFactor> factor =
        std::make_shared<RelativePoseFactor>(relative_pose_meas, sqrt_info,
                                             direction);
    // Create the parameter blocks to evaluate the factor at
    std::shared_ptr<PoseParameterBlock> pose_block_i =
        std::make_shared<PoseParameterBlock>("pose_i", direction);  
    pose_block_i->setFromPose(T_i);
    std::shared_ptr<PoseParameterBlock> pose_block_j =
        std::make_shared<PoseParameterBlock>("pose_j", direction);
    pose_block_j->setFromPose(T_j);

    std::vector<std::shared_ptr<ParameterBlockBase>> parameter_blocks;
    parameter_blocks.push_back(pose_block_i);
    parameter_blocks.push_back(pose_block_j);

    // Check the Jacobians numerically
    bool is_correct = ceres_swf::checkNumericalJacobians(
        factor, parameter_blocks, ceres_swf::NumericalJacobianMethod::CENTRAL,
        1e-6, false);    
    REQUIRE(is_correct);
  }
}