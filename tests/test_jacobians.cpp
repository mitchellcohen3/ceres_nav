/**
 * This file contains unit tests verifying the Jacobians of the factors
 * implemented in the library. Jacobians are checked against numerical Jacobians
 * for both left and right perturbations of Lie group states.
 */

#include <catch2/catch_test_macros.hpp>
#include <iostream>

#include "lib/ExtendedPoseParameterBlock.h"
#include "lib/ParameterBlock.h"
#include "lib/ParameterBlockBase.h"
#include "lib/PoseParameterBlock.h"

// Factors to test
#include "factors/AbsolutePositionFactor.h"
#include "factors/IMUPreintegrationFactor.h"
#include "factors/RelativeLandmarkFactor.h"
#include "factors/RelativePoseFactor.h"

#include "imu/IMUIncrement.h"
#include "lie/SE23.h"
#include "lie/SO3.h"

#include "lib/ExtendedPoseParameterBlock.h"
#include "utils/CostFunctionUtils.h"

using namespace ceres_nav;

TEST_CASE("Test AbsolutePositionFactor Jacobians") {
  // Create an absolute position factor
  Eigen::Matrix3d attitude = SO3::expMap(Eigen::Vector3d(0.5, -0.2, 0.3));
  Eigen::Vector3d position(2.0, 2.0, 3.0);
  Eigen::Vector3d measurement =
      position + Eigen::Vector3d(0.1, 0.2, 0.3); // Simulated measurement
  Eigen::Matrix3d sqrt_info = Eigen::Matrix3d::Identity();

  std::vector<LieDirection> directions = {LieDirection::left,
                                          LieDirection::right};

  std::vector<ExtendedPoseRepresentation> pose_types = {
      ExtendedPoseRepresentation::SE23, ExtendedPoseRepresentation::Decoupled};

  // Loop through each combination of pose type and Lie direction
  for (auto const &pose_type : pose_types) {
    for (auto const &direction : directions) {
      std::cout << "Testing AbsolutePositionFactor with direction: "
                << (direction == LieDirection::left ? "left" : "right")
                << std::endl;
      std::cout << "Pose type: "
                << (pose_type == ExtendedPoseRepresentation::SE23 ? "SE23"
                                                                  : "Decoupled")
                << std::endl;
      std::shared_ptr<AbsolutePositionFactor> factor =
          std::make_shared<AbsolutePositionFactor>(measurement, direction,
                                                   sqrt_info, pose_type);
      REQUIRE(factor->num_residuals() == 3);

      // Evaluate the factor with an ExtendedPoseParameterBlock
      std::shared_ptr<ExtendedPoseParameterBlock> pose_block =
          std::make_shared<ExtendedPoseParameterBlock>(pose_type, "test_pose",
                                                       direction);
      pose_block->setFromComponents(attitude, Eigen::Vector3d::Zero(),
                                    position);

      std::vector<std::shared_ptr<ParameterBlockBase>> parameter_blocks;
      parameter_blocks.push_back(pose_block);

      Eigen::VectorXd residuals(factor->num_residuals());
      std::vector<Eigen::MatrixXd> jacobians;

      bool success = ceres_nav::evaluateCostFunction(factor, parameter_blocks,
                                                     residuals, jacobians);
      REQUIRE(success);
      // Check the residuals
      Eigen::Vector3d expected_residual = measurement - position;
      REQUIRE(residuals.isApprox(expected_residual, 1e-6));

      // Try numerically evaluating the Jacobians
      std::vector<Eigen::MatrixXd> numerical_jacobians =
          ceres_nav::computeNumericalJacobians(
              factor, parameter_blocks, 1e-6,
              ceres_nav::NumericalJacobianMethod::CENTRAL);
      REQUIRE(numerical_jacobians.size() == jacobians.size());

      // Check the Jacobians numerically
      std::vector<Eigen::MatrixXd> analytical_jacobians;
      std::vector<Eigen::MatrixXd> numerical_jacs;
      bool is_correct = ceres_nav::checkNumericalJacobians(
          factor, parameter_blocks, analytical_jacobians, numerical_jacs,
          ceres_nav::NumericalJacobianMethod::CENTRAL, 1e-6, false);
      REQUIRE(is_correct);
    }
  }
}

TEST_CASE("RelativeLandmarkFactor") {
  Eigen::Matrix3d attitude = SO3::expMap(Eigen::Vector3d(0.5, -0.2, 0.3));
  Eigen::Vector3d position(2.0, 2.0, 3.0);

  Eigen::Vector3d landmark_position = Eigen::Vector3d(1.0, 1.0, 1.0);

  Eigen::Vector3d meas = attitude.transpose() * (landmark_position - position);
  Eigen::Matrix3d sqrt_info = Eigen::Matrix3d::Identity();

  std::vector<LieDirection> directions = {LieDirection::left,
                                          LieDirection::right};

  std::vector<ExtendedPoseRepresentation> pose_types = {
      ExtendedPoseRepresentation::SE23, ExtendedPoseRepresentation::Decoupled};

  for (auto const &pose_type : pose_types) {
    for (auto const &direction : directions) {
      std::shared_ptr<RelativeLandmarkFactor> factor =
          std::make_shared<RelativeLandmarkFactor>(meas, sqrt_info, 0.0,
                                                   direction, pose_type);
      // Create parameter blocks to evaluate this factor at
      std::shared_ptr<ExtendedPoseParameterBlock> pose_block =
          std::make_shared<ExtendedPoseParameterBlock>(pose_type, "test_pose",
                                                       direction);
      pose_block->setFromComponents(attitude, Eigen::Vector3d::Zero(),
                                    position);
      std::shared_ptr<ParameterBlock<3>> landmark_block =
          std::make_shared<ParameterBlock<3>>("landmark_1");
      landmark_block->setEstimate(landmark_position);

      std::vector<std::shared_ptr<ParameterBlockBase>> parameter_blocks;
      parameter_blocks.push_back(pose_block);
      parameter_blocks.push_back(landmark_block);

      // Check the Jacobians numerically
      std::vector<Eigen::MatrixXd> analytical_jacobians;
      std::vector<Eigen::MatrixXd> numerical_jacobians;
      bool is_correct = ceres_nav::checkNumericalJacobians(
          factor, parameter_blocks, analytical_jacobians, numerical_jacobians,
          ceres_nav::NumericalJacobianMethod::CENTRAL, 1e-6, false);
      REQUIRE(is_correct);
    }
  }
}

TEST_CASE("RelativePoseFactor") {
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
    std::vector<Eigen::MatrixXd> analytical_jacobians;
    std::vector<Eigen::MatrixXd> numerical_jacobians;
    bool is_correct = ceres_nav::checkNumericalJacobians(
        factor, parameter_blocks, analytical_jacobians, numerical_jacobians,
        ceres_nav::NumericalJacobianMethod::CENTRAL, 1e-6, false);
    REQUIRE(is_correct);
  }
}

TEST_CASE("IMUPreintegrationFactor") {
  // Create an IMU increment
  Eigen::Matrix<double, 12, 12> Q_ct =
      Eigen::Matrix<double, 12, 12>::Identity() * 0.01;
  Eigen::Vector3d init_gyro_bias{0.1, 0.2, 0.3};
  Eigen::Vector3d init_accel_bias{0.5, 0.7, 0.4};
  Eigen::Vector3d gravity(0.0, 0.0, -9.81);

  int num_imu_meas = 10;
  bool use_group_jacobians = true;

  Eigen::Matrix3d C_i = SO3::expMap(Eigen::Vector3d(0.1, 0.2, 0.3));
  Eigen::Vector3d v_i = Eigen::Vector3d(0.1, 0.2, 0.3);
  Eigen::Vector3d r_i = Eigen::Vector3d(1.0, 1.0, 1.0);
  Eigen::Matrix<double, 5, 5> X_i = SE23::fromComponents(C_i, v_i, r_i);
  Eigen::Matrix<double, 6, 1> b_i;
  b_i << -0.1, -0.2, -0.3, -0.4, -0.5, -0.6;
  // b_i << 0.1, 0.2, 0.3, 0.5, 0.7, 0.4;

  Eigen::Matrix3d C_j = SO3::expMap(Eigen::Vector3d(0.7, 0.5, 0.3));
  Eigen::Vector3d v_j = Eigen::Vector3d(0.4, 0.6, 0.76);
  Eigen::Vector3d r_j = Eigen::Vector3d(0.5, -1.0, -1.2);
  Eigen::Matrix<double, 5, 5> X_j = SE23::fromComponents(C_i, v_i, r_i);
  Eigen::Matrix<double, 6, 1> b_j;
  b_j << 0.5, 0.6, 0.2, 0.2, 0.1, 0.4;

  // Pose representations to test
  // std::vector<ExtendedPoseRepresentation> rep_types =
  //     {ExtendedPoseRepresentation::SE23,
  //     ExtendedPoseRepresentation::Decoupled};

  std::vector<LieDirection> directions = {LieDirection::left,
                                          LieDirection::right};

  std::vector<ExtendedPoseRepresentation> rep_types = {
      ExtendedPoseRepresentation::SE23};

  for (auto const &rep_type : rep_types) {
    for (auto const &direction : directions) {

      if (rep_type == ExtendedPoseRepresentation::Decoupled &&
          direction == LieDirection::left) {
        LOG(INFO) << "Left Lie direction for decoupled representation not "
                     "supported yet. Skipping test.";
        continue;
      }
      std::shared_ptr<ExtendedPoseParameterBlock> X_i_block =
          std::make_shared<ExtendedPoseParameterBlock>(X_i, rep_type, "X_i",
                                                       direction);
      std::shared_ptr<ParameterBlock<6>> b_i_block =
          std::make_shared<ParameterBlock<6>>(b_i, "b_i");

      std::shared_ptr<ExtendedPoseParameterBlock> X_j_block =
          std::make_shared<ExtendedPoseParameterBlock>(X_j, rep_type, "X_j",
                                                       direction);
      std::shared_ptr<ParameterBlock<6>> b_j_block =
          std::make_shared<ParameterBlock<6>>(b_j, "b_i");

      std::vector<std::shared_ptr<ParameterBlockBase>> parameter_blocks;
      parameter_blocks.push_back(X_i_block);
      parameter_blocks.push_back(b_i_block);
      parameter_blocks.push_back(X_j_block);
      parameter_blocks.push_back(b_j_block);

      // Create an IMU increment and propagate it forward
      LOG(INFO) << "Creating preintegration options...";
      std::shared_ptr<IMUIncrementOptions> preintegration_options = std::make_shared<
          IMUIncrementOptions>();
      preintegration_options->sigma_gyro_ct = 0.01;
      preintegration_options->sigma_accel_ct = 0.1;
      preintegration_options->sigma_gyro_bias_ct = 0.001;
      preintegration_options->sigma_accel_bias_ct = 0.001;
      preintegration_options->gravity = gravity;
      preintegration_options->direction = direction;
      preintegration_options->pose_rep = rep_type;
      IMUIncrement imu_increment(preintegration_options, init_gyro_bias,
                                 init_accel_bias);
      for (int i = 0; i < num_imu_meas; ++i) {
        double dt = 0.01;
        Eigen::Vector3d omega = Eigen::Vector3d::Random();
        Eigen::Vector3d accel = Eigen::Vector3d::Random();
        imu_increment.propagate(dt, omega, accel);
      }

      // Create the factor
      std::shared_ptr<IMUPreintegrationFactor> factor =
          std::make_shared<IMUPreintegrationFactor>(imu_increment,
                                                    use_group_jacobians);

      // Evaluate the factor with the parameter blocks
      std::vector<Eigen::MatrixXd> analytical_jacobians;
      std::vector<Eigen::MatrixXd> numerical_jacobians;
      bool is_correct = ceres_nav::checkNumericalJacobians(
          factor, parameter_blocks, analytical_jacobians, numerical_jacobians,
          ceres_nav::NumericalJacobianMethod::CENTRAL, 1e-6, false);

      for (size_t i = 0; i < analytical_jacobians.size(); ++i) {
        Eigen::MatrixXd difference =
            analytical_jacobians[i] - numerical_jacobians[i];
        double norm = difference.norm();
        // std::cout << "Jacobian " << i << " norm difference: " << norm
        //           << std::endl;
        // std::cout << "Jacobian " << i << std::endl;
        // std::cout << "Analytical:\n"
        //           << analytical_jacobians[i] << std::endl;
        // std::cout << "Numerical:\n"
        //           << numerical_jacobians[i] << std::endl;
        // std::cout << "Difference:\n" << difference << std::endl;
      }
      REQUIRE(is_correct);
    }
  }
}