#include <catch2/catch_test_macros.hpp>
#include <iostream>

// #include "base/ceres/swf/CostFunctionUtils.h"
#include "lib/ParameterBlock.h"
#include "lib/PoseParameterBlock.h"
#include "lib/ExtendedPoseParameterBlock.h"

#include "lie/SO3.h"

TEST_CASE("Test Parameter Blocks") {
  ParameterBlock<3> block("3d_point");
  REQUIRE(block.dimension() == 3);
  REQUIRE(block.minimalDimension() == 3);

  // Check the defualt estimate
  Eigen::VectorXd estimate = block.getEstimate();
  REQUIRE(estimate.size() == 3);

  // Set and get new estiamte
  Eigen::Vector3d new_estimate(1.0, 2.0, 3.0);
  block.setEstimate(new_estimate);
  REQUIRE(block.getEstimate().isApprox(new_estimate));

  // Check the estimate pointer
  double *estimate_ptr = block.estimatePointer();
  for (int i = 0; i < 3; ++i) {
    REQUIRE(estimate_ptr[i] == new_estimate[i]);
  }

  // Test the covariance
  Eigen::Matrix3d default_covariance = block.getCovariance();
  REQUIRE(default_covariance.rows() == 3);
  REQUIRE(default_covariance.cols() == 3);
  REQUIRE(default_covariance.isApprox(Eigen::Matrix3d::Identity()));
  // Set a new covariance
  Eigen::Matrix3d new_covariance = 2.0 * Eigen::Matrix3d::Identity();
  block.setCovariance(new_covariance);
  REQUIRE(block.getCovariance().isApprox(new_covariance));
}

TEST_CASE("Test Pose Parameter Block") {
  PoseParameterBlock pose_block("pose_block", LieDirection::left);
  REQUIRE(pose_block.dimension() == 12);
  REQUIRE(pose_block.minimalDimension() == 6);

  // Check the default estimate
  Eigen::VectorXd estimate = pose_block.getEstimate();
  REQUIRE(estimate.size() == 12);

  // Set the estiamte using rotation and position
  Eigen::Matrix3d attitude = SO3::expMap(Eigen::Vector3d(0.1, 0.2, 0.3));
  Eigen::Vector3d position(1.0, 2.0, 3.0);
  pose_block.setFromAttitudeAndPosition(attitude, position);

  Eigen::VectorXd new_estimate = pose_block.getEstimate();
  REQUIRE(new_estimate.size() == 12);

  // Get the estiamte values
  Eigen::Matrix3d estimated_attitude = pose_block.attitude();
  Eigen::Vector3d estimated_position = pose_block.position();
  REQUIRE(estimated_attitude.isApprox(attitude));
  REQUIRE(estimated_position.isApprox(position));

  // Check the estimate pointer
  double *estimate_ptr = pose_block.estimatePointer();
  for (int i = 0; i < 12; ++i) {
    REQUIRE(estimate_ptr[i] == new_estimate[i]);
  }

  // Test the covariance
  Eigen::MatrixXd default_covariance = pose_block.getCovariance();
  REQUIRE(default_covariance.rows() == 6);
  REQUIRE(default_covariance.cols() == 6);
  REQUIRE(default_covariance.isApprox(Eigen::MatrixXd::Identity(6, 6)));

  // Test the plus operation
  auto local_param = pose_block.getLocalParameterizationPointer();
  REQUIRE(local_param != nullptr);

  // Define a perturbation in the Lie algebra
  Eigen::Matrix<double, 6, 1> delta_xi;
  delta_xi << 0.01, 0.02, 0.03, 0.04, 0.05, 0.06;

  // Print the value of the pose before and after
  Eigen::Matrix4d pose_before = pose_block.pose();
  Eigen::Matrix<double, 12, 1> x_plus_delta;
  bool success = local_param->Plus(pose_block.getEstimate().data(),
                                   delta_xi.data(), x_plus_delta.data());
  REQUIRE(success);

  // Convert x_plus_delta to a pose block
  PoseParameterBlock perturbed_pose_block;
  perturbed_pose_block.setEstimate(x_plus_delta);

  Eigen::Matrix4d pose_after = perturbed_pose_block.pose();
  Eigen::Matrix4d expected_pose = SE3::expMap(delta_xi) * pose_before;
  REQUIRE(pose_after.isApprox(expected_pose, 1e-6));
}

TEST_CASE("Test Extended Poses") {
  ExtendedPoseParameterBlock extended_pose_block("extended_pose_block",
                                                 LieDirection::left);
  REQUIRE(extended_pose_block.dimension() == 15);
  REQUIRE(extended_pose_block.minimalDimension() == 9);

  // Check the default estimate
  Eigen::VectorXd estimate = extended_pose_block.getEstimate();
  REQUIRE(estimate.size() == 15);

  // Set the estimate using rotation, velocity, and position
  Eigen::Matrix3d attitude = SO3::expMap(Eigen::Vector3d(0.1, 0.2, 0.3));
  Eigen::Vector3d velocity(0.5, 0.6, 0.7);
  Eigen::Vector3d position(1.0, 2.0, 3.0);
  extended_pose_block.setFromComponents(attitude, velocity, position);
  Eigen::VectorXd new_estimate = extended_pose_block.getEstimate();
  REQUIRE(new_estimate.size() == 15);
  REQUIRE(extended_pose_block.attitude().isApprox(attitude));
  REQUIRE(extended_pose_block.velocity().isApprox(velocity));
  REQUIRE(extended_pose_block.position().isApprox(position));

  // Check the estimate pointer
  double *estimate_ptr = extended_pose_block.estimatePointer();
  for (int i = 0; i < 15; ++i) {
    REQUIRE(estimate_ptr[i] == new_estimate[i]);
  }

  // Test the covariance
  Eigen::MatrixXd default_covariance = extended_pose_block.getCovariance();
  REQUIRE(default_covariance.rows() == 9);
  REQUIRE(default_covariance.cols() == 9);
  REQUIRE(default_covariance.isApprox(Eigen::MatrixXd::Identity(9, 9)));

  /// Test the plus operation
  auto local_param = extended_pose_block.getLocalParameterizationPointer();
  REQUIRE(local_param != nullptr);

  Eigen::Matrix<double, 9, 1> delta_xi;
  delta_xi << 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09;
  Eigen::Matrix<double, 5, 5> extended_pose_before =
      extended_pose_block.extendedPose();

  Eigen::Matrix<double, 15, 1> x_plus_delta;
  bool success = local_param->Plus(extended_pose_block.getEstimate().data(),
                                   delta_xi.data(), x_plus_delta.data());
  REQUIRE(success);

  // Convert x_plus_delta to an extended pose block
  ExtendedPoseParameterBlock perturbed_extended_pose_block;
  perturbed_extended_pose_block.setEstimate(x_plus_delta);
  Eigen::Matrix<double, 5, 5> extended_pose_after =
      perturbed_extended_pose_block.extendedPose();

  Eigen::Matrix<double, 5, 5> expected_extended_pose =
      SE23::expMap(delta_xi) * extended_pose_before;
  REQUIRE(extended_pose_after.isApprox(expected_extended_pose, 1e-6));
}