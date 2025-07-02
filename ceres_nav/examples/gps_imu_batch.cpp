#include <algorithm>
#include <boost/program_options.hpp>
#include <iostream>
#include <string>
#include <vector>

#include <glog/logging.h>

#include "imu/IMUIncrement.h"
#include "lib/ExtendedPoseParameterBlock.h"
#include "lib/FactorGraph.h"
#include "lib/ParameterBlock.h"

#include "FactorGraphUtils.h"
#include "GPSIMUExampleUtils.h"
#include "utils/Timer.h"
#include "utils/Utils.h"

namespace po = boost::program_options;

po::variables_map handle_args(int argc, const char *argv[]) {
  po::options_description options("Allowed options");

  // clang-format off
    options.add_options()
    ("help", "produce_help_message")
    ("imu_data_fpath", po::value<std::string>()->required(), "Path to IMU data file")
    ("gps_data_filepath", po::value<std::string>()->required(), "Path to GPS data file")
    ("ground_truth_filepath", po::value<std::string>()->required(), "Path to ground truth data file")
    ("sigma_gyro_continuous", po::value<double>()->required(), "Continuous-time gyro white noise")
    ("sigma_accel_continuous", po::value<double>()->required(), "Continuous-time accel white noise")
    ("sigma_gyro_random_walk_continuous", po::value<double>()->required(), "Continuous-time gyro random walk noise")
    ("sigma_accel_random_walk_continuous", po::value<double>()->required(), "Continuous-time accel random walk noise")
    ("sigma_gps_position", po::value<double>()->required(), "GPS position measurement noise")
    ("gravity_mag", po::value<double>()->default_value(9.81), "Magnitude of gravity vector (default: 9.81 m/s^2)")
    ("output_dir", po::value<std::string>()->default_value("/tmp/"), "Output directory for results")
    ("lie_direction", po::value<std::string>()->required(), "Lie direction to run (left or right)");
  // clang-format on

  po::variables_map var_map;
  po::store(po::parse_command_line(argc, argv, options), var_map);

  if (var_map.count("help") || argc == 1) {
    std::cout << "Main entry point for the GPS-IMU fusion example."
              << std::endl;
    std::cout << options << std::endl;
    exit(0);
  }

  po::notify(var_map);
  return var_map;
}

int main(int argc, const char **argv) {
  // Configure glog
  google::InitGoogleLogging(argv[0]);
  FLAGS_logtostderr = true;
  FLAGS_colorlogtostderr = true;
  LOG(INFO) << "Starting GPS-IMU fusion example...";

  // Parse command line arguments
  auto args = handle_args(argc, argv);
  std::vector<IMUMessage> imu_data =
      loadIMUData(args["imu_data_fpath"].as<std::string>());
  std::vector<GPSMessage> gps_data =
      loadGPSData(args["gps_data_filepath"].as<std::string>());
  std::vector<IMUState> imu_states_gt =
      loadIMUStates(args["ground_truth_filepath"].as<std::string>());

  LOG(INFO) << "Number of IMU messages loaded: " << imu_data.size();
  LOG(INFO) << "Number of GPS messages loaded: " << gps_data.size();
  LOG(INFO) << "Number of groundtruth IMU states loaded: "
            << imu_states_gt.size();

  // Batch configuration parameters
  LieDirection lie_direction;
  if (args["lie_direction"].as<std::string>() == "left") {
    lie_direction = LieDirection::left;
  } else if (args["lie_direction"].as<std::string>() == "right") {
    lie_direction = LieDirection::right;
  } else {
    LOG(ERROR) << "Invalid lie direction specified. Use 'left' or 'right'.";
    return -1;
  }

  // Continuous-time IMU noise
  double sigma_gyro = args["sigma_gyro_continuous"].as<double>();
  double sigma_accel = args["sigma_accel_continuous"].as<double>();
  double sigma_gyro_rw = args["sigma_gyro_random_walk_continuous"].as<double>();
  double sigma_accel_rw =
      args["sigma_accel_random_walk_continuous"].as<double>();
  Eigen::Matrix<double, 12, 12> Q_ct =
      Eigen::Matrix<double, 12, 12>::Identity();
  Q_ct.block<3, 3>(0, 0) *= sigma_gyro * sigma_gyro;
  Q_ct.block<3, 3>(3, 3) *= sigma_accel * sigma_accel;
  Q_ct.block<3, 3>(6, 6) *= sigma_gyro_rw * sigma_gyro_rw;
  Q_ct.block<3, 3>(9, 9) *= sigma_accel_rw * sigma_accel_rw;

  double sigma_gps_meas = args["sigma_gps_position"].as<double>();
  Eigen::Matrix3d R_gps =
      sigma_gps_meas * sigma_gps_meas * Eigen::Matrix3d::Identity();
  // Eigen::Matrix3d R_gps = Eigen::Matrix3d::Identity();
  // std::cout << "R_gps: " << std::endl;
  // std::cout << R_gps << std::endl;

  //// Create the factor graph and add measurements
  // Create the factor graph
  ceres_swf::FactorGraph graph;
  factor_graph_utils::ProblemKeys keys;

  // Add the first IMU state to the graph and add a prior factor
  factor_graph_utils::addIMUState(graph, imu_states_gt[0], lie_direction);
  Eigen::Matrix<double, 15, 15> prior_covariance =
      Eigen::Matrix<double, 15, 15>::Identity() * 1e-6; // Small covariance
  factor_graph_utils::addPriorFactor(graph, imu_states_gt[0], prior_covariance,
                                     lie_direction, keys);

  // Now, loop through the measurements and add factors to the graph
  size_t imu_idx = 0;
  double gravity_mag = args["gravity_mag"].as<double>();
  Eigen::Vector3d gravity{0.0, 0.0, -gravity_mag};
  IMUIncrement imu_increment(
      Q_ct, imu_states_gt[0].gyroBias(), imu_states_gt[0].accelBias(),
      imu_states_gt[0].timestamp(), gravity, "continuous", lie_direction);

  IMUState cur_imu_state = imu_states_gt[0];

  // Create a new file for the output
  std::string output_dir = args["output_dir"].as<std::string>();
  std::string init_imu_states_file = output_dir + "/init_imu_states.txt";
  std::string optimized_imu_states_file =
      output_dir + "/optimized_imu_states.txt";
  std::string covariance_file = output_dir + "/covariances.txt";
  createNewFile(optimized_imu_states_file);
  createNewFile(init_imu_states_file);
  createNewFile(covariance_file);

  double prev_gps_timestamp = gps_data[0].timestamp;
  Timer timer_graph;
  timer_graph.tic();
  std::vector<double> est_stamps;

  for (size_t gps_index = 1; gps_index < gps_data.size(); ++gps_index) {
    double cur_gps_timestamp = gps_data[gps_index].timestamp;

    // Integrate IMU data between prev_gps_timestamp and current gps timestamp
    imu_increment.reset(prev_gps_timestamp, cur_imu_state.gyroBias(),
                        cur_imu_state.accelBias());

    while (imu_idx < (imu_data.size() - 1) &&
           imu_data[imu_idx].timestamp < cur_gps_timestamp) {
      double dt = imu_data[imu_idx + 1].timestamp - imu_data[imu_idx].timestamp;
      // Propagate preintegrated measurement
      imu_increment.propagate(dt, imu_data[imu_idx].gyro,
                              imu_data[imu_idx].accel);
      // Propagate IMU state and add to graph
      propagateIMUState(cur_imu_state, imu_data[imu_idx], gravity, dt);
      imu_idx++;
    }

    // Let's write this IMU state to a file
    writeVectorToFile(init_imu_states_file, cur_imu_state.toVector());

    // Add the IMU state to the graph
    factor_graph_utils::addIMUState(graph, cur_imu_state, lie_direction,
                                    keys);

    // Add the preintegrated IMU factor to the graph
    factor_graph_utils::addPreintegrationFactor(graph, imu_increment,
                                                lie_direction, keys);

    // Add the GPS factor to the graph
    factor_graph_utils::addGPSFactor(graph, gps_data[gps_index], lie_direction,
                                     R_gps, keys);

    prev_gps_timestamp = cur_gps_timestamp;
    est_stamps.push_back(cur_imu_state.timestamp());
  }

  LOG(INFO) << "Time to build the graph: " << timer_graph.toc() * 1e-3 << " ms";

  // Create the options
  LOG(INFO) << "Optimizing...";
  ceres::Solver::Options options;
  options.max_num_iterations = 100;
  options.minimizer_progress_to_stdout = true;
  options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
  options.trust_region_strategy_type = ceres::DOGLEG;
  options.max_solver_time_in_seconds = 20.0;

  // Solve the problem
  graph.solve(options);
  ceres::Solver::Summary summary = graph.getSummary();
  LOG(INFO) << "Solver summary: " << summary.FullReport();

  // Get the optimized states
  LOG(INFO) << "Computing covariances...";
  for (auto const &stamp : est_stamps) {
    IMUState cur_imu_state =
        factor_graph_utils::getIMUState(graph, stamp, keys);
    writeVectorToFile(optimized_imu_states_file, cur_imu_state.toVector());

    Eigen::Matrix<double, 15, 15> imu_cov =
        factor_graph_utils::computeIMUCovariance(
            graph, cur_imu_state.timestamp(), keys);
    Eigen::Matrix<double, 226, 1> flat_cov;
    flat_cov(0) = cur_imu_state.timestamp();
    flat_cov.block<225, 1>(1, 0) = flattenMatrix(imu_cov);
    writeVectorToFile(covariance_file, flat_cov);
  }

  return 0;
}