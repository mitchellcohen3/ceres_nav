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
    ("lie_direction", po::value<std::string>()->required(), "Lie direction to run (left or right)")
    ("estimator_type", po::value<std::string>()->default_value("full_batch"), "Estimator type (full_batch or sliding_window)")
    ("sliding_window_size", po::value<int>()->default_value(10), "Number of states to keep in sliding window, unused for full batch");
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

/**
 * @brief Runs the sliding window estimator for the GPS-IMU fusion example.
 *
 * The initial window is constructed by dead-reckoning the initial IMU state,
 * adding in new states for optimization at the GPS frequency. Optimization is
 * performed once the window is full.
 *
 * Then, the oldest IMU state is marginalized out of the window, and the process
 * continues until all measurements are processed.
 */
void runSlidingWindowEstimator(
    const std::vector<IMUMessage> &imu_data,
    const std::vector<GPSMessage> &gps_data, const IMUState &init_imu_state,
    const Eigen::Matrix<double, 15, 15> &init_cov, LieDirection lie_direction,
    const Eigen::Matrix<double, 12, 12> &Q_ct, const Eigen::Matrix3d &R_gps,
    const Eigen::Vector3d &gravity, const std::string &est_imu_file,
    const std::string &cov_file, int window_size) {
  ceres::Solver::Options options;
  options.max_num_iterations = 10;
  options.linear_solver_type = ceres::DENSE_QR;
  options.minimizer_progress_to_stdout = false;
  options.max_solver_time_in_seconds = 0.05;
  options.trust_region_strategy_type = ceres::DOGLEG;

  // Create the factor graph and problem keys
  ceres_swf::FactorGraph graph;
  factor_graph_utils::ProblemKeys keys;

  // Add the first IMU state to the graph and add a prior factor
  factor_graph_utils::addIMUState(graph, init_imu_state, lie_direction);
  factor_graph_utils::addPriorFactor(graph, init_imu_state, init_cov,
                                     lie_direction, keys);

  // Create the initial IMU increment
  IMUIncrement imu_increment(
      Q_ct, init_imu_state.gyroBias(), init_imu_state.accelBias(),
      init_imu_state.timestamp(), gravity, "continuous", lie_direction);
  IMUState cur_imu_state = init_imu_state;
  double prev_gps_timestamp = gps_data[0].timestamp;
  std::vector<double> est_stamps;
  Timer timer_graph;
  timer_graph.tic();

  // Main loop
  size_t imu_idx = 0;
  for (size_t gps_index = 1; gps_index < gps_data.size(); gps_index++) {
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

    // Add the new IMU state to the graph along with factors at the timestamp
    factor_graph_utils::addIMUState(graph, cur_imu_state, lie_direction, keys);
    factor_graph_utils::addPreintegrationFactor(graph, imu_increment,
                                                lie_direction, keys);
    factor_graph_utils::addGPSFactor(graph, gps_data[gps_index], lie_direction,
                                     R_gps, keys);
    
    // If we've reached the window size, optimize the graph and marginalize the oldest state
    if (graph.getStates().getNumStatesForType(keys.nav_state_key) >= window_size) {
      graph.solve(options);
      ceres::Solver::Summary summary = graph.getSummary();
      LOG(INFO) << summary.BriefReport();

      // Marginalize the oldest IMU state
      double oldest_stamp;
      bool success = graph.getStates().getLastStamp(
          keys.nav_state_key, oldest_stamp);
      if (!success) {
        LOG(ERROR) << "Failed to get oldest state stamp for key: "
                   << keys.nav_state_key;
        return;
      }
      factor_graph_utils::marginalizeIMUState(graph, oldest_stamp, keys);
    }
  }
}

void runFullBatchEstimator(
    const std::vector<IMUMessage> &imu_data,
    const std::vector<GPSMessage> &gps_data, const IMUState &init_imu_state,
    const Eigen::Matrix<double, 15, 15> &init_cov, LieDirection lie_direction,
    const Eigen::Matrix<double, 12, 12> &Q_ct, const Eigen::Matrix3d &R_gps,
    const Eigen::Vector3d &gravity, const std::string &est_imu_file,
    const std::string &cov_file) {

  // Create the factor graph and problem keys
  ceres_swf::FactorGraph graph;
  factor_graph_utils::ProblemKeys keys;

  // Add the first IMU state to the graph and add a prior factor
  factor_graph_utils::addIMUState(graph, init_imu_state, lie_direction);
  factor_graph_utils::addPriorFactor(graph, init_imu_state, init_cov,
                                     lie_direction, keys);

  // Create the initial IMU increment
  size_t imu_idx = 0;
  IMUIncrement imu_increment(
      Q_ct, init_imu_state.gyroBias(), init_imu_state.accelBias(),
      init_imu_state.timestamp(), gravity, "continuous", lie_direction);

  IMUState cur_imu_state = init_imu_state;
  double prev_gps_timestamp = gps_data[0].timestamp;
  Timer timer_graph;
  timer_graph.tic();
  std::vector<double> est_stamps;

  // Loop through the measurements and add factors to the graph
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

    // Add the new IMU state to the graph along with factors at the timestamp
    factor_graph_utils::addIMUState(graph, cur_imu_state, lie_direction, keys);
    factor_graph_utils::addPreintegrationFactor(graph, imu_increment,
                                                lie_direction, keys);
    factor_graph_utils::addGPSFactor(graph, gps_data[gps_index], lie_direction,
                                     R_gps, keys);

    prev_gps_timestamp = cur_gps_timestamp;
    est_stamps.push_back(cur_imu_state.timestamp());
  }

  LOG(INFO) << "Time to build the graph: " << timer_graph.toc() * 1e-3 << " ms";

  // Create the options
  LOG(INFO) << "Optimizing graph...";
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
    writeVectorToFile(est_imu_file, cur_imu_state.toVector());

    Eigen::Matrix<double, 15, 15> imu_cov =
        factor_graph_utils::computeIMUCovariance(
            graph, cur_imu_state.timestamp(), keys);
    Eigen::Matrix<double, 226, 1> flat_cov;
    flat_cov(0) = cur_imu_state.timestamp();
    flat_cov.block<225, 1>(1, 0) = flattenMatrix(imu_cov);
    writeVectorToFile(cov_file, flat_cov);
  }
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

  // Estunatir configuration parameters
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

  double gravity_mag = args["gravity_mag"].as<double>();
  Eigen::Vector3d gravity = Eigen::Vector3d(0.0, 0.0, -gravity_mag);

  double sigma_gps_meas = args["sigma_gps_position"].as<double>();
  Eigen::Matrix3d R_gps =
      sigma_gps_meas * sigma_gps_meas * Eigen::Matrix3d::Identity();

  // Determine the estimator type
  std::string estimator_type = args["estimator_type"].as<std::string>();
  if (estimator_type != "full_batch" && estimator_type != "sliding_window") {
    LOG(ERROR) << "Invalid estimator type specified. Use 'full_batch' or "
                  "'sliding_window'.";
    return -1;
  }

  LOG(INFO) << "Running estimator type: " << estimator_type;

  // Create output files
  std::string output_dir = args["output_dir"].as<std::string>();
  std::string est_imu_states_file = output_dir + "/optimized_imu_states.txt";
  std::string cov_file = output_dir + "/covariances.txt";
  createNewFile(est_imu_states_file);
  createNewFile(cov_file);

  IMUState init_imu_state = imu_states_gt[0];
  Eigen::Matrix<double, 15, 15> init_cov =
      Eigen::Matrix<double, 15, 15>::Identity() * 1e-6; // Small covariance
  if (estimator_type == "full_batch") {
    runFullBatchEstimator(imu_data, gps_data, init_imu_state, init_cov,
                          lie_direction, Q_ct, R_gps, gravity,
                          est_imu_states_file, cov_file);
  } else {
    int sliding_window_size = args["sliding_window_size"].as<int>();
    LOG(INFO) << "Using sliding window size: " << sliding_window_size;
    runSlidingWindowEstimator(
        imu_data, gps_data, init_imu_state, init_cov, lie_direction, Q_ct,
        R_gps, gravity, est_imu_states_file, cov_file, sliding_window_size);
  }

  return 0;
}