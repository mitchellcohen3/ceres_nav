#include "lib/Covariance.h"

#include "lib/StateCollection.h"
#include <ceres/ceres.h>
#include <glog/logging.h>
#include <thread>
#include <vector>

namespace ceres_nav {
bool calculateCovariance(ceres::Problem &graph, StateCollection &states,
                         const std::string &key, double timestamp) {

  // Check if the state exists in the collection
  if (!states.hasState(key, timestamp)) {
    LOG(ERROR) << "State with key: " << key << " and timestamp: " << timestamp
               << " does not exist in the state collection.";
    return false;
  }

  // Create a covariance object
  ceres::Covariance::Options cov_options;
  cov_options.num_threads =
      static_cast<int>(std::thread::hardware_concurrency());
  cov_options.apply_loss_function = true;
  cov_options.algorithm_type = ceres::CovarianceAlgorithmType::SPARSE_QR;
  ceres::Covariance covariance(cov_options);

  std::vector<const double *> parameter_block_ptrs;
  parameter_block_ptrs.push_back(
      states.getState(key, timestamp)->estimatePointer());

  // Try with sparse QR first
  if (covariance.Compute(parameter_block_ptrs, &graph)) {
    if (states.getState(key, timestamp)->getLocalParameterizationPointer() ==
        nullptr) {
      covariance.GetCovarianceBlock(
          states.getState(key, timestamp)->estimatePointer(),
          states.getState(key, timestamp)->estimatePointer(),
          states.getState(key, timestamp)->getCovariancePointer());
    } else {
      covariance.GetCovarianceBlockInTangentSpace(
          states.getState(key, timestamp)->estimatePointer(),
          states.getState(key, timestamp)->estimatePointer(),
          states.getState(key, timestamp)->getCovariancePointer());
    }
    return true;
  } else {
    LOG(ERROR) << "Sparse QR covariance computation failed for state: " << key
               << " at timestamp: " << timestamp;
    if (!graph.NumParameterBlocks() > 100) {
      LOG(ERROR) << "Covariance computation of " << key
                 << " failed. No covariance computed!";
      return false;
    }

    LOG(ERROR) << "Jacobian related to state " << key << " at timestamp "
               << timestamp << " is not full rank. Computing with SVD...";
    cov_options.algorithm_type = ceres::CovarianceAlgorithmType::DENSE_SVD;
    cov_options.null_space_rank = -1;
    ceres::Covariance covariance_svd(cov_options);

    // Try to compute again
    if (covariance_svd.Compute(parameter_block_ptrs, &graph)) {
      if (states.getState(key, timestamp)->getLocalParameterizationPointer() ==
          nullptr) {
        covariance_svd.GetCovarianceBlock(
            states.getState(key, timestamp)->estimatePointer(),
            states.getState(key, timestamp)->estimatePointer(),
            states.getState(key, timestamp)->getCovariancePointer());
      } else {
        covariance_svd.GetCovarianceBlockInTangentSpace(
            states.getState(key, timestamp)->estimatePointer(),
            states.getState(key, timestamp)->estimatePointer(),
            states.getState(key, timestamp)->getCovariancePointer());
      }
      return true;
    } else {
      LOG(ERROR) << "Failed to compute covariance for state: " << key
                 << " at timestamp: " << timestamp;
      return false;
    }
  }
}
}; // namespace ceres_nav