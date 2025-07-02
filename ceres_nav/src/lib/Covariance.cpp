#include "lib/Covariance.h"

#include <glog/logging.h>
#include <thread>
#include <vector>

namespace ceres_swf {
bool calculateCovariance(ceres::Problem &graph, StateCollection &states,
                         const std::string &key, double timestamp) {
  // Create a covariance object
  ceres::Covariance::Options cov_options;
  cov_options.num_threads =
      static_cast<int>(std::thread::hardware_concurrency());
  cov_options.apply_loss_function = true;
  ceres::Covariance covariance(cov_options);

  // Create covariance block for each state
  std::vector<const double *> parameter_block_ptrs;
  if (states.hasState(key, timestamp)) {
    parameter_block_ptrs.push_back(
        states.getState(key, timestamp)->estimatePointer());

    // Try the efficient algorithm first
    bool success = covariance.Compute(parameter_block_ptrs, &graph);
    // If we've succeeded, get the covariance
    if (success) {
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
    } else {
      LOG(ERROR) << "Covariance computation of " << key
                 << " failed. No covariance computed!";
      return false;
    }

  } else {
    LOG(ERROR) << "Covariance computation failed. State with key: " << key
               << " and timestamp: " << timestamp
               << " does not exist in the state collection.";
    return false;
  }

  return true;
}
}; // namespace ceres_swf