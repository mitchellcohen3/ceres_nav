#include "lib/Covariance.h"
#include "lib/StateCollection.h"
#include "lib/StateId.h"

#include <ceres/ceres.h>
#include <glog/logging.h>
#include <thread>
#include <vector>

namespace ceres_nav {
bool calculateCovariance(ceres::Problem &graph, StateCollection &states,
                         const StateID &state_id) {

  // Check if the state exists in the collection
  if (!states.hasState(state_id)) {
    LOG(ERROR) << "State with ID: " << state_id.toString()
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
  parameter_block_ptrs.push_back(states.getState(state_id)->estimatePointer());

  // Try with sparse QR first
  if (covariance.Compute(parameter_block_ptrs, &graph)) {
    if (states.getState(state_id)->getLocalParameterizationPointer() ==
        nullptr) {
      covariance.GetCovarianceBlock(
          states.getState(state_id)->estimatePointer(),
          states.getState(state_id)->estimatePointer(),
          states.getState(state_id)->getCovariancePointer());
    } else {
      covariance.GetCovarianceBlockInTangentSpace(
          states.getState(state_id)->estimatePointer(),
          states.getState(state_id)->estimatePointer(),
          states.getState(state_id)->getCovariancePointer());
    }
    return true;
  } else {
    LOG(ERROR) << "Sparse QR covariance computation failed for state: "
               << state_id.toString();
    if (!graph.NumParameterBlocks() > 100) {
      LOG(ERROR) << "Covariance computation of " << state_id.toString()
                 << " failed. No covariance computed!";
      return false;
    }

    LOG(ERROR) << "Jacobian related to state " << state_id.toString()
               << " is not full rank. Computing with SVD...";
    cov_options.algorithm_type = ceres::CovarianceAlgorithmType::DENSE_SVD;
    cov_options.null_space_rank = -1;
    ceres::Covariance covariance_svd(cov_options);

    // Try to compute again
    if (covariance_svd.Compute(parameter_block_ptrs, &graph)) {
      if (states.getState(state_id)->getLocalParameterizationPointer() ==
          nullptr) {
        covariance_svd.GetCovarianceBlock(
            states.getState(state_id)->estimatePointer(),
            states.getState(state_id)->estimatePointer(),
            states.getState(state_id)->getCovariancePointer());
      } else {
        covariance_svd.GetCovarianceBlockInTangentSpace(
            states.getState(state_id)->estimatePointer(),
            states.getState(state_id)->estimatePointer(),
            states.getState(state_id)->getCovariancePointer());
      }
      return true;
    } else {
      LOG(ERROR) << "Failed to compute covariance for state: "
                 << state_id.toString();
      return false;
    }
  }
}
}; // namespace ceres_nav