/***************************************************************************
 * libRSF - A Robust Sensor Fusion Library
 *
 * Copyright (C) 2019 Chair of Automation Technology / TU Chemnitz
 * For more information see https://www.tu-chemnitz.de/etit/proaut/libRSF
 *
 * libRSF is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * libRSF is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with libRSF.  If not, see <http://www.gnu.org/licenses/>.
 *
 * Author: Tim Pfeifer (tim.pfeifer@etit.tu-chemnitz.de)
 ***************************************************************************/

#include "factors/MarginalizationPrior.h"

#include "local_parameterizations/DecoupledExtendedPoseLocalParameterization.h"
#include "local_parameterizations/SE23LocalParameterization.h"
// #include "local_parameterizations/ExtendedPoseLocalParameterization.h"
#include "local_parameterizations/PoseLocalParameterization.h"

#include "lie/SE23.h"
#include "lie/SE3.h"
#include "lie/SO3.h"

#include <glog/logging.h>

namespace ceres_nav {

MarginalizationPrior::MarginalizationPrior(
    const std::vector<ParameterBlockInfo> &parameter_blocks, const Matrix &J,
    const Vector &R)
    : parameter_blocks_(parameter_blocks), LinearJacobian_(J), LinearResidual_(R) {

  // Compute overall system size
  int local_sum = 0;
  int global_sum = 0;
  for (const auto &param_block : parameter_blocks_) {
    local_sum += param_block.param_ptr->minimalDimension();
    global_sum += param_block.param_ptr->dimension();
  }

  LocalSizeSum_ = local_sum;
  GlobalSizeSum_ = global_sum;

  // Dimension of the residual is the sum of the local sizes
  this->set_num_residuals(LocalSizeSum_);

  for (const auto &param_block : parameter_blocks_) {
    this->mutable_parameter_block_sizes()->push_back(
        param_block.param_ptr->dimension());
  }
}

bool MarginalizationPrior::Evaluate(double const *const *Parameters,
                                    double *Residuals,
                                    double **Jacobians) const {
  /** use separate jacobian to represent the manifold operations */
  Matrix JacobianManifold;
  bool HasJacobian = false;
  if (Jacobians != nullptr) {
    JacobianManifold.resize(LocalSizeSum_, GlobalSizeSum_);
    JacobianManifold.setZero();
    HasJacobian = true;
  }

  // Eigen::Map<Eigen::VectorXd> Error(Residuals, LocalSizeSum_);
  VectorRef<double, Dynamic> Error(Residuals, LocalSizeSum_);
  Vector DeltaState(LocalSizeSum_);
  Eigen::VectorXd delta_xi(LocalSizeSum_);
  delta_xi.setZero();

  int index_error = 0;
  int index_state = 0;

  // Compute block-wise errors for each parameter block
  for (int nState = 0; nState < parameter_blocks_.size(); nState++){
    // LOG(INFO) << "Processing state block " << nState;
    // LOG(INFO) << "Parameter block size: " << parameter_blocks_[nState].param_ptr->dimension();
    // LOG(INFO) << "Local size: " << parameter_blocks_[nState].param_ptr->minimalDimension();

    ParameterBlockInfo param_block = parameter_blocks_[nState];
    int global_size = param_block.param_ptr->dimension();
    int local_size = param_block.param_ptr->minimalDimension();

    // Map the relevant variables
    const VectorRefConst<double, Dynamic> State(Parameters[nState],
                                                global_size);
    const Vector LinearState = param_block.linearization_point;


    // Compute the error for this state block
    Eigen::VectorXd error(local_size);
    param_block.param_ptr->minus(State.data(), LinearState.data(),
                                 error.data());
    DeltaState.segment(index_error, local_size) = error;

    // Compute the manifold Jacobian if required
    if (HasJacobian) {
      // Get the local parameterization pointer
      const ceres::LocalParameterization *local_param =
          param_block.param_ptr->getLocalParameterizationPointer();
      if (local_param == nullptr) {
        // No local parameterization - identity Jacobian
        // Ensure local size and global size are the same
        if (local_size != global_size) {
          LOG(ERROR)
              << "MarginalizationPrior: Local size does not match global size!";
        }
        JacobianManifold
            .block(index_error, index_state, local_size, global_size)
            .setIdentity();
      } else {
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
            jac(global_size, local_size);
        local_param->ComputeJacobian(State.data(), jac.data());
        JacobianManifold.block(index_error, index_state, local_size,
                               global_size) = jac.transpose();
      }
    }

    // Move indices to the next error block
    index_error += local_size;
    index_state += global_size;
  }


  // Compute the error
  Error = LinearJacobian_ * DeltaState + LinearResidual_;

  if (HasJacobian) {
    int nState = 0;
    int IndexStateJac = 0;
    for (auto const &param_block : parameter_blocks_) {
      int global_size = param_block.param_ptr->dimension();
      if (Jacobians[nState] != nullptr) {
        // Here Ceres expects us to provide the Jacobian with respect to the
        // global parameters.
        // We need to add a column of zeros to the LinearJacobian_, and this
        // is taken care of by Jacobian Manifold.
        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                                 Eigen::RowMajor>>
            Jacobian(Jacobians[nState], LocalSizeSum_, global_size);
        Jacobian = LinearJacobian_ *
                   JacobianManifold.middleCols(IndexStateJac, global_size);
      }

      nState++;
      IndexStateJac += global_size;
    }
  }

  return true;
}
} // namespace ceres_nav
