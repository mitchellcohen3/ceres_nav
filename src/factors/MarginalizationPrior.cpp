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

ParameterType
getLocalParamType(const ceres::LocalParameterization *local_param) {
  if (local_param == nullptr) {
    return ParameterType::Vector;
  }

  if (dynamic_cast<const SE23LocalParameterization *>(local_param) != nullptr) {
    return ParameterType::ExtendedPoseSE23;
  }

  if (dynamic_cast<const DecoupledExtendedPoseLocalParameterization *>(
          local_param) != nullptr) {
    return ParameterType::ExtendedPoseDecoupled;
  }

  if (dynamic_cast<const PoseLocalParameterization *>(local_param) != nullptr) {
    return ParameterType::Pose;
  }

  return ParameterType::Unknown;
}

MarginalizationPrior::MarginalizationPrior(
    const std::vector<int> &LocalSize, const std::vector<int> &GlobalSize,
    const std::vector<Vector> &LinearizationPoints,
    const std::vector<const ceres::LocalParameterization *> &LocalParam_ptrs,
    const Matrix &J, const Vector &R, const std::vector<StateID> &StateIDs) {
  /** check if the data is complete */
  if ((LocalSize.size() == GlobalSize.size()) &&
      (LocalSize.size() == LinearizationPoints.size()) &&
      (LocalSize.size() == LocalParam_ptrs.size())) {
    LocalSize_ = LocalSize;
    GlobalSize_ = GlobalSize;
    LocalParamPtrs_ = LocalParam_ptrs;

    /** compute overall system size */
    int LocalSum = 0, GlobalSum = 0;
    for (int n = 0; n < static_cast<int>(LocalSize.size()); n++) {
      LocalSum += LocalSize.at(n);
      GlobalSum += GlobalSize.at(n);
    }
    LocalSizeSum_ = LocalSum;
    GlobalSizeSum_ = GlobalSum;

    /** store linear system */
    LinearJacobian_ = J;
    LinearResidual_ = R;

    /** store linearization points */
    LinearizationPoints_.resize(GlobalSizeSum_);
    int CumSum = 0;
    for (int n = 0; n < static_cast<int>(LinearizationPoints.size()); n++) {
      LinearizationPoints_.segment(CumSum, GlobalSize.at(n)) =
          LinearizationPoints.at(n);
      CumSum += GlobalSize.at(n);
    }

    /** parametrize factor */
    this->set_num_residuals(LocalSizeSum_);
    for (int BlockSize : GlobalSize) {
      this->mutable_parameter_block_sizes()->push_back(BlockSize);
    }
  } else {
    LOG(ERROR) << "MarginalizationPrior: Inconsistent sizes of input data!";
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

  VectorRef<double, Dynamic> Error(Residuals, LocalSizeSum_);
  Vector DeltaState(LocalSizeSum_);

  // Compute block-wise errors
  int IndexError = 0;
  int IndexState = 0;
  for (int nState = 0; nState < static_cast<int>(GlobalSize_.size());
       nState++) {

    // Get the dimensions of the sub-block
    int GlobalSize = GlobalSize_.at(nState);
    int LocalSize = LocalSize_.at(nState);

    /** map relevant variables */
    const VectorRefConst<double, Dynamic> State(Parameters[nState], GlobalSize);
    const Vector LinearState =
        LinearizationPoints_.segment(IndexState, GlobalSize);

    // Downcast the local parameterization
    const ceres::LocalParameterization *local_param =
        LocalParamPtrs_.at(nState);

    ParameterType param_type = getLocalParamType(local_param);

    switch (param_type) {
    case ParameterType::Vector:
      if (LocalSize != GlobalSize) {
        LOG(ERROR)
            << "MarginalizationPrior: Local size does not match global size!";
        DeltaState.segment(IndexError, LocalSize).setZero();
      } else {
        DeltaState.segment(IndexError, LocalSize) = State - LinearState;
      }

      // Set identity Jacobian
      if (HasJacobian) {
        JacobianManifold.block(IndexError, IndexState, LocalSize, GlobalSize)
            .setIdentity();
      }
      break;
    case ParameterType::ExtendedPoseSE23: {

      const ExtendedPoseLocalParameterization *extended_param =
          dynamic_cast<const ExtendedPoseLocalParameterization *>(local_param);
      if (extended_param == nullptr) {
        LOG(ERROR) << "MarginalizationPrior: downcast to "
                      "ExtendedPoseLocalParameterization failed!";
        return false;
      }
      LieDirection direction = extended_param->direction();

      // Compute x.minus(x_prior) for the extended pose
      Eigen::Matrix3d C_prior = SO3::unflatten(LinearState.block<9, 1>(0, 0));
      Eigen::Vector3d v_prior = LinearState.block<3, 1>(9, 0);
      Eigen::Vector3d r_prior = LinearState.block<3, 1>(12, 0);

      // Eigen::Quaterniond q(State.block<4, 1>(0, 0));
      // Eigen::Matrix3d C = q.toRotationMatrix();
      Eigen::Matrix3d C = SO3::unflatten(State.block<9, 1>(0, 0));
      Eigen::Vector3d v = State.block<3, 1>(9, 0);
      Eigen::Vector3d r = State.block<3, 1>(12, 0);

      // Construct SE23 matrices and compute difference
      Eigen::Matrix<double, 5, 5> cur_pose = SE23::fromComponents(C, v, r);
      Eigen::Matrix<double, 5, 5> pose_prior =
          SE23::fromComponents(C_prior, v_prior, r_prior);
      Eigen::Matrix<double, 9, 1> pose_diff =
          SE23::minus(cur_pose, pose_prior, direction);

      DeltaState.segment(IndexError, LocalSize) = pose_diff;
      if (HasJacobian) {
        // Get the manifold Jacobian from Ceres
        Eigen::Matrix<double, 15, 9> jac = extended_param->getEigenJacobian();
        JacobianManifold.block(IndexError, IndexState, LocalSize, GlobalSize) =
            jac.transpose();
      }
      break;
    }
    case ParameterType::ExtendedPoseDecoupled: {
      const DecoupledExtendedPoseLocalParameterization *decoupled_param =
          dynamic_cast<const DecoupledExtendedPoseLocalParameterization *>(
              local_param);
      if (decoupled_param == nullptr) {
        LOG(ERROR) << "MarginalizationPrior: downcast failed.";
      }

      LieDirection direction = decoupled_param->direction();

      // Compute x.minus(x_prior) for the decoupled extended pose
      Eigen::Matrix3d C_prior = SO3::unflatten(LinearState.block<9, 1>(0, 0));
      Eigen::Vector3d v_prior = LinearState.block<3, 1>(9, 0);
      Eigen::Vector3d r_prior = LinearState.block<3, 1>(12, 0);

      Eigen::Matrix3d C = SO3::unflatten(State.block<9, 1>(0, 0));
      Eigen::Vector3d v = State.block<3, 1>(9, 0);
      Eigen::Vector3d r = State.block<3, 1>(12, 0);

      Eigen::Matrix<double, 9, 1> error;
      error.block<3, 1>(0, 0) = SO3::minus(C, C_prior, direction);
      error.block<3, 1>(3, 0) = v - v_prior;
      error.block<3, 1>(6, 0) = r - r_prior;
      DeltaState.segment(IndexError, LocalSize) = error;
      if (HasJacobian) {
        // Get the manifold Jacobian from Ceres
        Eigen::Matrix<double, 15, 9> jac = decoupled_param->getEigenJacobian();
        JacobianManifold.block(IndexError, IndexState, LocalSize, GlobalSize) =
            jac.transpose();
      }
      break;
    }
    case ParameterType::Pose: {
      const PoseLocalParameterization *pose_param =
          dynamic_cast<const PoseLocalParameterization *>(local_param);
      if (pose_param == nullptr) {
        LOG(ERROR) << "MarginalizationPrior: downcast to "
                      "PoseLocalParameterization failed!";
        return false;
      }

      LieDirection direction = pose_param->direction();
      // Compute x.minus(x_prior) for the pose
      Eigen::Matrix3d C_prior = SO3::unflatten(LinearState.block<9, 1>(0, 0));
      Eigen::Vector3d r_prior = LinearState.block<3, 1>(9, 0);

      Eigen::Matrix3d C = SO3::unflatten(State.block<9, 1>(0, 0));
      Eigen::Vector3d r = State.block<3, 1>(9, 0);

      Eigen::Matrix<double, 4, 4> cur_pose = SE3::fromComponents(C, r);
      Eigen::Matrix<double, 4, 4> pose_prior =
          SE3::fromComponents(C_prior, r_prior);
      Eigen::Matrix<double, 6, 1> pose_diff =
          SE3::minus(cur_pose, pose_prior, direction);

      DeltaState.segment(IndexError, LocalSize) = pose_diff;
      if (HasJacobian) {
        // Get the manifold Jacobian from Ceres
        Eigen::Matrix<double, 12, 6> jac = pose_param->getEigenJacobian();
        JacobianManifold.block(IndexError, IndexState, LocalSize, GlobalSize) =
            jac.transpose();
      }
      break;
    }
    case ParameterType::Unknown:
      LOG(ERROR) << "MarginalizationPrior: Unknown parameter type!";
      return false;
    default:
      LOG(ERROR) << "MarginalizationPrior: Unknown parameter type!";
      return false;
    }

    // Move index to next error block
    IndexError += LocalSize;
    IndexState += GlobalSize;
  }

  // Compute the error
  Error = LinearJacobian_ * DeltaState + LinearResidual_;

  /** jacobians are composed of linear jacobian and manifold jacobian */
  if (HasJacobian) {
    int IndexStateJac = 0;
    for (int nState = 0; nState < static_cast<int>(GlobalSize_.size());
         nState++) {
      int GlobalSize = GlobalSize_.at(nState);

      if (Jacobians[nState] != nullptr) {
        // Here Ceres expects us to provide the Jacobian with respect to the
        // global parameters.
        // We need to add a column of zeros to the LinearJacobian_, and this
        // is taken care of by Jacobian Manifold.
        MatrixRef<double, Dynamic, Dynamic> Jacobian(Jacobians[nState],
                                                     LocalSizeSum_, GlobalSize);
        Jacobian = LinearJacobian_ *
                   JacobianManifold.middleCols(IndexStateJac, GlobalSize);
      }

      IndexStateJac += GlobalSize;
    }
  }

  return true;
}
} // namespace ceres_nav
