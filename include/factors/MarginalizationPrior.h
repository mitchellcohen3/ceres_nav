/***************************************************************************
 * Original code Copyright (C) 2019 Chair of Automation Technology / TU Chemnitz
 * libRSF - A Robust Sensor Fusion Library
 * For more information see https://www.tu-chemnitz.de/etit/proaut/libRSF
 *
 * This file contains code derived from the libRSF library.
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

/**
 * @file MarginalPrior.h
 * @author Tim Pfeifer
 * @date 03.12.2019
 * @brief Specialized factor that encodes the linearized information from
 * marginalization.
 * @copyright GNU Public License.
 *
 */

#pragma once

#include "lib/ParameterBlock.h"
#include "lib/StateCollection.h"
#include "lib/StateId.h"
#include "utils/VectorTypes.h"

#include <ceres/ceres.h>

namespace ceres_nav {

/**
 * @brief Stores all the information about a parameter block.
 * Needed for the marginalization prior to evaluate delta_xi.
 */
struct ParameterBlockInfo {
  std::shared_ptr<ParameterBlockBase> param_ptr;
  Eigen::VectorXd linearization_point;
  StateID state_id;

  ParameterBlockInfo(std::shared_ptr<ParameterBlockBase> param_ptr_,
                     const Eigen::VectorXd &linearization_point_,
                     const StateID &state_id_)
      : param_ptr(param_ptr_), linearization_point(linearization_point_),
        state_id(state_id_) {}

  ParameterBlockInfo(std::shared_ptr<ParameterBlockBase> param_ptr_,
                     const StateID &state_id_)
      : param_ptr(param_ptr_), state_id(state_id_) {
    // Initialize the linearization point to the current estimate
    linearization_point = param_ptr->getEstimate();
  }

  ParameterBlockInfo() = default;

  /**
   * @brief sets the linearization point of this parameter block
   */
  void setLinearizationPoint(const Eigen::VectorXd &linearization_point_) {
    // Ensure that the size of this vector matches the size of the parameter
    // block
    if (linearization_point_.size() != param_ptr->dimension()) {
      LOG(ERROR) << "Size of the linearization point does not match the size "
                    "of the parameter block.";
    }

    linearization_point = linearization_point_;
  }
};

class MarginalizationPrior : public ceres::CostFunction {
public:
  // MarginalizationPrior(
  //     const std::vector<int> &LocalSize, const std::vector<int> &GlobalSize,
  //     const std::vector<Eigen::VectorXd> &LinearizationPoints,
  //     const std::vector<const ceres::LocalParameterization *>
  //     &LocalParamPtrs, const Matrix &J, const Vector &R, const
  //     std::vector<StateID> &StateIDs);
  MarginalizationPrior(const std::vector<ParameterBlockInfo> &parameter_blocks,
                       const Matrix &J, const Vector &R);

  ~MarginalizationPrior() override = default;

  bool Evaluate(double const *const *parameters, double *residuals,
                double **jacobians) const override;

private:
  std::vector<ParameterBlockInfo> parameter_blocks_;

  // std::vector<int> LocalSize_;
  // std::vector<int> GlobalSize_;
  // std::vector<const ceres::LocalParameterization *> LocalParamPtrs_;
  // std::vector<StateID> StateIDs;
  int GlobalSizeSum_;
  int LocalSizeSum_;
  Vector LinearResidual_;
  Matrix LinearJacobian_;
};
} // namespace ceres_nav
