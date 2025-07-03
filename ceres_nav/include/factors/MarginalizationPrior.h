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

#include "lib/StateCollection.h"
#include "lib/StateId.h"
#include "utils/VectorTypes.h"

#include <ceres/ceres.h>

namespace ceres_nav {

enum class ParameterType { ExtendedPose, Pose, Vector, Unknown };

ParameterType
getLocalParamType(const ceres::LocalParameterization *local_param);

class MarginalizationPrior : public ceres::CostFunction {
public:
  MarginalizationPrior(
      const std::vector<int> &LocalSize, const std::vector<int> &GlobalSize,
      const std::vector<ceres_nav::Vector> &LinearizationPoints,
      const std::vector<const ceres::LocalParameterization *> &LocalParamPtrs,
      const Matrix &J, const Vector &R, const std::vector<StateID> &StateIDs);

  ~MarginalizationPrior() override = default;

  bool Evaluate(double const *const *parameters, double *residuals,
                double **jacobians) const override;

private:
  std::vector<int> LocalSize_;
  std::vector<int> GlobalSize_;
  std::vector<const ceres::LocalParameterization *> LocalParamPtrs_;
  std::vector<StateID> StateIDs;
  int GlobalSizeSum_;
  int LocalSizeSum_;
  Vector LinearizationPoints_;
  Vector LinearResidual_;
  Matrix LinearJacobian_;
};
} // namespace ceres_nav
