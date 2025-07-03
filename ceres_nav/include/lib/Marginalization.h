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
 * @file Marginalization.h
 * @author Tim Pfeifer
 * @date 10.12.2019
 * @brief Functions that cover the math behind marginalizing states out.
 * @copyright GNU Public License.
 *
 */


#include "utils/VectorMath.h"

namespace ceres_nav {
void Marginalize(const Vector &Residual, const Matrix &Jacobian, Vector &ResidualMarg,
                 Matrix &JacobianMarg, int MarginalSize, double HessianInflation = 1.0);
}