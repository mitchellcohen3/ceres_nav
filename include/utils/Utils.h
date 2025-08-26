#pragma once

#include <Eigen/Dense>
#include <iostream>
#include <math.h>

#include <ceres/ceres.h>

namespace ceres_nav {

double roundTo(double value, double precision);

double roundStamp(double value);

Eigen::MatrixXd computeSquareRootInformation(const Eigen::MatrixXd &covariance);

/**
 * @brief Writes a vector as a single line to a file.
 */
void writeVectorToFile(const std::string &filename,
                       const Eigen::VectorXd &vector);

/**
 * @brief Flattens a square matrix to a vector using row-major order.
 * @param matrix The square matrix to flatten
 * @return A vector containing all matrix elements in row-major order
 */
Eigen::VectorXd flattenMatrix(const Eigen::MatrixXd &matrix);

/**
 * @brief Creates a new file at the specified path.
 * @param filename The path to the file to create
 */
void createNewFile(const std::string &filename);


enum class DiscretizationMethod {
    TaylorSeries,
    FirstOrderHold,
};

/**
 * @brief Discretizes a continuous-time linear system using the specified
 * method.
 */
void discretizeSystem(const Eigen::MatrixXd &A_ct, const Eigen::MatrixXd &L_ct,
                      const Eigen::MatrixXd &Q_ct, double dt,
                      Eigen::MatrixXd &A_d, Eigen::MatrixXd &Q_d,
                      DiscretizationMethod method = DiscretizationMethod::TaylorSeries);

} // namespace ceres_nav