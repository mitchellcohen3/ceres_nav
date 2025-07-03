#pragma once

#include <iostream>
#include <math.h>
#include <Eigen/Dense>

#include <ceres/ceres.h>

double roundTo(double value, double precision);

double roundStamp(double value);

Eigen::MatrixXd
computeSquareRootInformation(const Eigen::MatrixXd &covariance);


/**
 * @brief Writes a vector as a single line to a file.
*/
void writeVectorToFile(const std::string &filename, const Eigen::VectorXd &vector);

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

