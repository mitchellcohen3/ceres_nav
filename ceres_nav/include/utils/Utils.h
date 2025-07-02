#pragma once

#include <iostream>
#include <math.h>
#include <Eigen/Dense>

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

void createNewFile(const std::string &filename);

