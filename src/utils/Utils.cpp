#include "utils/Utils.h"
#include <fstream>

double roundTo(double value, double precision) {
  return std::round(value / precision) * precision;
}

double roundStamp(double value) { return roundTo(value, 1e-3); }

Eigen::MatrixXd
computeSquareRootInformation(const Eigen::MatrixXd &covariance) {
  // Check that we have a valid covariance that we can get the information of
  Eigen::MatrixXd I =
      Eigen::MatrixXd::Identity(covariance.rows(), covariance.rows());
  Eigen::MatrixXd information = covariance.llt().solve(I);
  if (std::isnan(information.norm())) {
    std::cerr << "Covariance : " << std::endl
              << covariance << std::endl
              << std::endl;
    std::cerr << "Inverse covariance : " << std::endl
              << covariance.inverse() << std::endl
              << std::endl;
  }
  // Compute square-root of the information matrix
  Eigen::LLT<Eigen::MatrixXd> lltOfI(information);
  Eigen::MatrixXd sqrt_info = lltOfI.matrixL().transpose();
  return sqrt_info;
}

void writeVectorToFile(const std::string &file_path,
                       const Eigen::VectorXd &vector) {
  std::ofstream file(file_path, std::ios::app);
  if (!file.is_open()) {
    std::cerr << "Error opening file: " << file_path << std::endl;
    return;
  }

  // Write the vector elements to a file
  for (int i = 0; i < vector.size(); i++) {
    file << vector[i];
    if (i < vector.size() - 1) {
      file << ", "; // Add a comma between elements
    }
  }
  file << std::endl;
  file.close();
}

void createNewFile(const std::string &file_path) {
  std::ofstream file(file_path);
  if (!file.is_open()) {
    std::cerr << "Error creating file: " << file_path << std::endl;
    return;
  }
  file.close();
}

Eigen::VectorXd flattenMatrix(const Eigen::MatrixXd &matrix) {
  // Use Eigen::Map for efficient memory mapping without copying data
  return Eigen::Map<const Eigen::VectorXd>(matrix.data(), matrix.size());
}