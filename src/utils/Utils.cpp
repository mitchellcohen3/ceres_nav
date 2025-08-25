#include "utils/Utils.h"
#include <fstream>

double roundTo(double value, double precision) {
  return std::round(value / precision) * precision;
}

double roundStamp(double value) { return roundTo(value, 1e-3); }

namespace ceres_nav {

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

void discretizeSystem(const Eigen::MatrixXd &A_ct, const Eigen::MatrixXd &L_ct,
                      const Eigen::MatrixXd &Q_ct, double dt,
                      Eigen::MatrixXd &A_d, Eigen::MatrixXd &Q_d,
                      DiscretizationMethod method) {
  // Compute discrete-time A matrix
  Eigen::MatrixXd A_dt = A_ct * dt;
  Eigen::MatrixXd A_dt_square = A_dt * A_dt;
  Eigen::MatrixXd A_dt_cube = A_dt_square * A_dt;

  A_d = Eigen::MatrixXd::Identity(A_ct.rows(), A_ct.cols()) + A_dt +
        0.5 * A_dt_square + (1.0 / 6.0) * A_dt_cube;

  // Compute the discrete-time noise covariance
  if (method == ceres_nav::DiscretizationMethod::TaylorSeries) {
    Eigen::Matrix<double, 15, 15> Q = L_ct * Q_ct * L_ct.transpose();
    Eigen::Matrix<double, 15, 15> first_term = Q * dt;
    Eigen::Matrix<double, 15, 15> second_term =
        (A_ct * Q + Q * A_ct.transpose()) * (dt * dt) / 2.0;
    Eigen::Matrix<double, 15, 15> third_term =
        (A_ct * A_ct * Q + 2.0 * A_ct * Q * A_ct.transpose() +
         Q * A_ct.transpose() * A_ct.transpose()) *
        (dt * dt * dt) / 6.0;
    Q_d = first_term + second_term + third_term;
    Q_d = 0.5 * (Q_d + Q_d.transpose());
  } else {
    throw std::invalid_argument("Unsupported discretization method");
  }
}

} // namespace ceres_nav