#pragma once

#include "ParameterBlockBase.h"
#include <Eigen/Dense>

namespace ceres_nav {

/**
 * @brief Templated parameter block for optimization in Ceres.
 *
 * This class represents a parameter block with a fixed dimension and minimal
 * dimension. It provides storage for the parameter estimate, covariance, and
 * local parameterization.
 */
template <int Dim, int MinimalDim = Dim>
class ParameterBlock : public ParameterBlockBase {
public:
  /**
   * @brief Constructor for a parameter block.
   */
  explicit ParameterBlock(const std::string &name = "parameter_block")
      : ParameterBlockBase(name) {
    estimate_.setZero();
    covariance_.setIdentity();
  }

  /**
   * @brief Constructor for a parameter block with an initial state
   */
  ParameterBlock(const Eigen::Matrix<double, Dim, 1> &initial_state,
                 const std::string &name = "parameter_block")
      : ParameterBlockBase(name) {
    estimate_ = initial_state;
    covariance_.setIdentity();
  }

  // Core functionalty for parameter dimensions and access
  int dimension() const override { return Dim; }
  int minimalDimension() const override { return MinimalDim; }
  double *estimatePointer() override { return estimate_.data(); }
  std::string name() const override { return name_; }

  // Parameter block estimate
  Eigen::VectorXd getEstimate() const override { return estimate_; }

  void setEstimate(const Eigen::VectorXd &estimate) override {
    if (estimate.size() != Dim) {
      throw std::runtime_error(
          "Estimate size does not match parameter block dimension.");
    }
    estimate_ = estimate;
  }

  Eigen::MatrixXd getCovariance() const override { return covariance_; }

  double *getCovariancePointer() override { return covariance_.data(); }

  void setCovariance(const Eigen::MatrixXd &covariance) {
    if (covariance.rows() != MinimalDim || covariance.cols() != MinimalDim) {
      throw std::runtime_error(
          "Covariance size does not match minimal dimension.");
    }
    covariance_ = covariance;
  }

  // The plus operator
  virtual void plus(const double *x0, const double *Delta_Chi,
                    double *x0_plus_Delta) const override {
    Eigen::Map<const Eigen::Matrix<double, Dim, 1>> x0_(x0);
    Eigen::Map<const Eigen::Matrix<double, Dim, 1>> Delta_Chi_(Delta_Chi);
    Eigen::Map<Eigen::Matrix<double, Dim, 1>> x0_plus_Delta_(x0_plus_Delta);
    x0_plus_Delta_ = x0_ + Delta_Chi_;
  }

  virtual void plusJacobian(const double * /*unused: x*/,
                            double *jacobian) const override {
    Eigen::Map<Eigen::Matrix<double, Dim, Dim, Eigen::RowMajor>> identity(
        jacobian);
    identity.setIdentity();
  }

  // The minus operator - simple subtraction for vector spaces
  virtual void minus(const double *y, const double *x, double *y_minus_x) const override {
    Eigen::Map<const Eigen::Matrix<double, Dim, 1>> y_(y);
    Eigen::Map<const Eigen::Matrix<double, Dim, 1>> x_(x);
    Eigen::Map<Eigen::Matrix<double, Dim, 1>> y_minus_x_(y_minus_x);
    y_minus_x_ = y_ - x_;
  }

protected:
  // Storage for the parameter estimate and covariance
  Eigen::Matrix<double, Dim, 1> estimate_;
  Eigen::Matrix<double, MinimalDim, MinimalDim> covariance_;
};

} // namespace ceres_nav
