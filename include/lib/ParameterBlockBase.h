#pragma once

#include <Eigen/Dense>
#include <ceres/ceres.h>
#include <memory>

/**
 * @brief Abstract base class for parameter blocks in
 * Ceres.
 *
 * This class provides a common interface for different types of parameter
 * blocks to be used within Ceres.
 *
 * The estimate is stored as an Eigen::VectorXd, and this class allows for
 * storing information related to the parameter block including its name,
 * dimention, local parameterization, and covariance.
 */
class ParameterBlockBase {
public:
  virtual ~ParameterBlockBase() = default;

  explicit ParameterBlockBase(const std::string &name = "parameter_block")
      : name_(name) {
    local_parameterization_ptr_ = nullptr;
  }

  // Core functionality for parameter dimensions and access
  virtual int dimension() const = 0;
  virtual int minimalDimension() const = 0;
  virtual double *estimatePointer() = 0;
  // virtual const double *estimatePointer() const = 0;
  virtual std::string name() const = 0;

  // Parameter block estimate
  virtual Eigen::VectorXd getEstimate() const = 0;
  virtual void setEstimate(const Eigen::VectorXd &estimate) = 0;


  // Eigen::Map<const Eigen::VectorXd> getEstimateMap() const {
  //   return Eigen::Map<const Eigen::VectorXd>(estimatePointer(), dimension());
  // }
  
  Eigen::Map<Eigen::VectorXd> getEstimateMap() {
    return Eigen::Map<Eigen::VectorXd>(estimatePointer(), dimension());
  }

  // Local parameterization for optimization on manifold
  ceres::LocalParameterization *getLocalParameterizationPointer() const {
    return local_parameterization_ptr_;
  }

  // Functionality related to local parameterization
  virtual void plus(const double *x0, const double *delta,
                    double *x0_plus_Delta) const = 0;
  virtual void plusJacobian(const double *x0, double *jacobian) const = 0;

  // Getters and setters for the covariance
  virtual Eigen::MatrixXd getCovariance() const = 0;
  virtual double *getCovariancePointer() = 0;
  virtual void setCovariance(const Eigen::MatrixXd &covariance) = 0;

  // Fixed/free state management
  virtual bool isFixed() const { return fixed_; }
  virtual void setFixed(bool fixed) { fixed_ = fixed; }

protected:
  // Whether the parameter block is fixed or not
  bool fixed_ = false;
  std::string name_ = "unnamed";

  // Local parameterization for this parameter block
  ceres::LocalParameterization *local_parameterization_ptr_;
};