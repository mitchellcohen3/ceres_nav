/**
 * @brief Custom robust loss functions for use in Ceres.
 */

#include <ceres/ceres.h>

namespace ceres_nav {

/**
 * @brief Truncated least squares loss function.
 * 
 * This loss function behaves like a standard least squares loss for residuals 
 * below a certain threshold, but truncates the loss for residuals above that
 * threshold.
 */
class TruncatedLeastSquaresLoss : public ceres::LossFunction {

public:
  explicit TruncatedLeastSquaresLoss(double threshold)
      : threshold_(threshold) {}

  virtual ~TruncatedLeastSquaresLoss() override = default;
  void Evaluate(double sq_norm, double *rho) const override {
    if (sq_norm > threshold_) {
      rho[0] = threshold_;
      rho[1] = 0.0;
      rho[2] = 0.0;
    } else {
      rho[0] = sq_norm;
      rho[1] = 1.0;
      rho[2] = 0.0;
    }
  }

private:
  double threshold_;
};

} // namespace ceres_nav