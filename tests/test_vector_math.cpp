#include <catch2/catch_test_macros.hpp>

#include <Eigen/Dense>
#include "utils/VectorMath.h"

TEST_CASE("Test Matrix square root") {
    // Simple case with a diagonal matrix
    ceres_nav::Matrix A(3, 3);
    A << 4, 0, 0,
         0, 9, 0,
         0, 0, 16;

    // Test with a PSD matrix
    ceres_nav::Matrix A_psd(3, 3);
    A_psd << 4, 2, 4,
             2, 9, 6,
             4, 6, 16;
    A_psd = A_psd.transpose() * A_psd;


    std::vector<ceres_nav::Matrix> A_vec;
    A_vec.push_back(A);
    A_vec.push_back(A_psd);

    for (auto const &A : A_vec) {
        std::cout << "Testing matrix:\n" << A << std::endl;
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> A_sqrt, A_sqrtinv;
        ceres_nav::RobustSqrtAndInvSqrt(A, A_sqrt, A_sqrtinv);

        // Test that A_sqrt * A_sqrt = A
        Eigen::MatrixXd I = A_sqrt * A_sqrtinv;
        REQUIRE(I.isApprox(Eigen::MatrixXd::Identity(3, 3), 1e-10));

        // Test that A_sqrtinv * A_sqrtinv = A^-1
        ceres_nav::Matrix A_inv = A.inverse();
        Eigen::MatrixXd I_inv = A_sqrtinv * A_sqrtinv;
        REQUIRE(I_inv.isApprox(A_inv, 1e-10));

        // Test reconstruction
        Eigen::MatrixXd A_reconstructed = A_sqrt.transpose() * A_sqrt;
        REQUIRE(A_reconstructed.isApprox(A, 1e-10));
    }

}

// TEST_CASE("Test jacobian stuff") {
//     // Test Jacobian 
// }