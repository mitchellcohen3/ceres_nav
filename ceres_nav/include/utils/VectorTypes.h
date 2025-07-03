/**
 * @brief This code is taken from libRSF (https://github.com/TUC-ProAut/libRSF) under the GPL-3.0 license.
 */
#pragma once

#include <Eigen/Dense>
#include <Eigen/StdVector>

namespace ceres_nav {
/** index for matrix types */
using Index = Eigen::Index;
const Index Dynamic = Eigen::Dynamic;

/** generic matrix */
template <typename T, int Row, int Col>
using MatrixT = Eigen::Matrix<T, Row, Col, ((Col == 1) ? Eigen::ColMajor : Eigen::RowMajor)>;

template <typename T, int Row> using VectorT = MatrixT<T, Row, 1>;

/** double static template types */
template <int Row, int Col> using MatrixStatic = MatrixT<double, Row, Col>;

template <int Row> using VectorStatic = VectorT<double, Row>;

/** double dynamic types */
using Vector = VectorStatic<Dynamic>;
using Matrix = MatrixStatic<Dynamic, Dynamic>;

/** double static fixed types */
using Vector1 = VectorStatic<1>;
using Vector2 = VectorStatic<2>;
using Vector3 = VectorStatic<3>;
using Vector4 = VectorStatic<4>;
using Vector5 = VectorStatic<5>;
using Vector6 = VectorStatic<6>;
using Vector7 = VectorStatic<7>;
using Vector8 = VectorStatic<8>;
using Vector9 = VectorStatic<9>;
using Vector10 = VectorStatic<10>;
using Vector11 = VectorStatic<11>;
using Vector12 = VectorStatic<12>;
using Vector13 = VectorStatic<13>;
using Vector14 = VectorStatic<14>;
using Vector15 = VectorStatic<15>;

using Matrix11 = MatrixStatic<1, 1>;
using Matrix22 = MatrixStatic<2, 2>;
using Matrix33 = MatrixStatic<3, 3>;
using Matrix34 = MatrixStatic<3, 4>;
using Matrix43 = MatrixStatic<4, 3>;
using Matrix44 = MatrixStatic<4, 4>;
using Matrix66 = MatrixStatic<6, 6>;
using Matrix77 = MatrixStatic<7, 7>;
using Matrix88 = MatrixStatic<8, 8>;
using Matrix99 = MatrixStatic<9, 9>;
using Matrix1010 = MatrixStatic<10, 10>;

using Matrix2X = MatrixStatic<2, Dynamic>;
using Matrix3X = MatrixStatic<3, Dynamic>;

/** reference wrappers */
template <typename T, int Row, int Col> using MatrixRef = Eigen::Map<MatrixT<T, Row, Col>>;

template <typename T, int Row> using VectorRef = Eigen::Map<VectorT<T, Row>>;

/** constant reference wrappers */
template <typename T, int Row, int Col>
using MatrixRefConst = Eigen::Map<const MatrixT<T, Row, Col>>;

template <typename T, int Row> using VectorRefConst = Eigen::Map<const VectorT<T, Row>>;

/** 2D rotation matrix */
template <typename T> using Rotation2DT = Eigen::Rotation2D<T>;

using Rotation2D = Rotation2DT<double>;

/** 3D quaternions */
template <typename T> using QuaternionT = Eigen::Quaternion<T>;

template <typename T> using QuaternionRef = Eigen::Map<QuaternionT<T>>;

template <typename T> using QuaternionRefConst = Eigen::Map<const QuaternionT<T>>;

using Quaternion = QuaternionT<double>;

template <typename T> QuaternionT<T> VectorToQuaternion(const VectorT<T, 4> &Vec) {
  QuaternionT<T> Quat(Vec(3), Vec(0), Vec(1),
                      Vec(2)); /**< storage order is different from constructor! */
  return Quat;
}

/** 3D angle-axis representation */
template <typename T> using AngleAxisT = Eigen::AngleAxis<T>;

using AngleAxis = AngleAxisT<double>;

/** safe STL container */
template <int Row, int Col>
using MatrixVectorSTL =
    std::vector<MatrixT<double, Row, Col>, Eigen::aligned_allocator<MatrixT<double, Row, Col>>>;

template <int Row> using VectorVectorSTL = MatrixVectorSTL<Row, 1>;
} // namespace libRSF