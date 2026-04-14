#include <gtest/gtest.h>

#include <cstddef>

#include "clustering/math/distance.h"
#include "clustering/ndarray.h"

using clustering::NDArray;
using clustering::math::distance::pointwiseSq;
using clustering::math::distance::SqEuclideanTag;

TEST(MathDistanceSqEuclidean, HandComputedSmallVector) {
  NDArray<float, 1> a({3});
  NDArray<float, 1> b({3});
  a(0) = 1.0F;
  a(1) = 2.0F;
  a(2) = 3.0F;
  b(0) = 4.0F;
  b(1) = 6.0F;
  b(2) = 8.0F;
  const float d = pointwiseSq(SqEuclideanTag{}, a, b);
  EXPECT_FLOAT_EQ(d, 50.0F);
}

TEST(MathDistanceSqEuclidean, ZeroLengthReturnsZero) {
  const NDArray<float, 1> a({0});
  const NDArray<float, 1> b({0});
  const float d = pointwiseSq(SqEuclideanTag{}, a, b);
  EXPECT_FLOAT_EQ(d, 0.0F);
}

TEST(MathDistanceSqEuclidean, DoublePrecisionHandComputed) {
  NDArray<double, 1> a({4});
  NDArray<double, 1> b({4});
  a(0) = 0.0;
  a(1) = 1.0;
  a(2) = 2.0;
  a(3) = 3.0;
  b(0) = 1.0;
  b(1) = 1.0;
  b(2) = 1.0;
  b(3) = 1.0;
  const double d = pointwiseSq(SqEuclideanTag{}, a, b);
  EXPECT_DOUBLE_EQ(d, 1.0 + 0.0 + 1.0 + 4.0);
}

TEST(MathDistanceSqEuclidean, IdenticalOperandsReturnsZero) {
  NDArray<float, 1> a({5});
  for (std::size_t i = 0; i < 5; ++i) {
    a(i) = static_cast<float>(i) * 1.25F;
  }
  const float d = pointwiseSq(SqEuclideanTag{}, a, a);
  EXPECT_FLOAT_EQ(d, 0.0F);
}

TEST(MathDistanceSqEuclidean, StridedOperandMatchesContig) {
  // src.t().row(0) walks column 0 of src with stride equal to src's row-stride (4),
  // so the view is length-3 stride-4 -- truly non-unit stride. Sentinel values in the
  // other columns would be visited by a buggy stride-1 walk.
  NDArray<float, 2> src({3, 4});
  src[0][0] = 1.0F;
  src[1][0] = 2.0F;
  src[2][0] = 3.0F;
  src[0][1] = 99.0F;
  src[0][2] = 99.0F;
  src[0][3] = 99.0F;
  src[1][1] = 99.0F;
  src[1][2] = 99.0F;
  src[1][3] = 99.0F;
  src[2][1] = 99.0F;
  src[2][2] = 99.0F;
  src[2][3] = 99.0F;

  NDArray<float, 1> b({3});
  b(0) = 4.0F;
  b(1) = 6.0F;
  b(2) = 8.0F;

  auto col0_contig = src.col(0);
  EXPECT_EQ(col0_contig.strideAt(0), static_cast<std::ptrdiff_t>(4));
  const float d_strided = pointwiseSq(SqEuclideanTag{}, col0_contig, b);
  EXPECT_FLOAT_EQ(d_strided, 50.0F);

  auto strided_row0 = src.t().row(0);
  EXPECT_EQ(strided_row0.dim(0), 3u);
  EXPECT_EQ(strided_row0.strideAt(0), static_cast<std::ptrdiff_t>(4));
  const float d_strided_row = pointwiseSq(SqEuclideanTag{}, strided_row0, b);
  EXPECT_FLOAT_EQ(d_strided_row, 50.0F);
}

TEST(MathDistanceSqEuclidean, BothStridedOperands) {
  // Both operands are length-3 stride-4 views of a (3,4) source via src.t().row(0).
  // Sentinel 99s in the off-column entries turn a stride-1 walk into a loud failure.
  NDArray<float, 2> srcA({3, 4});
  NDArray<float, 2> srcB({3, 4});
  srcA[0][0] = 1.0F;
  srcA[1][0] = 2.0F;
  srcA[2][0] = 3.0F;
  srcB[0][0] = 4.0F;
  srcB[1][0] = 6.0F;
  srcB[2][0] = 8.0F;
  for (std::size_t i = 0; i < 3; ++i) {
    for (std::size_t j = 1; j < 4; ++j) {
      srcA[i][j] = 99.0F;
      srcB[i][j] = 99.0F;
    }
  }

  auto strA = srcA.t().row(0);
  auto strB = srcB.t().row(0);
  EXPECT_EQ(strA.strideAt(0), static_cast<std::ptrdiff_t>(4));
  EXPECT_EQ(strB.strideAt(0), static_cast<std::ptrdiff_t>(4));
  const float d = pointwiseSq(SqEuclideanTag{}, strA, strB);
  EXPECT_FLOAT_EQ(d, 50.0F);
}

TEST(MathDistanceSqEuclidean, NoexceptPropagates) {
  const NDArray<float, 1> a({3});
  const NDArray<float, 1> b({3});
  static_assert(noexcept(pointwiseSq(SqEuclideanTag{}, a, b)),
                "pointwiseSq must be noexcept when the selected overload is noexcept");
  NDArray<float, 2> src({2, 3});
  auto strA = src.t().col(0);
  auto strB = src.t().col(1);
  static_assert(noexcept(pointwiseSq(SqEuclideanTag{}, strA, strB)),
                "pointwiseSq must be noexcept for MaybeStrided operands too");
  SUCCEED();
}
