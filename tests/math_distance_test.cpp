#include <gtest/gtest.h>

#include <bit>
#include <cstddef>
#include <cstdint>

#include "clustering/math/distance.h"
#include "clustering/ndarray.h"

#ifdef CLUSTERING_USE_AVX2
#include "clustering/math/detail/distance_avx2.h"
#endif

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

#ifdef CLUSTERING_USE_AVX2

namespace {

// Fills with values whose squared differences are small non-negative integers. With D up to 256
// the partial sums stay well under 2^24 (the largest contiguous integer range in f32), so every
// accumulation order -- scalar head-to-tail, 8-lane parallel with horizontal reduction, or any
// intermediate -- yields the same bits. This is what makes bit-identity between the scalar and
// AVX2 paths enforceable without ULP tolerance on these inputs.
void fillIntegerValued(NDArray<float, 1> &a, NDArray<float, 1> &b) {
  const std::size_t n = a.dim(0);
  for (std::size_t i = 0; i < n; ++i) {
    a(i) = static_cast<float>(i);
    b(i) = static_cast<float>((i * 3U) % 13U);
  }
}

float scalarReference(const NDArray<float, 1> &a, const NDArray<float, 1> &b) {
  const std::size_t n = a.dim(0);
  float sum = 0.0F;
  for (std::size_t i = 0; i < n; ++i) {
    const float d = a(i) - b(i);
    sum += d * d;
  }
  return sum;
}

} // namespace

class SqEuclideanAvx2BitIdentity : public ::testing::TestWithParam<std::size_t> {};

TEST_P(SqEuclideanAvx2BitIdentity, ScalarKernelMatchesAvx2Bitwise) {
  const std::size_t D = GetParam();
  NDArray<float, 1> a({D});
  NDArray<float, 1> b({D});
  fillIntegerValued(a, b);

  ASSERT_TRUE(a.isAligned<32>()) << "owned NDArray<float, 1> must be 32-byte aligned";
  ASSERT_TRUE(b.isAligned<32>()) << "owned NDArray<float, 1> must be 32-byte aligned";

  const float cpo = pointwiseSq(SqEuclideanTag{}, a, b);
  const float ref = scalarReference(a, b);

  // Bit-exact compare via bit_cast: EXPECT_FLOAT_EQ allows 4-ULP slack which would mask a real
  // divergence on these associativity-friendly inputs.
  EXPECT_EQ(std::bit_cast<std::uint32_t>(cpo), std::bit_cast<std::uint32_t>(ref))
      << "D=" << D << " cpo=" << cpo << " ref=" << ref;
}

INSTANTIATE_TEST_SUITE_P(DimGrid, SqEuclideanAvx2BitIdentity,
                         ::testing::Values<std::size_t>(1, 2, 7, 8, 9, 15, 16, 17, 24, 32, 33, 64,
                                                        128, 129));

TEST(SqEuclideanAvx2, DirectKernelMatchesScalarOnD8) {
  // Calls the detail kernel directly to pin its bit-identity with the scalar path, independent
  // of the CPO dispatch gate. If dispatch ever drops below D=8 or changes gate semantics, this
  // test still anchors the kernel's numerical contract.
  constexpr std::size_t D = 8;
  NDArray<float, 1> a({D});
  NDArray<float, 1> b({D});
  fillIntegerValued(a, b);

  ASSERT_TRUE(a.isAligned<32>());
  ASSERT_TRUE(b.isAligned<32>());

  const float kern = clustering::math::distance::detail::sqEuclideanAvx2F32(a.alignedData<32>(),
                                                                            b.alignedData<32>(), D);
  const float ref = scalarReference(a, b);

  EXPECT_EQ(std::bit_cast<std::uint32_t>(kern), std::bit_cast<std::uint32_t>(ref))
      << "direct kernel vs scalar: kern=" << kern << " ref=" << ref;
}

TEST(SqEuclideanAvx2, DispatchStaysScalarBelowEightDims) {
  // D=7 is below the AVX2 gate; the CPO must return the scalar result. We can't directly observe
  // which branch was taken, but the bit-identity check is enough: if any branch disagrees with
  // the scalar reference, the test fails regardless of which one ran.
  constexpr std::size_t D = 7;
  NDArray<float, 1> a({D});
  NDArray<float, 1> b({D});
  fillIntegerValued(a, b);

  const float cpo = pointwiseSq(SqEuclideanTag{}, a, b);
  const float ref = scalarReference(a, b);
  EXPECT_EQ(std::bit_cast<std::uint32_t>(cpo), std::bit_cast<std::uint32_t>(ref));
}

TEST(SqEuclideanAvx2, StridedViewFallsBackToScalarAndStaysCorrect) {
  // col(0) of a (16, 3) source is a length-16 stride-3 view -- long enough to clear the D=8 gate
  // if the contiguity check failed. The AVX2 kernel cannot walk a non-contiguous view (it issues
  // contiguous 8-lane loads), so the CPO must fall back to scalar. Sentinel 99s in the off-column
  // cells turn a buggy stride-1 walk into a loud failure.
  NDArray<float, 2> srcA({16, 3});
  NDArray<float, 2> srcB({16, 3});
  for (std::size_t i = 0; i < 16; ++i) {
    srcA[i][0] = static_cast<float>(i);
    srcB[i][0] = static_cast<float>((i * 3U) % 13U);
    for (std::size_t j = 1; j < 3; ++j) {
      srcA[i][j] = 99.0F;
      srcB[i][j] = 99.0F;
    }
  }

  auto a = srcA.col(0);
  auto b = srcB.col(0);
  ASSERT_EQ(a.dim(0), 16U);
  ASSERT_EQ(a.strideAt(0), static_cast<std::ptrdiff_t>(3));
  ASSERT_FALSE(a.isContiguous());

  // Build the oracle from the scalar path on a dense copy of column 0.
  NDArray<float, 1> aDense({16});
  NDArray<float, 1> bDense({16});
  for (std::size_t i = 0; i < 16; ++i) {
    aDense(i) = srcA[i][0];
    bDense(i) = srcB[i][0];
  }

  const float strided = pointwiseSq(SqEuclideanTag{}, a, b);
  const float ref = scalarReference(aDense, bDense);
  EXPECT_EQ(std::bit_cast<std::uint32_t>(strided), std::bit_cast<std::uint32_t>(ref));
}

#endif // CLUSTERING_USE_AVX2
