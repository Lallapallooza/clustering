#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <span>

#include "clustering/math/reduce.h"
#include "clustering/ndarray.h"

using clustering::NDArray;
using clustering::math::argmax;
using clustering::math::argmin;
using clustering::math::meanVar;
using clustering::math::sum;
using clustering::math::sumKahan;
using clustering::math::topk;

namespace {

template <class T, std::size_t N> NDArray<T, 1> vec(std::array<T, N> values) {
  NDArray<T, 1> a({N});
  for (std::size_t i = 0; i < N; ++i) {
    a(i) = values[i];
  }
  return a;
}

} // namespace

TEST(Sum, EmptyReturnsZero) {
  const NDArray<float, 1> a({0});
  EXPECT_FLOAT_EQ(sum(a), 0.0F);
}

TEST(Sum, MatchesExpectedOnFixedVector) {
  const auto a = vec<double, 4>({1.25, -0.5, 2.0, 3.5});
  EXPECT_DOUBLE_EQ(sum(a), 6.25);
}

TEST(Sum, StridedInputsWork) {
  NDArray<double, 2> a({3, 2});
  for (std::size_t i = 0; i < 3; ++i) {
    for (std::size_t j = 0; j < 2; ++j) {
      a[i][j] = static_cast<double>((i * 2) + j);
    }
  }
  // Column 0 is strided (row-major base, stride > 1).
  const auto col = a.col(0);
  EXPECT_DOUBLE_EQ(sum(col), 0.0 + 2.0 + 4.0);
}

TEST(SumKahan, EmptyReturnsZero) {
  const NDArray<double, 1> a({0});
  EXPECT_DOUBLE_EQ(sumKahan(a), 0.0);
}

TEST(SumKahan, MatchesNaiveOnSmallClean) {
  const auto a = vec<double, 5>({1.0, 2.0, 3.0, 4.0, 5.0});
  EXPECT_DOUBLE_EQ(sumKahan(a), sum(a));
  EXPECT_DOUBLE_EQ(sumKahan(a), 15.0);
}

TEST(SumKahan, BeatsNaiveOnLargeSignalPlusNoise) {
  // Pathological pattern: one dominant f64 term 2^53 (the last integer above which f64 ULP = 2)
  // followed by many 1.0 additions. Each 1.0 added to 2^53 rounds to 2^53 under round-to-even
  // because the gap becomes 2 ULPs, so naive accumulation silently drops the noise while the
  // true sum 2^53 + n is exactly representable (it stays an integer above 2^53). Kahan's
  // compensation scalar recovers the full precision.
  constexpr std::size_t n = 1'000'001;
  NDArray<double, 1> a({n});
  a(0) = 9007199254740992.0; // 2^53
  for (std::size_t i = 1; i < n; ++i) {
    a(i) = 1.0;
  }
  // n-1 = 1'000'000 ones added to 2^53; 2^53 + 1e6 is even and exactly representable in f64.
  const double trueSum = 9007199254740992.0 + static_cast<double>(n - 1);
  const double kahanResult = sumKahan(a);
  const double naiveResult = sum(a);
  const double kahanError = std::fabs(kahanResult - trueSum);
  const double naiveError = std::fabs(naiveResult - trueSum);
  EXPECT_LT(kahanError, 1.0) << "Kahan off by " << kahanError;
  EXPECT_GT(naiveError, 1e5) << "Naive should absorb small addends; got error " << naiveError;
  EXPECT_GT(naiveError, kahanError * 100.0);
}

TEST(MeanVar, EmptyReturnsZeroes) {
  const NDArray<double, 1> a({0});
  const auto [mean, var] = meanVar(a);
  EXPECT_DOUBLE_EQ(mean, 0.0);
  EXPECT_DOUBLE_EQ(var, 0.0);
}

TEST(MeanVar, MatchesExpectedOnFixedVector) {
  const auto a = vec<double, 5>({1.0, 2.0, 3.0, 4.0, 5.0});
  const auto [mean, var] = meanVar(a);
  EXPECT_DOUBLE_EQ(mean, 3.0);
  // Population variance of {1..5} is 2.0 (sample variance would be 2.5).
  EXPECT_DOUBLE_EQ(var, 2.0);
}

TEST(MeanVar, SingleElementReturnsZeroVariance) {
  const auto a = vec<double, 1>({42.0});
  const auto [mean, var] = meanVar(a);
  EXPECT_DOUBLE_EQ(mean, 42.0);
  EXPECT_DOUBLE_EQ(var, 0.0);
}

TEST(MeanVar, WelfordBeatsNaiveOnPathologicalCase) {
  // Classic cancellation trap: v = [1e9+1, 1e9+2, 1e9+3] in f64.
  // Mean = 1e9 + 2. Population variance = 2/3. The naive E[x^2] - E[x]^2 form collapses to 0
  // because both expectations land in the same f64 ULP bucket.
  const std::array<double, 3> raw{1.0e9 + 1.0, 1.0e9 + 2.0, 1.0e9 + 3.0};
  NDArray<double, 1> a({3});
  for (std::size_t i = 0; i < 3; ++i) {
    a(i) = raw[i];
  }
  const auto [mean, var] = meanVar(a);
  EXPECT_NEAR(mean, 1.0e9 + 2.0, 1e-6);
  EXPECT_NEAR(var, 2.0 / 3.0, 1e-6);

  // Witness: the naive formula collapses on this input.
  double eX = 0.0;
  double eX2 = 0.0;
  for (const double v : raw) {
    eX += v;
    eX2 += v * v;
  }
  eX /= 3.0;
  eX2 /= 3.0;
  const double naiveVar = eX2 - (eX * eX);
  EXPECT_GT(std::fabs(naiveVar - (2.0 / 3.0)), 0.5)
      << "naive variance should be wildly wrong on this input; got " << naiveVar;
}

TEST(MeanVar, StridedInputMatchesContig) {
  NDArray<double, 2> a({4, 2});
  for (std::size_t i = 0; i < 4; ++i) {
    for (std::size_t j = 0; j < 2; ++j) {
      a[i][j] = static_cast<double>((i * 2) + j);
    }
  }
  const auto col = a.col(1); // values {1, 3, 5, 7}
  const auto [mean, var] = meanVar(col);
  EXPECT_DOUBLE_EQ(mean, 4.0);
  EXPECT_DOUBLE_EQ(var, 5.0);
}

TEST(ArgMin, ReturnsFirstIndexOfMinimum) {
  const auto a = vec<float, 8>({3.0F, 1.0F, 4.0F, 1.0F, 5.0F, 9.0F, 2.0F, 6.0F});
  EXPECT_EQ(argmin(a), 1u);
}

TEST(ArgMax, ReturnsFirstIndexOfMaximum) {
  const auto a = vec<float, 8>({3.0F, 1.0F, 4.0F, 1.0F, 5.0F, 9.0F, 2.0F, 6.0F});
  EXPECT_EQ(argmax(a), 5u);
}

TEST(ArgMin, SingleElementReturnsZero) {
  const auto a = vec<double, 1>({7.5});
  EXPECT_EQ(argmin(a), 0u);
}

TEST(ArgMax, SingleElementReturnsZero) {
  const auto a = vec<double, 1>({7.5});
  EXPECT_EQ(argmax(a), 0u);
}

TEST(ArgMin, TieBreaksByFirstIndex) {
  const auto a = vec<float, 5>({2.0F, 2.0F, 2.0F, 1.0F, 1.0F});
  EXPECT_EQ(argmin(a), 3u);
  const auto b = vec<float, 5>({1.0F, 1.0F, 2.0F, 2.0F, 1.0F});
  EXPECT_EQ(argmin(b), 0u);
}

TEST(ArgMin, StridedInputMatchesContig) {
  NDArray<double, 2> a({3, 2});
  a[0][0] = 5.0;
  a[0][1] = 9.0;
  a[1][0] = 2.0;
  a[1][1] = 4.0;
  a[2][0] = 7.0;
  a[2][1] = 1.0;
  const auto col1 = a.col(1); // values {9, 4, 1}
  EXPECT_EQ(argmin(col1), 2u);
  EXPECT_EQ(argmax(col1), 0u);
}

#ifndef NDEBUG
TEST(ArgMinDeathTest, AssertsOnEmptyInput) {
  const NDArray<float, 1> a({0});
  EXPECT_DEATH({ (void)argmin(a); }, "");
}

TEST(ArgMaxDeathTest, AssertsOnEmptyInput) {
  const NDArray<float, 1> a({0});
  EXPECT_DEATH({ (void)argmax(a); }, "");
}
#endif

TEST(TopK, ReturnsLargestKInDescendingOrder) {
  const auto a = vec<float, 7>({5.0F, 1.0F, 9.0F, 3.0F, 7.0F, 2.0F, 8.0F});
  std::array<std::size_t, 3> out{};
  topk(a, 3, std::span<std::size_t>(out));
  EXPECT_EQ(out[0], 2u);
  EXPECT_EQ(out[1], 6u);
  EXPECT_EQ(out[2], 4u);
}

TEST(TopK, KEqualsZeroIsNoop) {
  const auto a = vec<float, 4>({1.0F, 2.0F, 3.0F, 4.0F});
  std::array<std::size_t, 0> out{};
  topk(a, 0, std::span<std::size_t>(out));
  SUCCEED();
}

TEST(TopK, KEqualsNGivesFullPermutation) {
  const auto a = vec<double, 5>({4.0, 1.0, 3.0, 5.0, 2.0});
  std::array<std::size_t, 5> out{};
  topk(a, 5, std::span<std::size_t>(out));
  // Descending by value: 5.0 (idx 3), 4.0 (idx 0), 3.0 (idx 2), 2.0 (idx 4), 1.0 (idx 1).
  EXPECT_EQ(out[0], 3u);
  EXPECT_EQ(out[1], 0u);
  EXPECT_EQ(out[2], 2u);
  EXPECT_EQ(out[3], 4u);
  EXPECT_EQ(out[4], 1u);
}

TEST(TopK, TiesBrokenByIndexAscending) {
  const auto a = vec<float, 4>({5.0F, 5.0F, 5.0F, 5.0F});
  std::array<std::size_t, 2> out{};
  topk(a, 2, std::span<std::size_t>(out));
  EXPECT_EQ(out[0], 0u);
  EXPECT_EQ(out[1], 1u);
}

TEST(TopK, StridedInputMatchesContig) {
  NDArray<float, 2> a({4, 2});
  a[0][0] = 0.0F;
  a[0][1] = 9.0F;
  a[1][0] = 0.0F;
  a[1][1] = 2.0F;
  a[2][0] = 0.0F;
  a[2][1] = 7.0F;
  a[3][0] = 0.0F;
  a[3][1] = 4.0F;
  const auto col = a.col(1); // values {9, 2, 7, 4}
  std::array<std::size_t, 2> out{};
  topk(col, 2, std::span<std::size_t>(out));
  EXPECT_EQ(out[0], 0u); // 9 at position 0
  EXPECT_EQ(out[1], 2u); // 7 at position 2
}
