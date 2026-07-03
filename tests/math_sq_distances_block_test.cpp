#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include "clustering/math/detail/sq_distances_block.h"
#include "clustering/math/rng.h"

using clustering::math::detail::refreshMinSqAgainstRow;
using clustering::math::detail::sqDistancesAosBlock;
using clustering::math::detail::sqEuclideanRowPtr;

namespace {

// Random AoS points plus a query row drawn from the same range.
struct RefreshFixture {
  std::vector<float> points;
  std::vector<float> row;

  RefreshFixture(std::size_t count, std::size_t d) : points(count * d), row(d) {
    clustering::math::pcg64 rng;
    rng.seed(1234U);
    for (float &v : points) {
      v = (clustering::math::randUnit<float>(rng) * 200.0F) - 100.0F;
    }
    for (float &v : row) {
      v = (clustering::math::randUnit<float>(rng) * 200.0F) - 100.0F;
    }
  }
};

float relTolerance(float reference) {
  // FMA contraction differences between the batched kernel and the scalar
  // reference stay within a few ULP of the accumulated magnitude.
  return 1e-5F * (1.0F + std::abs(reference));
}

} // namespace

TEST(MathSqDistancesBlock, MatchesRowKernelAcrossShapes) {
  for (const std::size_t d : {8UL, 16UL, 24UL, 37UL}) {
    for (const std::size_t count : {1UL, 4UL, 5UL, 64UL, 129UL}) {
      const RefreshFixture fx(count, d);
      std::vector<float> out(count, -1.0F);
      sqDistancesAosBlock(fx.row.data(), fx.points.data(), count, d, out.data());
      for (std::size_t i = 0; i < count; ++i) {
        const float ref = sqEuclideanRowPtr(fx.points.data() + (i * d), fx.row.data(), d);
        EXPECT_NEAR(out[i], ref, relTolerance(ref)) << "d=" << d << " i=" << i;
      }
    }
  }
}

TEST(MathRefreshMinSq, MatchesScalarFoldAcrossShapes) {
  for (const std::size_t d : {1UL, 2UL, 3UL, 7UL, 8UL, 16UL, 37UL}) {
    for (const std::size_t count : {0UL, 1UL, 7UL, 8UL, 9UL, 63UL, 64UL, 65UL, 200UL}) {
      const RefreshFixture fx(count, d);
      clustering::math::pcg64 rng;
      rng.seed(99U);
      std::vector<float> minSq(count);
      for (float &v : minSq) {
        v = clustering::math::randUnit<float>(rng) * 50000.0F;
      }
      std::vector<float> expected = minSq;
      for (std::size_t i = 0; i < count; ++i) {
        const float cand = sqEuclideanRowPtr(fx.points.data() + (i * d), fx.row.data(), d);
        expected[i] = std::min(expected[i], cand);
      }
      std::vector<float> actual = minSq;
      refreshMinSqAgainstRow(fx.row.data(), fx.points.data(), count, d, actual.data());
      for (std::size_t i = 0; i < count; ++i) {
        EXPECT_NEAR(actual[i], expected[i], relTolerance(expected[i]))
            << "d=" << d << " count=" << count << " i=" << i;
        EXPECT_LE(actual[i], minSq[i] + relTolerance(minSq[i]))
            << "refresh must never increase the running minimum";
      }
    }
  }
}

TEST(MathRefreshMinSq, InfinityBaselineYieldsDistances) {
  constexpr std::size_t kCount = 51;
  constexpr std::size_t kDim = 2;
  const RefreshFixture fx(kCount, kDim);
  std::vector<float> minSq(kCount, std::numeric_limits<float>::infinity());
  refreshMinSqAgainstRow(fx.row.data(), fx.points.data(), kCount, kDim, minSq.data());
  for (std::size_t i = 0; i < kCount; ++i) {
    const float ref = sqEuclideanRowPtr(fx.points.data() + (i * kDim), fx.row.data(), kDim);
    EXPECT_NEAR(minSq[i], ref, relTolerance(ref)) << "i=" << i;
  }
}
