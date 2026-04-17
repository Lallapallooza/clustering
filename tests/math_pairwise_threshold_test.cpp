#include <gtest/gtest.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <random>
#include <set>
#include <utility>
#include <vector>

#include "clustering/math/pairwise.h"
#include "clustering/math/thread.h"
#include "clustering/ndarray.h"

using clustering::NDArray;
using clustering::math::pairwiseSqEuclideanThresholded;
using clustering::math::Pool;

namespace {

template <class T> void fillRandom(NDArray<T, 2> &a, std::uint32_t seed) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<T> dist(static_cast<T>(-1.0), static_cast<T>(1.0));
  const std::size_t n = a.dim(0);
  const std::size_t d = a.dim(1);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < d; ++j) {
      a[i][j] = dist(gen);
    }
  }
}

template <class T>
std::set<std::pair<std::size_t, std::size_t>>
scalarOraclePairs(const NDArray<T, 2> &X, const NDArray<T, 2> &Y, T radiusSq) {
  std::set<std::pair<std::size_t, std::size_t>> out;
  const std::size_t n = X.dim(0);
  const std::size_t m = Y.dim(0);
  const std::size_t d = X.dim(1);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < m; ++j) {
      T s = T{0};
      for (std::size_t k = 0; k < d; ++k) {
        const T diff = X(i, k) - Y(j, k);
        s += diff * diff;
      }
      if (s <= radiusSq) {
        out.emplace(i, j);
      }
    }
  }
  return out;
}

} // namespace

TEST(PairwiseThresholdAgreement, FusedMatchesOracleAcrossEnvelope) {
  // Fixture sweep spanning both sides of the chunk-row boundary (256) and the partial-tile /
  // partial-panel tails (n, m not multiples of Mr=8 or Nr=6).
  const std::vector<std::size_t> ns{64, 137, 512, 2000};
  const std::vector<std::size_t> ds{16, 32, 64, 128};
  constexpr std::size_t m = 400; // clean partial panels regardless of n

  std::uint32_t seed = 1000U;
  for (const std::size_t n : ns) {
    for (const std::size_t d : ds) {
      NDArray<float, 2> X({n, d});
      NDArray<float, 2> Y({m, d});
      fillRandom(X, seed++);
      fillRandom(Y, seed++);

      // Pick a radius that selects ~15% survivors on a uniform fixture in [-1, 1]. A smaller
      // radius picks up zero, a larger one saturates -- both are covered by dedicated tests
      // below.
      const float radiusSq = 0.5f * static_cast<float>(d);

      std::set<std::pair<std::size_t, std::size_t>> fusedPairs;
      auto emit = [&fusedPairs](std::size_t r, std::size_t c) { fusedPairs.emplace(r, c); };
      pairwiseSqEuclideanThresholded<float>(X, Y, radiusSq, Pool{nullptr}, emit);

      const auto oracle = scalarOraclePairs<float>(X, Y, radiusSq);
      EXPECT_EQ(fusedPairs, oracle) << "mismatch at n=" << n << " d=" << d;
    }
  }
}

TEST(PairwiseThresholdAgreement, EmptyRadiusNeverEmits) {
  constexpr std::size_t n = 64;
  constexpr std::size_t m = 128;
  constexpr std::size_t d = 32;
  NDArray<float, 2> X({n, d});
  NDArray<float, 2> Y({m, d});
  fillRandom(X, 10U);
  fillRandom(Y, 20U);

  // Shift Y far away so no pair survives even a zero radius. Adding 1000 to every cell pushes
  // the minimum squared distance well above 10^4 at d >= 16.
  for (std::size_t j = 0; j < m; ++j) {
    for (std::size_t k = 0; k < d; ++k) {
      Y(j, k) = Y(j, k) + 1000.0f;
    }
  }

  std::atomic<std::size_t> emitCount{0};
  auto emit = [&emitCount](std::size_t, std::size_t) {
    emitCount.fetch_add(1, std::memory_order_relaxed);
  };
  pairwiseSqEuclideanThresholded<float>(X, Y, 0.0f, Pool{nullptr}, emit);
  EXPECT_EQ(emitCount.load(), std::size_t{0});
}

TEST(PairwiseThresholdAgreement, RadiusLargerThanDiameterEmitsEveryPair) {
  constexpr std::size_t n = 40;
  constexpr std::size_t m = 35;
  constexpr std::size_t d = 32;
  NDArray<float, 2> X({n, d});
  NDArray<float, 2> Y({m, d});
  fillRandom(X, 30U);
  fillRandom(Y, 40U);

  // Each coordinate is in [-1, 1], so the max squared distance per dimension is 4 and the max
  // squared pair distance is 4 * d. A radius whose square is 10 * d is a generous cushion.
  const float radiusSq = 10.0f * static_cast<float>(d);
  std::size_t count = 0;
  auto emit = [&count](std::size_t, std::size_t) { ++count; };
  pairwiseSqEuclideanThresholded<float>(X, Y, radiusSq, Pool{nullptr}, emit);
  EXPECT_EQ(count, n * m);
}

TEST(PairwiseThresholdAgreement, PartialTileAndPartialPanelTails) {
  // n = 13, m = 11, d = 16: forces both an incomplete M-tile (13 % 8 = 5) and an incomplete
  // N-panel (11 % 6 = 5). Padded rows and padded columns must not show up in emit output.
  constexpr std::size_t n = 13;
  constexpr std::size_t m = 11;
  constexpr std::size_t d = 16;
  NDArray<float, 2> X({n, d});
  NDArray<float, 2> Y({m, d});
  fillRandom(X, 101U);
  fillRandom(Y, 102U);

  const float radiusSq = 0.5f * static_cast<float>(d);
  std::set<std::pair<std::size_t, std::size_t>> pairs;
  auto emit = [&pairs, n, m](std::size_t r, std::size_t c) {
    EXPECT_LT(r, n);
    EXPECT_LT(c, m);
    pairs.emplace(r, c);
  };
  pairwiseSqEuclideanThresholded<float>(X, Y, radiusSq, Pool{nullptr}, emit);

  const auto oracle = scalarOraclePairs<float>(X, Y, radiusSq);
  EXPECT_EQ(pairs, oracle);
}

TEST(PairwiseThresholdAlloc, FusedKernelEmitIsAllocationFreeOnPreReservedOutput) {
  // After a warm-up call to prime the outer driver's packing buffers' cache state, a second
  // call with a pre-reserved output vector must emit every surviving pair without driving any
  // unexpected per-emit allocation. The outer driver's per-call packing scratch is a fixed
  // O(m*d + m) cost that is not per-emit; the check here isolates the kernel emit path.
  constexpr std::size_t n = 128;
  constexpr std::size_t m = 256;
  constexpr std::size_t d = 32;
  NDArray<float, 2> X({n, d});
  NDArray<float, 2> Y({m, d});
  fillRandom(X, 99U);
  fillRandom(Y, 98U);

  // Radius large enough that ~every pair survives -- worst case for emit volume.
  const float radiusSq = 10.0f * static_cast<float>(d);

  // First, compute the true survivor count so the caller can pre-reserve exactly.
  std::size_t expectedCount = 0;
  {
    auto emit = [&expectedCount](std::size_t, std::size_t) { ++expectedCount; };
    pairwiseSqEuclideanThresholded<float>(X, Y, radiusSq, Pool{nullptr}, emit);
  }
  EXPECT_GT(expectedCount, std::size_t{0});

  std::vector<std::pair<std::size_t, std::size_t>> out;
  out.reserve(expectedCount);

  auto &counter = clustering::detail::alignedAllocCallCount();
  const std::uint64_t before = counter.load(std::memory_order_relaxed);
  auto emit = [&out](std::size_t r, std::size_t c) { out.emplace_back(r, c); };
  pairwiseSqEuclideanThresholded<float>(X, Y, radiusSq, Pool{nullptr}, emit);
  const std::uint64_t after = counter.load(std::memory_order_relaxed);

  EXPECT_EQ(out.size(), expectedCount);
  // The outer driver allocates four aligned vectors per call at steady state: packed Y,
  // packed Y-norms, X row-norms, and Y row-norms. The bound is a small constant independent
  // of survivor count, so the emit path itself does not scale allocations with result size.
  EXPECT_LE(after - before, std::uint64_t{4})
      << "unexpected " << (after - before) << " aligned allocations during emit";
}
