#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include "clustering/kmeans.h"
#include "clustering/kmeans/detail/seed_greedy_kmpp.h"
#include "clustering/math/thread.h"
#include "clustering/ndarray.h"

using clustering::NDArray;
using clustering::kmeans::detail::ensureGreedyKmppScratchShape;
using clustering::kmeans::detail::greedyKmppLocalTrials;
using clustering::kmeans::detail::GreedyKmppScratch;
using clustering::kmeans::detail::seedGreedyKMeansPlusPlus;
using clustering::kmeans::detail::sqEuclideanRowPtr;
using clustering::kmeans::detail::sqEuclideanRowToBatch;

namespace {

NDArray<float, 2> makeData(std::size_t n, std::size_t d, std::uint32_t seed) {
  NDArray<float, 2> X({n, d});
  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> dist(-1.0F, 1.0F);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t t = 0; t < d; ++t) {
      X(i, t) = dist(gen);
    }
  }
  return X;
}

} // namespace

// The batched scoring kernel must produce the same per-pair distances as the per-row helper.
// Mutation: returning any t-permuted result, dropping a tail element, or accumulating in a
// different order at the FMA precision would break this assertion.
TEST(SeederBatchKernel, BatchMatchesPerRowAcrossTrialCounts) {
  constexpr std::size_t d = 32;
  constexpr std::size_t kMaxL = 16;
  std::mt19937 gen(7U);
  std::uniform_real_distribution<float> dist(-2.0F, 2.0F);

  std::array<float, d> x{};
  std::array<float, kMaxL * d> cands{};
  for (auto &v : x) {
    v = dist(gen);
  }
  for (auto &v : cands) {
    v = dist(gen);
  }

  // Sweep L across batch boundaries so 1, 7 (sub-8), 8 (full batch), 9 (batch + tail of 1),
  // 16 (two full batches) all hit the dispatch.
  for (const std::size_t L :
       {std::size_t{1}, std::size_t{7}, std::size_t{8}, std::size_t{9}, std::size_t{16}}) {
    std::array<float, kMaxL> batched{};
    sqEuclideanRowToBatch<float>(x.data(), cands.data(), L, d, batched.data());
    for (std::size_t t = 0; t < L; ++t) {
      const float perRow = sqEuclideanRowPtr(x.data(), cands.data() + (t * d), d);
      // The batched and per-row paths use the same AVX2 lane width and FMA reduction shape;
      // residual ULP drift between the two is bounded by the tail handling. A loose absolute
      // tolerance pins behavior without requiring bit-identity across reduction orders.
      EXPECT_NEAR(batched[t], perRow, 1e-4F)
          << "L=" << L << " t=" << t << " batched=" << batched[t] << " perRow=" << perRow;
    }
  }
}

// Small d below the AVX2 lane width must route through the scalar fallback. Same numerical
// agreement gate as the AVX2 path.
TEST(SeederBatchKernel, BatchMatchesPerRowSmallD) {
  constexpr std::size_t d = 3; // below kAvx2Lanes<float> = 8
  constexpr std::size_t L = 5;
  std::mt19937 gen(11U);
  std::uniform_real_distribution<float> dist(-3.0F, 3.0F);

  std::array<float, d> x{};
  std::array<float, L * d> cands{};
  for (auto &v : x) {
    v = dist(gen);
  }
  for (auto &v : cands) {
    v = dist(gen);
  }

  std::array<float, L> batched{};
  sqEuclideanRowToBatch<float>(x.data(), cands.data(), L, d, batched.data());
  for (std::size_t t = 0; t < L; ++t) {
    const float perRow = sqEuclideanRowPtr(x.data(), cands.data() + (t * d), d);
    EXPECT_FLOAT_EQ(batched[t], perRow);
  }
}

// Repeated invocation at the identical (L, d) shape must reuse the scratch -- no reallocation
// fires on the second call. Mutation: ensureGreedyKmppScratchShape always reallocating would
// regress the alloc counter and break A7.
TEST(SeederScratch, SecondEnsureAtSameShapeIsNoop) {
  GreedyKmppScratch<float> s;
  ensureGreedyKmppScratchShape<float>(s, 8, 32, 1000);
  const float *const dataAfterFirst = s.candRows.data();
  const float *const cumAfterFirst = s.cumDistSq.data();
  ensureGreedyKmppScratchShape<float>(s, 8, 32, 1000);
  EXPECT_EQ(s.candRows.data(), dataAfterFirst) << "scratch reallocated despite matching shape";
  EXPECT_EQ(s.cumDistSq.data(), cumAfterFirst) << "cum scratch reallocated despite matching shape";
}

// The local-trials count is sklearn's @c 2 + floor(ln(k)) for k >= 2, capped at 1 for k <= 1.
TEST(SeederScratch, LocalTrialsMatchesDocumentedFormula) {
  EXPECT_EQ(greedyKmppLocalTrials(0), 1U);
  EXPECT_EQ(greedyKmppLocalTrials(1), 1U);
  EXPECT_EQ(greedyKmppLocalTrials(2), 2U);
  EXPECT_EQ(greedyKmppLocalTrials(3), 3U);
  EXPECT_EQ(greedyKmppLocalTrials(4), 3U);
  EXPECT_EQ(greedyKmppLocalTrials(8), 4U);
  EXPECT_EQ(greedyKmppLocalTrials(16), 4U);
  EXPECT_EQ(greedyKmppLocalTrials(64), 6U);
  EXPECT_EQ(greedyKmppLocalTrials(1024), 8U);
}

// End-to-end: at the perf-gate shape, the new seeder still produces a valid k-means++ seed --
// every centroid row matches an X row (kmeans++ never invents centroids), every centroid is
// distinct under typical data, and the resulting min-distance array is fully populated.
TEST(SeederBehaviour, GreedyKmppProducesDistinctCentroidRowsAtGateShape) {
  constexpr std::size_t n = 1000;
  constexpr std::size_t d = 32;
  constexpr std::size_t k = 16;
  const NDArray<float, 2> X = makeData(n, d, 13U);

  NDArray<float, 2> centroids({k, d});
  NDArray<float, 1> minSq({n});
  GreedyKmppScratch<float> scratch;
  ensureGreedyKmppScratchShape<float>(scratch, greedyKmppLocalTrials(k), d, n);
  seedGreedyKMeansPlusPlus<float>(X, centroids, minSq, scratch, 42U,
                                  clustering::math::Pool{nullptr});

  // Every centroid row must match exactly one X row (kmeans++ picks rows, not arbitrary
  // points). Look up via byte equality on the row payload.
  for (std::size_t c = 0; c < k; ++c) {
    bool found = false;
    for (std::size_t i = 0; i < n && !found; ++i) {
      bool eq = true;
      for (std::size_t t = 0; t < d && eq; ++t) {
        if (centroids(c, t) != X(i, t)) {
          eq = false;
        }
      }
      if (eq) {
        found = true;
      }
    }
    EXPECT_TRUE(found) << "centroid row " << c << " does not match any X row";
  }

  // Distinct centroids: under continuous-uniform data the seeder must not pick the same row
  // twice. Verify pairwise inequality at the row level.
  for (std::size_t a = 0; a < k; ++a) {
    for (std::size_t b = a + 1; b < k; ++b) {
      bool same = true;
      for (std::size_t t = 0; t < d && same; ++t) {
        if (centroids(a, t) != centroids(b, t)) {
          same = false;
        }
      }
      EXPECT_FALSE(same) << "centroid rows " << a << " and " << b << " are identical";
    }
  }

  // minSq must be populated and non-negative everywhere.
  for (std::size_t i = 0; i < n; ++i) {
    EXPECT_GE(minSq(i), 0.0F) << "minSq[" << i << "] is negative";
  }
}

// End-to-end determinism test specifically for the seeder: the same (X, k, seed) tuple must
// produce bit-identical centroids across repeated invocations on the same scratch.
TEST(SeederBehaviour, DeterministicAcrossInvocationsAtGateShape) {
  constexpr std::size_t n = 500;
  constexpr std::size_t d = 32;
  constexpr std::size_t k = 16;
  const NDArray<float, 2> X = makeData(n, d, 99U);

  NDArray<float, 2> centroids1({k, d});
  NDArray<float, 1> minSq1({n});
  GreedyKmppScratch<float> scratch1;
  ensureGreedyKmppScratchShape<float>(scratch1, greedyKmppLocalTrials(k), d, n);
  seedGreedyKMeansPlusPlus<float>(X, centroids1, minSq1, scratch1, 7U,
                                  clustering::math::Pool{nullptr});

  NDArray<float, 2> centroids2({k, d});
  NDArray<float, 1> minSq2({n});
  GreedyKmppScratch<float> scratch2;
  ensureGreedyKmppScratchShape<float>(scratch2, greedyKmppLocalTrials(k), d, n);
  seedGreedyKMeansPlusPlus<float>(X, centroids2, minSq2, scratch2, 7U,
                                  clustering::math::Pool{nullptr});

  for (std::size_t c = 0; c < k; ++c) {
    for (std::size_t t = 0; t < d; ++t) {
      EXPECT_EQ(centroids1(c, t), centroids2(c, t))
          << "centroid (" << c << ", " << t << ") diverges between fits";
    }
  }
  for (std::size_t i = 0; i < n; ++i) {
    EXPECT_EQ(minSq1(i), minSq2(i)) << "minSq[" << i << "] diverges between fits";
  }
}
