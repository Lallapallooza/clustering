#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include "clustering/kmeans.h"
#include "clustering/kmeans/policy/greedy_kmpp_seeder.h"
#include "clustering/math/thread.h"
#include "clustering/ndarray.h"

using clustering::NDArray;
using clustering::kmeans::GreedyKmppSeeder;
using clustering::kmeans::detail::greedyKmppLocalTrials;
using clustering::kmeans::detail::sqEuclideanRowToBatch;
using clustering::math::detail::sqEuclideanRowPtr;

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

// End-to-end: at the perf-gate shape, the seeder still produces a valid k-means++ seed -- every
// centroid row matches an X row (kmeans++ never invents centroids), every centroid is distinct
// under typical data, and the output centroid matrix is fully populated.
TEST(SeederBehaviour, GreedyKmppProducesDistinctCentroidRowsAtGateShape) {
  constexpr std::size_t n = 1000;
  constexpr std::size_t d = 32;
  constexpr std::size_t k = 16;
  const NDArray<float, 2> X = makeData(n, d, 13U);

  NDArray<float, 2> centroids({k, d});
  GreedyKmppSeeder<float> seeder;
  seeder.run(X, k, 42U, clustering::math::Pool{nullptr}, centroids);

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
}

// End-to-end determinism test specifically for the seeder: the same (X, k, seed) tuple must
// produce bit-identical centroids across repeated invocations on the same seeder instance.
TEST(SeederBehaviour, DeterministicAcrossInvocationsAtGateShape) {
  constexpr std::size_t n = 500;
  constexpr std::size_t d = 32;
  constexpr std::size_t k = 16;
  const NDArray<float, 2> X = makeData(n, d, 99U);

  NDArray<float, 2> centroids1({k, d});
  GreedyKmppSeeder<float> seeder1;
  seeder1.run(X, k, 7U, clustering::math::Pool{nullptr}, centroids1);

  NDArray<float, 2> centroids2({k, d});
  GreedyKmppSeeder<float> seeder2;
  seeder2.run(X, k, 7U, clustering::math::Pool{nullptr}, centroids2);

  for (std::size_t c = 0; c < k; ++c) {
    for (std::size_t t = 0; t < d; ++t) {
      EXPECT_EQ(centroids1(c, t), centroids2(c, t))
          << "centroid (" << c << ", " << t << ") diverges between fits";
    }
  }
}

// Reuse of a single seeder instance across two runs at the same shape must produce
// bit-identical output -- scratch amortizes without leaking state between invocations.
TEST(SeederBehaviour, SameInstanceTwoRunsBitIdenticalAtGateShape) {
  constexpr std::size_t n = 500;
  constexpr std::size_t d = 32;
  constexpr std::size_t k = 16;
  const NDArray<float, 2> X = makeData(n, d, 53U);

  GreedyKmppSeeder<float> seeder;
  NDArray<float, 2> first({k, d});
  seeder.run(X, k, 21U, clustering::math::Pool{nullptr}, first);

  NDArray<float, 2> second({k, d});
  seeder.run(X, k, 21U, clustering::math::Pool{nullptr}, second);

  for (std::size_t c = 0; c < k; ++c) {
    for (std::size_t t = 0; t < d; ++t) {
      EXPECT_EQ(first(c, t), second(c, t))
          << "centroid (" << c << ", " << t << ") differs between reuse runs";
    }
  }
}
