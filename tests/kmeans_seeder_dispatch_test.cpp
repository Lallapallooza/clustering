#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

#include "clustering/kmeans.h"
#include "clustering/kmeans/policy/afkmc2_seeder.h"
#include "clustering/kmeans/policy/auto_seeder.h"
#include "clustering/kmeans/policy/greedy_kmpp_seeder.h"
#include "clustering/kmeans/policy/lloyd.h"
#include "clustering/kmeans/policy/lloyd_fused_gemm.h"
#include "clustering/kmeans/policy/seeder.h"
#include "clustering/ndarray.h"

using clustering::KMeans;
using clustering::NDArray;
using clustering::kmeans::AfkMc2Seeder;
using clustering::kmeans::AutoSeeder;
using clustering::kmeans::GreedyKmppSeeder;
using clustering::kmeans::LloydFusedGemm;

namespace {

struct Blobs {
  NDArray<float, 2> X;
  std::vector<std::int32_t> truth;
};

Blobs makeBlobs(std::size_t n, std::size_t d, std::size_t k, float sigma, std::uint32_t seed) {
  NDArray<float, 2> X({n, d});
  std::vector<std::int32_t> truth(n, 0);
  std::mt19937 gen(seed);

  std::vector<std::vector<float>> centers(k, std::vector<float>(d, 0.0F));
  for (std::size_t c = 0; c < k; ++c) {
    centers[c][0] = static_cast<float>(c * 50);
  }

  std::normal_distribution<float> noise(0.0F, sigma);
  std::uniform_int_distribution<std::int32_t> pickCluster(0, static_cast<std::int32_t>(k) - 1);
  for (std::size_t i = 0; i < n; ++i) {
    const std::int32_t c = pickCluster(gen);
    truth[i] = c;
    for (std::size_t t = 0; t < d; ++t) {
      X(i, t) = centers[static_cast<std::size_t>(c)][t] + noise(gen);
    }
  }

  return Blobs{.X = std::move(X), .truth = std::move(truth)};
}

} // namespace

static_assert(clustering::kmeans::LloydStrategy<LloydFusedGemm<float>, float>);
static_assert(clustering::kmeans::SeederStrategy<GreedyKmppSeeder<float>, float>);
static_assert(clustering::kmeans::SeederStrategy<AfkMc2Seeder<float>, float>);

// Below the AFK-MC2 k-floor, a pinned @c AfkMc2Seeder must silently delegate to greedy so the
// output matches a pinned @c GreedyKmppSeeder at the same seed + data. This pins the HEAD-era
// public contract under which forcing AFK-MC2 at low @c k produced greedy output.
TEST(AfkMc2, LowKFallsThroughToGreedyOutput) {
  constexpr std::size_t n = 1200;
  constexpr std::size_t d = 4;
  constexpr std::size_t k = 8;
  ASSERT_LT(k, AfkMc2Seeder<float>::kFloor);
  const Blobs b = makeBlobs(n, d, k, 1.0F, 7U);

  KMeans<float, LloydFusedGemm<float>, AfkMc2Seeder<float>> kmAfk(k, 1);
  kmAfk.run(b.X, 100, 1e-4F, 42U);

  KMeans<float, LloydFusedGemm<float>, GreedyKmppSeeder<float>> kmG(k, 1);
  kmG.run(b.X, 100, 1e-4F, 42U);

  ASSERT_EQ(kmAfk.labels().dim(0), kmG.labels().dim(0));
  for (std::size_t i = 0; i < n; ++i) {
    EXPECT_EQ(kmAfk.labels()(i), kmG.labels()(i)) << "label " << i << " diverges";
  }
  ASSERT_EQ(kmAfk.centroids().dim(0), k);
  ASSERT_EQ(kmAfk.centroids().dim(1), d);
  for (std::size_t c = 0; c < k; ++c) {
    for (std::size_t t = 0; t < d; ++t) {
      EXPECT_EQ(std::bit_cast<std::uint32_t>(kmAfk.centroids()(c, t)),
                std::bit_cast<std::uint32_t>(kmG.centroids()(c, t)))
          << "centroid (" << c << ", " << t << ") diverges";
    }
  }
  EXPECT_EQ(std::bit_cast<std::uint64_t>(kmAfk.inertia()),
            std::bit_cast<std::uint64_t>(kmG.inertia()));
}

// At @c k >= kFloor, AFK-MC2's log-k guarantee tracks greedy best-of-3 within 5% on
// well-separated blobs. A wider gap would signal the Markov chain is stalling.
TEST(AfkMc2, InertiaWithinRelaxedBoundVsGreedy) {
  constexpr std::size_t n = 500000;
  constexpr std::size_t d = 4;
  constexpr std::size_t k = 100;
  const Blobs b = makeBlobs(n, d, k, 0.8F, 99U);

  KMeans<float, LloydFusedGemm<float>, AfkMc2Seeder<float>> kmAfk(k, 1);
  kmAfk.run(b.X, 30, 1e-3F, 42U);
  const double afkInertia = kmAfk.inertia();

  double greedyBest = std::numeric_limits<double>::infinity();
  for (const std::uint64_t s : {1ULL, 2ULL, 3ULL}) {
    KMeans<float, LloydFusedGemm<float>, GreedyKmppSeeder<float>> kmG(k, 1);
    kmG.run(b.X, 30, 1e-3F, s);
    greedyBest = std::min(greedyBest, kmG.inertia());
  }

  const double rel = (afkInertia - greedyBest) / std::max(1.0, greedyBest);
  EXPECT_LT(rel, 0.05) << "afk=" << afkInertia << " greedyBest=" << greedyBest << " rel=" << rel;
}

// Same seed + nJobs must produce bit-identical centroids + labels + inertia across five fits
// when the AFK-MC2 seeder is pinned. The chain draws from the PRNG in a fixed order regardless
// of acceptance-branch outcomes.
TEST(AfkMc2, DeterministicAcrossRepeatedRuns) {
  constexpr std::size_t n = 600000;
  constexpr std::size_t d = 4;
  constexpr std::size_t k = 128;
  const Blobs b = makeBlobs(n, d, k, 0.8F, 31U);

  constexpr std::size_t nJobs = 4;
  constexpr std::uint64_t seed = 2025U;
  constexpr std::size_t maxIter = 15;
  constexpr float tol = 1e-3F;

  KMeans<float, LloydFusedGemm<float>, AfkMc2Seeder<float>> km0(k, nJobs);
  km0.run(b.X, maxIter, tol, seed);
  const std::vector<std::int32_t> refLabels(km0.labels().data(), km0.labels().data() + n);
  const std::vector<float> refCentroids(km0.centroids().data(), km0.centroids().data() + (k * d));
  const double refInertia = km0.inertia();

  for (int rep = 1; rep < 5; ++rep) {
    KMeans<float, LloydFusedGemm<float>, AfkMc2Seeder<float>> km(k, nJobs);
    km.run(b.X, maxIter, tol, seed);
    const std::vector<std::int32_t> labels(km.labels().data(), km.labels().data() + n);
    ASSERT_EQ(refLabels, labels) << "afkmc2 labels diverge at repetition " << rep;
    for (std::size_t j = 0; j < refCentroids.size(); ++j) {
      ASSERT_EQ(std::bit_cast<std::uint32_t>(refCentroids[j]),
                std::bit_cast<std::uint32_t>(km.centroids().data()[j]))
          << "afkmc2 centroid[" << j << "] diverges at repetition " << rep;
    }
    ASSERT_EQ(std::bit_cast<std::uint64_t>(refInertia), std::bit_cast<std::uint64_t>(km.inertia()));
  }
}

// k=1 is below @c kFloor so a pinned AFK-MC2 seeder falls through to greedy; the result is the
// data mean per the k=1 Lloyd contract.
TEST(AfkMc2, AtK1ProducesOneCentroid) {
  constexpr std::size_t n = 100;
  constexpr std::size_t d = 4;
  NDArray<float, 2> X({n, d});
  std::mt19937 gen(17U);
  std::normal_distribution<float> dist(0.0F, 1.0F);
  std::array<double, d> sum{};
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t t = 0; t < d; ++t) {
      const float v = dist(gen);
      X(i, t) = v;
      sum[t] += static_cast<double>(v);
    }
  }
  KMeans<float, LloydFusedGemm<float>, AfkMc2Seeder<float>> km(1, 1);
  km.run(X, 50, 1e-4F, 99U);
  for (std::size_t i = 0; i < n; ++i) {
    EXPECT_EQ(km.labels()(i), 0);
  }
  for (std::size_t t = 0; t < d; ++t) {
    const auto expected = static_cast<float>(sum[t] / static_cast<double>(n));
    EXPECT_NEAR(km.centroids()(0, t), expected, 1e-4F);
  }
}

// Pinning the seeder to the greedy policy at a small-envelope shape must match the default
// @c KMeans<float> output bit-for-bit: below the auto-dispatch thresholds, @c AutoSeeder<T>::pick
// returns the greedy alternative, so the two spellings resolve to identical runtime behavior.
TEST(AutoSeeder, DefaultResolvesToGreedyAtSmallEnvelope) {
  constexpr std::size_t n = 1000;
  constexpr std::size_t d = 4;
  constexpr std::size_t k = 8;
  ASSERT_LT(n, AutoSeeder<float>::afkmc2NThreshold);
  const Blobs b = makeBlobs(n, d, k, 0.5F, 11U);

  KMeans<float> kmDefault(k, 1);
  kmDefault.run(b.X, 100, 1e-4F, 42U);

  KMeans<float, LloydFusedGemm<float>, GreedyKmppSeeder<float>> kmPinned(k, 1);
  kmPinned.run(b.X, 100, 1e-4F, 42U);

  ASSERT_EQ(kmDefault.labels().dim(0), kmPinned.labels().dim(0));
  for (std::size_t i = 0; i < n; ++i) {
    EXPECT_EQ(kmDefault.labels()(i), kmPinned.labels()(i)) << "label " << i << " diverges";
  }
  for (std::size_t c = 0; c < k; ++c) {
    for (std::size_t t = 0; t < d; ++t) {
      EXPECT_EQ(std::bit_cast<std::uint32_t>(kmDefault.centroids()(c, t)),
                std::bit_cast<std::uint32_t>(kmPinned.centroids()(c, t)))
          << "centroid (" << c << ", " << t << ") diverges";
    }
  }
  EXPECT_EQ(std::bit_cast<std::uint64_t>(kmDefault.inertia()),
            std::bit_cast<std::uint64_t>(kmPinned.inertia()));
}
