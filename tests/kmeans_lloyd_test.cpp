#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

#include "clustering/kmeans.h"
#include "clustering/kmeans/policy/lloyd_fused_gemm.h"
#include "clustering/math/defaults.h"
#include "clustering/ndarray.h"

using clustering::KMeans;
using clustering::NDArray;

namespace {

// Generate isotropic-gaussian blobs separable at sigma << inter-center distance. The first
// returned array is the data matrix; the second is the truth label per point, so label recovery
// can be scored up to cluster-permutation via majority vote.
struct Blobs {
  NDArray<float, 2> X;
  std::vector<std::int32_t> truth;
};

Blobs makeBlobs(std::size_t n, std::size_t d, std::size_t k, float sigma, std::uint32_t seed) {
  NDArray<float, 2> X({n, d});
  std::vector<std::int32_t> truth(n, 0);
  std::mt19937 gen(seed);

  // Place cluster centers along the first axis at {0, 50, 100, ...} so every pair is at least
  // 50 apart -- orders of magnitude above sigma = 0.5 so the blobs are cleanly separable.
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

// Score cluster-permutation-invariant purity: for each predicted cluster, count its most common
// truth label; sum majority counts / n. At perfect recovery this is 1.0.
double purity(const NDArray<std::int32_t, 1> &pred, const std::vector<std::int32_t> &truth,
              std::size_t k) {
  const std::size_t n = pred.dim(0);
  std::vector<std::vector<std::size_t>> confusion(k, std::vector<std::size_t>(k, 0));
  for (std::size_t i = 0; i < n; ++i) {
    const auto p = static_cast<std::size_t>(pred(i));
    const auto tLabel = static_cast<std::size_t>(truth[i]);
    ++confusion[p][tLabel];
  }
  std::size_t majority = 0;
  for (std::size_t p = 0; p < k; ++p) {
    majority += *std::max_element(confusion[p].begin(), confusion[p].end());
  }
  return static_cast<double>(majority) / static_cast<double>(n);
}

// Reference Lloyd: textbook assignment / update with uniform-random init. Used to gate inertia
// within 1% of the best of 3 independent seeds.
double referenceLloydInertia(const NDArray<float, 2> &X, std::size_t k, std::uint64_t seed,
                             std::size_t maxIter) {
  const std::size_t n = X.dim(0);
  const std::size_t d = X.dim(1);
  std::mt19937_64 gen(seed);
  std::uniform_int_distribution<std::size_t> pick(0, n - 1);

  std::vector<float> centroids(k * d, 0.0F);
  std::vector<bool> chosen(n, false);
  for (std::size_t c = 0; c < k; ++c) {
    std::size_t idx = pick(gen);
    while (chosen[idx]) {
      idx = pick(gen);
    }
    chosen[idx] = true;
    for (std::size_t t = 0; t < d; ++t) {
      centroids[(c * d) + t] = X(idx, t);
    }
  }

  std::vector<std::size_t> labels(n, 0);
  std::vector<float> sums(k * d, 0.0F);
  std::vector<std::size_t> counts(k, 0);

  for (std::size_t iter = 0; iter < maxIter; ++iter) {
    for (std::size_t i = 0; i < n; ++i) {
      std::size_t best = 0;
      float bestVal = std::numeric_limits<float>::infinity();
      for (std::size_t c = 0; c < k; ++c) {
        float s = 0.0F;
        for (std::size_t t = 0; t < d; ++t) {
          const float diff = X(i, t) - centroids[(c * d) + t];
          s += diff * diff;
        }
        if (s < bestVal) {
          bestVal = s;
          best = c;
        }
      }
      labels[i] = best;
    }

    std::fill(sums.begin(), sums.end(), 0.0F);
    std::fill(counts.begin(), counts.end(), 0U);
    for (std::size_t i = 0; i < n; ++i) {
      const std::size_t c = labels[i];
      ++counts[c];
      for (std::size_t t = 0; t < d; ++t) {
        sums[(c * d) + t] += X(i, t);
      }
    }
    for (std::size_t c = 0; c < k; ++c) {
      if (counts[c] == 0) {
        continue;
      }
      const float inv = 1.0F / static_cast<float>(counts[c]);
      for (std::size_t t = 0; t < d; ++t) {
        centroids[(c * d) + t] = sums[(c * d) + t] * inv;
      }
    }
  }

  double total = 0.0;
  double comp = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const std::size_t c = labels[i];
    double s = 0.0;
    for (std::size_t t = 0; t < d; ++t) {
      const double diff =
          static_cast<double>(X(i, t)) - static_cast<double>(centroids[(c * d) + t]);
      s += diff * diff;
    }
    const double y = s - comp;
    const double tt = total + y;
    comp = (tt - total) - y;
    total = tt;
  }
  return total;
}

} // namespace

// Exercises the Elkan prune path (d > pairwiseArgminMaxD and k > kHamerlyMaxK). At this shape
// the bound matrix is @c n * k floats, populated on the first full assignment and maintained
// across iterations. Recovery of well-separated blobs is the correctness gate.
TEST(KMeansEndToEnd, ElkanPathRecoversBlobs) {
  constexpr std::size_t n = 600;
  constexpr std::size_t d = 128;
  constexpr std::size_t k = 80;
  const Blobs b = makeBlobs(n, d, k, 0.5F, 99U);

  KMeans<float> km(k, 16);
  km.run(b.X, 100, 1e-4F, 5U);

  EXPECT_EQ(km.labels().dim(0), n);
  for (std::size_t i = 0; i < n; ++i) {
    const std::int32_t lbl = km.labels()(i);
    EXPECT_GE(lbl, 0);
    EXPECT_LT(static_cast<std::size_t>(lbl), k);
  }
  EXPECT_TRUE(std::isfinite(km.inertia()));
  EXPECT_GT(purity(km.labels(), b.truth, k), 0.95);
}

TEST(KMeansEndToEnd, RecoversTruthLabelsOnBlobs) {
  constexpr std::size_t n = 600;
  constexpr std::size_t d = 8;
  constexpr std::size_t k = 5;
  const Blobs b = makeBlobs(n, d, k, 0.5F, 42U);

  KMeans<float> km(k, 2);
  km.run(b.X, 100, 1e-4F, 123U);

  EXPECT_EQ(km.labels().dim(0), n);
  EXPECT_EQ(km.centroids().dim(0), k);
  EXPECT_EQ(km.centroids().dim(1), d);
  for (std::size_t i = 0; i < n; ++i) {
    const std::int32_t lbl = km.labels()(i);
    EXPECT_GE(lbl, 0);
    EXPECT_LT(static_cast<std::size_t>(lbl), k);
  }
  EXPECT_GT(purity(km.labels(), b.truth, k), 0.95);
}

TEST(KMeansEndToEnd, InertiaWithinOnePercentOfReferenceBestOfThree) {
  constexpr std::size_t n = 800;
  constexpr std::size_t d = 12;
  constexpr std::size_t k = 6;
  const Blobs b = makeBlobs(n, d, k, 0.8F, 7U);

  KMeans<float> km(k, 2);
  km.run(b.X, 300, 1e-4F, 11U);
  const double ours = km.inertia();

  double refBest = std::numeric_limits<double>::infinity();
  for (const std::uint64_t s : {1ULL, 2ULL, 3ULL}) {
    refBest = std::min(refBest, referenceLloydInertia(b.X, k, s, 300));
  }

  double oursBest = ours;
  for (const std::uint64_t s : {101ULL, 202ULL, 303ULL}) {
    KMeans<float> km2(k, 2);
    km2.run(b.X, 300, 1e-4F, s);
    oursBest = std::min(oursBest, km2.inertia());
  }
  const double gate = 1.01 * std::min(oursBest, refBest);
  EXPECT_LE(ours, gate) << "ours=" << ours << " oursBest=" << oursBest << " refBest=" << refBest;
}

TEST(KMeansEndToEnd, KEqualsOneProducesMeanCentroid) {
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
  KMeans<float> km(1, 1);
  km.run(X, 50, 1e-4F, 99U);
  for (std::size_t i = 0; i < n; ++i) {
    EXPECT_EQ(km.labels()(i), 0);
  }
  for (std::size_t t = 0; t < d; ++t) {
    const auto expected = static_cast<float>(sum[t] / static_cast<double>(n));
    EXPECT_NEAR(km.centroids()(0, t), expected, 1e-4F);
  }
}

TEST(KMeansEndToEnd, KEqualsNGivesZeroInertia) {
  constexpr std::size_t n = 16;
  constexpr std::size_t d = 3;
  NDArray<float, 2> X({n, d});
  std::mt19937 gen(23U);
  std::uniform_real_distribution<float> dist(-5.0F, 5.0F);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t t = 0; t < d; ++t) {
      X(i, t) = dist(gen);
    }
  }
  KMeans<float> km(n, 1);
  km.run(X, 300, 1e-6F, 5U);
  EXPECT_LE(km.inertia(), 1e-4);
  std::vector<std::int32_t> seen(n, 0);
  for (std::size_t i = 0; i < n; ++i) {
    const std::int32_t l = km.labels()(i);
    ASSERT_GE(l, 0);
    ASSERT_LT(static_cast<std::size_t>(l), n);
    ++seen[static_cast<std::size_t>(l)];
  }
  for (const std::int32_t s : seen) {
    EXPECT_EQ(s, 1);
  }
}

TEST(KMeansEndToEndDeathTest, KGreaterThanNAborts) {
  NDArray<float, 2> X({4, 2});
  for (std::size_t i = 0; i < 4; ++i) {
    for (std::size_t t = 0; t < 2; ++t) {
      X(i, t) = static_cast<float>(i);
    }
  }
  KMeans<float> km(8, 1);
  EXPECT_DEATH({ km.run(X, 10, 1e-4F, 0U); }, "clustering: always-assert failed");
}

TEST(KMeansEndToEnd, ReuseAcrossCallsDifferentSeeds) {
  constexpr std::size_t n = 400;
  constexpr std::size_t d = 6;
  constexpr std::size_t k = 4;
  const Blobs b = makeBlobs(n, d, k, 1.2F, 101U);

  KMeans<float> km(k, 2);
  km.run(b.X, 100, 1e-4F, 1U);
  const double i1 = km.inertia();
  const std::vector<std::int32_t> l1(km.labels().data(), km.labels().data() + n);

  km.run(b.X, 100, 1e-4F, 2U);
  const double i2 = km.inertia();
  const std::vector<std::int32_t> l2(km.labels().data(), km.labels().data() + n);

  km.run(b.X, 100, 1e-4F, 3U);
  const double i3 = km.inertia();

  // Different seeds land on different init sequences; at least two of three inertia values
  // should differ (with three reasonable blobs all three usually land on the same optimum, so
  // assert that at least the internal labels vector differs across runs).
  EXPECT_FALSE(l1 == l2 && i1 == i2 && i2 == i3);
  EXPECT_TRUE(std::isfinite(i1));
  EXPECT_TRUE(std::isfinite(i2));
  EXPECT_TRUE(std::isfinite(i3));
}

TEST(KMeansEndToEnd, AdversarialEmptyClusterReseed) {
  // n=16, k=12 with near-duplicated points forces multiple clusters to collapse; the reseed
  // policy must redistribute without looping indefinitely. Stage four clusters worth of
  // coincident rows plus 12 slightly perturbed ones so at least k distinct donors exist.
  constexpr std::size_t n = 16;
  constexpr std::size_t d = 3;
  constexpr std::size_t k = 12;
  NDArray<float, 2> X({n, d});
  for (std::size_t i = 0; i < n; ++i) {
    const auto mode = static_cast<float>(i % 4);
    const auto perturb = static_cast<float>(i) * 1e-3F; // enough to separate the 16 rows.
    for (std::size_t t = 0; t < d; ++t) {
      X(i, t) = mode + perturb;
    }
  }

  KMeans<float> km(k, 1);
  km.run(X, 50, 1e-4F, 0U);
  EXPECT_LE(km.nIter(), 50U);
  std::vector<std::int32_t> counts(k, 0);
  for (std::size_t i = 0; i < n; ++i) {
    ++counts[static_cast<std::size_t>(km.labels()(i))];
  }
  for (const std::int32_t c : counts) {
    EXPECT_GE(c, 1);
  }
}

// Pins the sklearn-compatible tol semantic: on data with variance O(1), tol=1e-4 converges in
// far fewer than maxIter iterations. Under the earlier raw-L2-shift convention the internal
// threshold was tol*tol = 1e-8 which, on typical blob data, required hundreds of iterations;
// the variance-scaled threshold follows sklearn's _tolerance(X, tol) = tol * mean_var and
// terminates near the sklearn iteration count.
TEST(KMeansEndToEnd, TolScaledByVarianceMatchesSklearnConvergencePace) {
  constexpr std::size_t n = 5000;
  constexpr std::size_t d = 4;
  constexpr std::size_t k = 8;
  // Separation 50 vs sigma 0.5 produces variance-of-columns ~O(1e3), so the variance-scaled
  // threshold (tol * mean_var) is comfortably looser than the pre-fix (tol*tol) and iteration
  // counts collapse to a typical Lloyd pace of 2-20 steps.
  const Blobs b = makeBlobs(n, d, k, 0.5F, 77U);

  KMeans<float> km(k, 1);
  km.run(b.X, 300U, 1e-4F, 17U);
  EXPECT_TRUE(km.converged());
  EXPECT_LT(km.nIter(), 50U) << "nIter=" << km.nIter();
}

TEST(KMeansEndToEnd, ResetReleasesThenReruns) {
  constexpr std::size_t n = 200;
  constexpr std::size_t d = 4;
  constexpr std::size_t k = 3;
  const Blobs b = makeBlobs(n, d, k, 0.6F, 55U);

  KMeans<float> km(k, 1);
  km.run(b.X, 100, 1e-4F, 7U);
  EXPECT_GT(km.nIter(), 0U);
  km.reset();
  EXPECT_EQ(km.nIter(), 0U);
  EXPECT_FALSE(km.converged());
  km.run(b.X, 100, 1e-4F, 8U);
  EXPECT_GT(km.nIter(), 0U);
}

TEST(KMeansEndToEnd, LabelsMatchBruteForceArgminAtMidDim) {
  // d > pairwiseArgminMaxD routes the Lloyd assignment step through the internal chunked
  // materialized path. Shape is chosen so n exceeds one pairwiseArgminChunkRows boundary --
  // exercising both a whole chunk and a tail pass -- and so k is well inside the
  // [1, kNc<float>] envelope the single-call packB supports. The test pins label correctness
  // by comparing against scalar brute-force argmin over the final centroids, so any divergence
  // between the chunked driver and the textbook argmin_c ||x - c||^2 surfaces as a labels[i]
  // mismatch.
  constexpr std::size_t n = 600;
  constexpr std::size_t d = 96;
  constexpr std::size_t k = 16;
  const Blobs b = makeBlobs(n, d, k, 0.4F, 2027U);
  ASSERT_GT(d, clustering::math::defaults::pairwiseArgminMaxD);

  KMeans<float> km(k, 1);
  km.run(b.X, 100, 1e-4F, 42U);

  const auto &centroids = km.centroids();
  const auto &labels = km.labels();
  ASSERT_EQ(centroids.dim(0), k);
  ASSERT_EQ(centroids.dim(1), d);
  ASSERT_EQ(labels.dim(0), n);

  for (std::size_t i = 0; i < n; ++i) {
    float bestVal = std::numeric_limits<float>::infinity();
    std::int32_t bestIdx = 0;
    for (std::size_t j = 0; j < k; ++j) {
      float s = 0.0F;
      for (std::size_t t = 0; t < d; ++t) {
        const float diff = b.X(i, t) - centroids(j, t);
        s += diff * diff;
      }
      if (s < bestVal) {
        bestVal = s;
        bestIdx = static_cast<std::int32_t>(j);
      }
    }
    EXPECT_EQ(labels(i), bestIdx) << "i=" << i;
  }
}
