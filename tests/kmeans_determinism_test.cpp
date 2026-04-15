#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

#include "clustering/kmeans.h"
#include "clustering/ndarray.h"

using clustering::KMeans;
using clustering::NDArray;

namespace {

NDArray<float, 2> makeData(std::size_t n, std::size_t d, std::uint32_t seed) {
  NDArray<float, 2> X({n, d});
  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> dist(-3.0F, 3.0F);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t t = 0; t < d; ++t) {
      X(i, t) = dist(gen);
    }
  }
  return X;
}

struct Fit {
  std::vector<std::int32_t> labels;
  std::vector<float> centroids;
  double inertia;
};

Fit capture(const KMeans<float> &km, std::size_t n, std::size_t k, std::size_t d) {
  Fit f;
  f.labels.assign(km.labels().data(), km.labels().data() + n);
  f.centroids.assign(km.centroids().data(), km.centroids().data() + (k * d));
  f.inertia = km.inertia();
  return f;
}

} // namespace

// Bit-identity across ten repeated fits at the same nJobs.
TEST(KMeansDeterminism, TenRunsBitIdentical) {
  constexpr std::size_t n = 512;
  constexpr std::size_t d = 16;
  constexpr std::size_t k = 8;
  const NDArray<float, 2> X = makeData(n, d, 314U);

  constexpr std::size_t nJobs = 4;
  constexpr std::uint64_t seed = 2024U;
  constexpr std::size_t maxIter = 100;
  constexpr float tol = 1e-4F;

  KMeans<float> km0(k, nJobs);
  km0.run(X, maxIter, tol, seed);
  const Fit reference = capture(km0, n, k, d);

  for (int rep = 1; rep < 10; ++rep) {
    KMeans<float> km(k, nJobs);
    km.run(X, maxIter, tol, seed);
    const Fit r = capture(km, n, k, d);
    ASSERT_EQ(reference.labels, r.labels) << "labels diverge at repetition " << rep;
    // Bit-identity on centroid matrix and inertia.
    ASSERT_EQ(reference.centroids.size(), r.centroids.size());
    for (std::size_t j = 0; j < reference.centroids.size(); ++j) {
      ASSERT_EQ(std::bit_cast<std::uint32_t>(reference.centroids[j]),
                std::bit_cast<std::uint32_t>(r.centroids[j]))
          << "centroid[" << j << "] diverges at repetition " << rep;
    }
    ASSERT_EQ(std::bit_cast<std::uint64_t>(reference.inertia),
              std::bit_cast<std::uint64_t>(r.inertia));
  }
}

// Cross-nJobs is NOT bit-identical (ULP drift acknowledged). Inertia should agree within a
// relaxed relative tolerance.
TEST(KMeansDeterminism, CrossNJobsInertiaWithinTol) {
  constexpr std::size_t n = 512;
  constexpr std::size_t d = 16;
  constexpr std::size_t k = 8;
  const NDArray<float, 2> X = makeData(n, d, 271U);

  KMeans<float> km1(k, 1);
  km1.run(X, 100, 1e-4F, 17U);

  KMeans<float> km4(k, 4);
  km4.run(X, 100, 1e-4F, 17U);

  const double i1 = km1.inertia();
  const double i4 = km4.inertia();
  const double rel = std::abs(i1 - i4) / std::max(1.0, std::abs(i1));
  EXPECT_LT(rel, 1e-4) << "i1=" << i1 << " i4=" << i4;
}
