#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <random>
#include <set>
#include <type_traits>
#include <vector>

#include "clustering/index/nn_descent.h"
#include "clustering/math/thread.h"
#include "clustering/ndarray.h"

using clustering::NDArray;
using clustering::index::NnDescentIndex;
using clustering::math::Pool;

namespace {

// Generate a Gaussian point cloud at (n, d) with unit variance and deterministic seed.
NDArray<float, 2> makeGaussian(std::size_t n, std::size_t d, std::uint64_t seed) {
  NDArray<float, 2> X({n, d});
  std::mt19937_64 gen(seed);
  std::normal_distribution<float> dist(0.0F, 1.0F);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < d; ++j) {
      X(i, j) = dist(gen);
    }
  }
  return X;
}

// Brute-force kNN: for each point, compute distance to all others and return the k smallest
// (self-excluded), sorted ascending by distance.
std::vector<std::vector<std::int32_t>> bruteForceKnn(const NDArray<float, 2> &X, std::size_t k) {
  const std::size_t n = X.dim(0);
  const std::size_t d = X.dim(1);
  std::vector<std::vector<std::int32_t>> out(n);
  for (std::size_t i = 0; i < n; ++i) {
    std::vector<std::pair<float, std::int32_t>> cands;
    cands.reserve(n - 1);
    for (std::size_t j = 0; j < n; ++j) {
      if (i == j) {
        continue;
      }
      float sq = 0.0F;
      for (std::size_t t = 0; t < d; ++t) {
        const float diff = X(i, t) - X(j, t);
        sq += diff * diff;
      }
      cands.emplace_back(sq, static_cast<std::int32_t>(j));
    }
    std::partial_sort(cands.begin(), cands.begin() + static_cast<std::ptrdiff_t>(k), cands.end(),
                      [](const auto &a, const auto &b) {
                        if (a.first < b.first) {
                          return true;
                        }
                        if (b.first < a.first) {
                          return false;
                        }
                        return a.second < b.second;
                      });
    out[i].reserve(k);
    for (std::size_t s = 0; s < k; ++s) {
      out[i].push_back(cands[s].second);
    }
  }
  return out;
}

// Recall @ k: fraction of brute-force neighbors also present in approximate neighbors, averaged
// across every query point.
double recallAtK(const std::vector<std::vector<std::int32_t>> &approx,
                 const std::vector<std::vector<std::int32_t>> &bruteForce) {
  const std::size_t n = approx.size();
  double sum = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const std::set<std::int32_t> exact(bruteForce[i].begin(), bruteForce[i].end());
    std::size_t hits = 0;
    for (const std::int32_t v : approx[i]) {
      if (exact.contains(v)) {
        ++hits;
      }
    }
    sum += static_cast<double>(hits) / static_cast<double>(exact.size());
  }
  return sum / static_cast<double>(n);
}

// Strip the distance field to just indices for recall comparison.
std::vector<std::vector<std::int32_t>>
stripIdx(const std::vector<std::vector<NnDescentIndex<float>::KnnEntry>> &src) {
  std::vector<std::vector<std::int32_t>> out(src.size());
  for (std::size_t i = 0; i < src.size(); ++i) {
    out[i].reserve(src[i].size());
    for (const auto &e : src[i]) {
      out[i].push_back(e.idx);
    }
  }
  return out;
}

} // namespace

// ---------------------------------------------------------------------------
// Compile-time API surface:
// - Zero-argument construction must be rejected. Since k has no safe default, the class's only
//   ctor requires a std::size_t, so `NnDescentIndex<float>()` is ill-formed.
// ---------------------------------------------------------------------------

static_assert(!std::is_default_constructible_v<NnDescentIndex<float>>,
              "NnDescentIndex must require k at construction");
static_assert(std::is_constructible_v<NnDescentIndex<float>, std::size_t>,
              "NnDescentIndex must accept k as a single argument");

// ---------------------------------------------------------------------------
// Recall at three high-dimensional Gaussian configurations spanning the intended kNN regime.
// Thresholds match the spec; measured headroom is substantial.
// ---------------------------------------------------------------------------

TEST(NnDescent, RecallAtD64K15) {
  const std::size_t n = 1000;
  const std::size_t d = 64;
  const std::size_t k = 15;
  auto X = makeGaussian(n, d, 0xFEED0001ULL);
  NnDescentIndex<float> idx(k, /*maxIter=*/10, /*delta=*/1e-3F, /*seed=*/42ULL);
  const Pool pool;
  idx.build(X, pool);
  const auto exact = bruteForceKnn(X, k);
  const auto approx = stripIdx(idx.neighbors());
  const double recall = recallAtK(approx, exact);
  EXPECT_GE(recall, 0.90) << "Observed recall " << recall << " at (n=" << n << ", d=" << d
                          << ", k=" << k << ")";
}

TEST(NnDescent, RecallAtD128K15) {
  const std::size_t n = 1000;
  const std::size_t d = 128;
  const std::size_t k = 15;
  auto X = makeGaussian(n, d, 0xFEED0002ULL);
  NnDescentIndex<float> idx(k, /*maxIter=*/10, /*delta=*/1e-3F, /*seed=*/43ULL);
  const Pool pool;
  idx.build(X, pool);
  const auto exact = bruteForceKnn(X, k);
  const auto approx = stripIdx(idx.neighbors());
  const double recall = recallAtK(approx, exact);
  EXPECT_GE(recall, 0.85) << "Observed recall " << recall << " at (n=" << n << ", d=" << d
                          << ", k=" << k << ")";
}

TEST(NnDescent, RecallAtD300K30) {
  const std::size_t n = 1000;
  const std::size_t d = 300;
  const std::size_t k = 30;
  auto X = makeGaussian(n, d, 0xFEED0003ULL);
  NnDescentIndex<float> idx(k, /*maxIter=*/10, /*delta=*/1e-3F, /*seed=*/44ULL);
  const Pool pool;
  idx.build(X, pool);
  const auto exact = bruteForceKnn(X, k);
  const auto approx = stripIdx(idx.neighbors());
  const double recall = recallAtK(approx, exact);
  EXPECT_GE(recall, 0.80) << "Observed recall " << recall << " at (n=" << n << ", d=" << d
                          << ", k=" << k << ")";
}

// ---------------------------------------------------------------------------
// Connectivity: all three shapes should yield a connected graph at k >= 15.
// ---------------------------------------------------------------------------

TEST(NnDescent, ConnectedAtD64K15) {
  auto X = makeGaussian(1000, 64, 0xCAFE0001ULL);
  NnDescentIndex<float> idx(15);
  const Pool pool;
  idx.build(X, pool);
  EXPECT_TRUE(idx.isConnected());
}

TEST(NnDescent, ConnectedAtD128K15) {
  auto X = makeGaussian(1000, 128, 0xCAFE0002ULL);
  NnDescentIndex<float> idx(15);
  const Pool pool;
  idx.build(X, pool);
  EXPECT_TRUE(idx.isConnected());
}

TEST(NnDescent, ConnectedAtD300K30) {
  auto X = makeGaussian(1000, 300, 0xCAFE0003ULL);
  NnDescentIndex<float> idx(30);
  const Pool pool;
  idx.build(X, pool);
  EXPECT_TRUE(idx.isConnected());
}

// ---------------------------------------------------------------------------
// Warm start: two consecutive builds on the same input must converge faster on the second.
// @c k is immutable per @c NnDescentIndex instance, so different-@c k cold-start is an
// inherent property of the class (no test needed).
// ---------------------------------------------------------------------------

TEST(NnDescent, WarmStartFasterOnRepeatBuild) {
  auto X = makeGaussian(1000, 64, 0xBEEF0001ULL);
  NnDescentIndex<float> idx(15);
  const Pool pool;
  idx.build(X, pool);
  const std::size_t coldIter = idx.lastIterations();

  idx.build(X, pool);
  const std::size_t warmIter = idx.lastIterations();

  EXPECT_GT(coldIter, 0u);
  EXPECT_LT(warmIter, coldIter) << "Cold iterations: " << coldIter
                                << " Warm iterations: " << warmIter;
}

// ---------------------------------------------------------------------------
// k and neighbor shape sanity
// ---------------------------------------------------------------------------

TEST(NnDescent, NeighborsShapeMatchesK) {
  auto X = makeGaussian(200, 16, 0xAAAA0001ULL);
  NnDescentIndex<float> idx(10);
  const Pool pool;
  idx.build(X, pool);
  EXPECT_EQ(idx.k(), 10u);
  const auto &neighbors = idx.neighbors();
  EXPECT_EQ(neighbors.size(), 200u);
  for (const auto &row : neighbors) {
    EXPECT_EQ(row.size(), 10u);
  }
  // Every neighbor must be a valid in-range index, distinct from the query point.
  for (std::size_t i = 0; i < neighbors.size(); ++i) {
    for (const auto &e : neighbors[i]) {
      EXPECT_GE(e.idx, 0);
      EXPECT_LT(e.idx, 200);
      EXPECT_NE(e.idx, static_cast<std::int32_t>(i));
    }
  }
}

TEST(NnDescent, NeighborsSortedByDistanceAscending) {
  auto X = makeGaussian(300, 32, 0xAAAA0002ULL);
  NnDescentIndex<float> idx(12);
  const Pool pool;
  idx.build(X, pool);
  for (const auto &row : idx.neighbors()) {
    for (std::size_t s = 1; s < row.size(); ++s) {
      EXPECT_LE(row[s - 1].sqDist, row[s].sqDist);
    }
  }
}

// ---------------------------------------------------------------------------
// Precondition death: k must be >= 1 at construction; k < n at build time.
// ---------------------------------------------------------------------------

TEST(NnDescentDeath, KZeroConstructAborts) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_DEATH({ const NnDescentIndex<float> idx(0); }, "always-assert failed: k >= 1");
}

TEST(NnDescentDeath, KGeNAborts) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  NnDescentIndex<float> idx(10);
  const auto X = makeGaussian(5, 4, 0xAAAA0003ULL);
  EXPECT_DEATH(
      {
        const Pool p;
        idx.build(X, p);
      },
      "always-assert failed: kFitsN");
}

// ---------------------------------------------------------------------------
// Stress under multi-threaded pool: no nested parallelism. The join step fans across
// nodes via the pool; workers never re-enter the pool.
// ---------------------------------------------------------------------------

TEST(NnDescent, StressUnderPoolNJobs16) {
  auto X = makeGaussian(2000, 32, 0xAAAA0004ULL);
  NnDescentIndex<float> idx(15);
  BS::light_thread_pool poolImpl(16);
  const Pool pool{&poolImpl};
  idx.build(X, pool);
  EXPECT_TRUE(idx.isConnected());
  // Sanity: neighbors should still be valid, distinct from self, in-range.
  for (std::size_t i = 0; i < idx.neighbors().size(); ++i) {
    for (const auto &e : idx.neighbors()[i]) {
      EXPECT_GE(e.idx, 0);
      EXPECT_LT(e.idx, 2000);
      EXPECT_NE(e.idx, static_cast<std::int32_t>(i));
    }
  }
}
