#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <utility>
#include <vector>

#include "clustering/index/kdtree.h"
#include "clustering/math/thread.h"
#include "clustering/ndarray.h"

using clustering::KDTree;
using clustering::NDArray;
using clustering::math::Pool;

namespace {

// Brute-force kNN oracle: for every row i, sort the (dsq, j) pairs over j != i and take the
// first k. Ties resolve on smaller j so results are comparable to the KDTree's bounded-heap
// tie-break rule.
struct KnnOracle {
  std::vector<std::vector<std::int32_t>> indices;
  std::vector<std::vector<float>> sqDists;
};

KnnOracle bruteForceKnn(const NDArray<float, 2> &points, std::int32_t k) {
  const std::size_t n = points.dim(0);
  const std::size_t d = points.dim(1);
  const auto kSz = static_cast<std::size_t>(k);
  KnnOracle oracle;
  oracle.indices.resize(n);
  oracle.sqDists.resize(n);

  std::vector<std::pair<float, std::int32_t>> scratch;
  scratch.reserve(n);

  for (std::size_t i = 0; i < n; ++i) {
    scratch.clear();
    for (std::size_t j = 0; j < n; ++j) {
      if (i == j) {
        continue;
      }
      float dsq = 0.0F;
      for (std::size_t t = 0; t < d; ++t) {
        const float diff = points[i][t] - points[j][t];
        dsq += diff * diff;
      }
      scratch.emplace_back(dsq, static_cast<std::int32_t>(j));
    }
    // Matched tie-break: (dsq, idx) ascending -- smaller dsq wins, equal dsq breaks on smaller
    // idx. Identical rule to the BoundedMaxHeap's less() predicate.
    std::sort(scratch.begin(), scratch.end(),
              [](const std::pair<float, std::int32_t> &a, const std::pair<float, std::int32_t> &b) {
                if (a.first != b.first) {
                  return a.first < b.first;
                }
                return a.second < b.second;
              });
    oracle.indices[i].reserve(kSz);
    oracle.sqDists[i].reserve(kSz);
    for (std::size_t t = 0; t < kSz; ++t) {
      oracle.indices[i].push_back(scratch[t].second);
      oracle.sqDists[i].push_back(scratch[t].first);
    }
  }
  return oracle;
}

NDArray<float, 2> makeRandomPoints(std::size_t n, std::size_t d, std::uint64_t seed) {
  NDArray<float, 2> points({n, d});
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<float> dist(-5.0F, 5.0F);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < d; ++j) {
      points[i][j] = dist(rng);
    }
  }
  return points;
}

// Verify that the KDTree kNN result matches the brute-force oracle.
void expectMatchesOracle(const NDArray<std::int32_t, 2> &idxOut, const NDArray<float, 2> &distOut,
                         const KnnOracle &oracle, std::int32_t k) {
  const std::size_t n = idxOut.dim(0);
  const auto kSz = static_cast<std::size_t>(k);
  ASSERT_EQ(oracle.indices.size(), n);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < kSz; ++j) {
      EXPECT_EQ(idxOut[i][j], oracle.indices[i][j]) << "i=" << i << " j=" << j;
      // Relative float tolerance: the KDTree's FMA-based squared-distance accumulator rounds
      // once per FMA, while the brute-force oracle does a plain multiply-then-add. The magnitude
      // of the rounding gap is a few ULP per dim summed across d additions.
      const float rel = std::max(1.0F, std::abs(oracle.sqDists[i][j])) * 1e-5F;
      EXPECT_NEAR(distOut[i][j], oracle.sqDists[i][j], rel) << "i=" << i << " j=" << j;
    }
  }
}

} // namespace

TEST(KdtreeKnn, MatchesBruteForceAtDim2) {
  const std::size_t n = 100;
  const std::size_t d = 2;
  const std::int32_t k = 5;
  const auto points = makeRandomPoints(n, d, 0xA11CE);
  const KDTree<float> tree(points);
  auto [idxOut, distOut] = tree.knnQuery(k, Pool{});
  const auto oracle = bruteForceKnn(points, k);
  expectMatchesOracle(idxOut, distOut, oracle, k);
}

TEST(KdtreeKnn, MatchesBruteForceAtDim8) {
  const std::size_t n = 100;
  const std::size_t d = 8;
  const std::int32_t k = 5;
  const auto points = makeRandomPoints(n, d, 0xB0B);
  const KDTree<float> tree(points);
  auto [idxOut, distOut] = tree.knnQuery(k, Pool{});
  const auto oracle = bruteForceKnn(points, k);
  expectMatchesOracle(idxOut, distOut, oracle, k);
}

TEST(KdtreeKnn, MatchesBruteForceAtDim32) {
  const std::size_t n = 100;
  const std::size_t d = 32;
  const std::int32_t k = 5;
  const auto points = makeRandomPoints(n, d, 0xCAFE);
  const KDTree<float> tree(points);
  auto [idxOut, distOut] = tree.knnQuery(k, Pool{});
  const auto oracle = bruteForceKnn(points, k);
  expectMatchesOracle(idxOut, distOut, oracle, k);
}

TEST(KdtreeKnn, SelfExclusion) {
  const std::size_t n = 50;
  const std::size_t d = 4;
  const std::int32_t k = 7;
  const auto points = makeRandomPoints(n, d, 0xF00D);
  const KDTree<float> tree(points);
  auto [idxOut, distOut] = tree.knnQuery(k, Pool{});
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < static_cast<std::size_t>(k); ++j) {
      EXPECT_NE(idxOut[i][j], static_cast<std::int32_t>(i))
          << "point " << i << " returned itself at slot " << j;
    }
  }
}

TEST(KdtreeKnn, SortedAscendingBySqDist) {
  const std::size_t n = 75;
  const std::size_t d = 3;
  const std::int32_t k = 10;
  const auto points = makeRandomPoints(n, d, 0xDEAD);
  const KDTree<float> tree(points);
  auto [idxOut, distOut] = tree.knnQuery(k, Pool{});
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 1; j < static_cast<std::size_t>(k); ++j) {
      EXPECT_LE(distOut[i][j - 1], distOut[i][j])
          << "row " << i << " not sorted ascending at slot " << j;
    }
  }
}

TEST(KdtreeKnn, KEqualsOneReproducesNearest) {
  const std::size_t n = 60;
  const std::size_t d = 2;
  const std::int32_t k = 1;
  const auto points = makeRandomPoints(n, d, 0xBABA);
  const KDTree<float> tree(points);
  auto [idxOut, distOut] = tree.knnQuery(k, Pool{});
  const auto oracle = bruteForceKnn(points, k);
  for (std::size_t i = 0; i < n; ++i) {
    EXPECT_EQ(idxOut[i][0], oracle.indices[i][0]) << "i=" << i;
    const float rel = std::max(1.0F, std::abs(oracle.sqDists[i][0])) * 1e-5F;
    EXPECT_NEAR(distOut[i][0], oracle.sqDists[i][0], rel) << "i=" << i;
  }
}

TEST(KdtreeKnn, KEqualsNMinusOneReturnsAllOthers) {
  const std::size_t n = 20;
  const std::size_t d = 3;
  const auto kn = static_cast<std::int32_t>(n - 1);
  const auto points = makeRandomPoints(n, d, 0x42);
  const KDTree<float> tree(points);
  auto [idxOut, distOut] = tree.knnQuery(kn, Pool{});
  // For each row, the set of returned indices must be exactly every j != i.
  for (std::size_t i = 0; i < n; ++i) {
    std::vector<std::int32_t> returned;
    returned.reserve(static_cast<std::size_t>(kn));
    for (std::size_t j = 0; j < static_cast<std::size_t>(kn); ++j) {
      returned.push_back(idxOut[i][j]);
    }
    std::sort(returned.begin(), returned.end());
    std::vector<std::int32_t> expected;
    expected.reserve(static_cast<std::size_t>(kn));
    for (std::size_t j = 0; j < n; ++j) {
      if (j != i) {
        expected.push_back(static_cast<std::int32_t>(j));
      }
    }
    EXPECT_EQ(returned, expected) << "i=" << i;
  }
}

TEST(KdtreeKnn, AllEqualPointsIsDeterministic) {
  // Degenerate input: every point is at the origin, so every pair has squared distance 0. The
  // bounded max-heap's (key, val) tie-break retains the k smallest original indices, and the
  // kNN result must be reproducible across runs.
  const std::size_t n = 32;
  const std::size_t d = 4;
  const std::int32_t k = 5;
  NDArray<float, 2> points({n, d});
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < d; ++j) {
      points[i][j] = 0.0F;
    }
  }

  const KDTree<float> treeA(points);
  const KDTree<float> treeB(points);
  auto [idxA, distA] = treeA.knnQuery(k, Pool{});
  auto [idxB, distB] = treeB.knnQuery(k, Pool{});
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < static_cast<std::size_t>(k); ++j) {
      EXPECT_EQ(idxA[i][j], idxB[i][j]) << "i=" << i << " j=" << j;
      EXPECT_EQ(distA[i][j], 0.0F) << "i=" << i << " j=" << j;
      EXPECT_EQ(distB[i][j], 0.0F) << "i=" << i << " j=" << j;
    }
    // Self-exclusion must still hold.
    for (std::size_t j = 0; j < static_cast<std::size_t>(k); ++j) {
      EXPECT_NE(idxA[i][j], static_cast<std::int32_t>(i));
    }
  }
}

TEST(KdtreeKnn, ParallelAgreesWithSerial) {
  // Worker pools must produce the same result as serial because kNN is a pure per-row query.
  const std::size_t n = 200;
  const std::size_t d = 6;
  const std::int32_t k = 8;
  const auto points = makeRandomPoints(n, d, 0xBEEF);
  const KDTree<float> tree(points);

  auto [idxSerial, distSerial] = tree.knnQuery(k, Pool{});

  BS::light_thread_pool poolHandle(4);
  auto [idxPar, distPar] = tree.knnQuery(k, Pool{&poolHandle});

  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < static_cast<std::size_t>(k); ++j) {
      EXPECT_EQ(idxSerial[i][j], idxPar[i][j]) << "i=" << i << " j=" << j;
      EXPECT_EQ(distSerial[i][j], distPar[i][j]) << "i=" << i << " j=" << j;
    }
  }
}

TEST(KdtreeKnnDeath, KGreaterOrEqualToNAborts) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  const std::size_t n = 10;
  const std::size_t d = 2;
  const auto points = makeRandomPoints(n, d, 1);
  const KDTree<float> tree(points);
  EXPECT_DEATH(
      { (void)tree.knnQuery(static_cast<std::int32_t>(n), Pool{}); },
      "always-assert failed: std::cmp_less\\(k, n\\)");
  EXPECT_DEATH(
      { (void)tree.knnQuery(static_cast<std::int32_t>(n + 1), Pool{}); },
      "always-assert failed: std::cmp_less\\(k, n\\)");
}

TEST(KdtreeKnnDeath, KZeroAborts) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  const std::size_t n = 10;
  const std::size_t d = 2;
  const auto points = makeRandomPoints(n, d, 2);
  const KDTree<float> tree(points);
  EXPECT_DEATH({ (void)tree.knnQuery(0, Pool{}); }, "always-assert failed: k >= 1");
}
