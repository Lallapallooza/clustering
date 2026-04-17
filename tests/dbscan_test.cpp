#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <random>

#include "clustering/dbscan.h"
#include "clustering/index/auto_range_index.h"
#include "clustering/index/brute_force_pairwise.h"
#include "clustering/index/kdtree.h"
#include "clustering/index/range_query.h"

using clustering::BruteForcePairwise;
using clustering::DBSCAN;
using clustering::KDTree;
using clustering::KDTreeDistanceType;
using clustering::NDArray;
using clustering::index::AutoRangeIndex;

namespace {
struct MissingQueryIndex {
  explicit MissingQueryIndex(const NDArray<float, 2> & /*points*/) {}
};
} // namespace

static_assert(!clustering::index::RangeIndex<MissingQueryIndex, float>);
static_assert(clustering::index::RangeIndex<KDTree<float, KDTreeDistanceType::kEucledian>, float>);
static_assert(clustering::index::RangeIndex<BruteForcePairwise<float>, float>);
static_assert(clustering::index::RangeIndex<AutoRangeIndex<float>, float>);

namespace {

// Three well-separated unit-variance blobs at centres +/- 10. Any eps well below 10 resolves
// three clusters regardless of dim.
NDArray<float, 2> makeThreeBlobs(std::size_t perBlob, std::size_t d, std::uint64_t seed) {
  NDArray<float, 2> points({perBlob * 3, d});
  std::mt19937_64 rng(seed);
  std::normal_distribution<float> jitter(0.0f, 0.05f);
  const std::array<float, 3> centers{-10.0f, 0.0f, 10.0f};
  for (std::size_t b = 0; b < 3; ++b) {
    for (std::size_t i = 0; i < perBlob; ++i) {
      for (std::size_t k = 0; k < d; ++k) {
        points[(b * perBlob) + i][k] = centers[b] + jitter(rng);
      }
    }
  }
  return points;
}

} // namespace

TEST(DBSCAN, FindsTwoWellSeparatedClusters) {
  NDArray<float, 2> points({10, 2});
  for (std::size_t i = 0; i < 5; ++i) {
    points[i][0] = static_cast<float>(i) * 0.1f;
    points[i][1] = 0.0f;
  }
  for (std::size_t i = 5; i < 10; ++i) {
    points[i][0] = 10.0f + (static_cast<float>(i - 5) * 0.1f);
    points[i][1] = 0.0f;
  }

  DBSCAN<float> dbscan(points, 0.3f, 2, 1);
  dbscan.run();

  EXPECT_EQ(dbscan.nClusters(), 2u);
  EXPECT_NE(dbscan.labels()[0], dbscan.labels()[5]);
}

TEST(DBSCAN, MarksIsolatedPointsAsNoise) {
  NDArray<float, 2> points({6, 2});
  for (std::size_t i = 0; i < 5; ++i) {
    points[i][0] = static_cast<float>(i) * 0.1f;
    points[i][1] = 0.0f;
  }
  points[5][0] = 1000.0f;
  points[5][1] = 1000.0f;

  DBSCAN<float> dbscan(points, 0.3f, 2, 1);
  dbscan.run();

  EXPECT_EQ(dbscan.nClusters(), 1u);
  EXPECT_EQ(dbscan.labels()[5], DBSCAN<float>::NOISY);
}

// Both backends must produce the same partition across the KDTree/brute-force dimension
// boundary so callers who pin either explicitly see identical clustering behaviour.
TEST(DBSCAN, ExplicitBackendsAgreeAcrossDims) {
  for (const std::size_t d : {4U, 8U, 16U, 32U, 64U}) {
    const auto points = makeThreeBlobs(200, d, /*seed=*/0xC10D + d);

    DBSCAN<float, KDTree<float, KDTreeDistanceType::kEucledian>> viaTree(points, 1.0f, 5, 2);
    viaTree.run();

    DBSCAN<float, BruteForcePairwise<float>> viaBrute(points, 1.0f, 5, 2);
    viaBrute.run();

    ASSERT_EQ(viaTree.nClusters(), 3u) << "fixture degenerate at d=" << d;
    ASSERT_EQ(viaBrute.nClusters(), viaTree.nClusters()) << "backend mismatch at d=" << d;
    for (std::size_t i = 0; i < points.dim(0); ++i) {
      EXPECT_EQ(viaTree.labels()[i], viaBrute.labels()[i])
          << "label mismatch at d=" << d << " i=" << i;
    }
  }
}
