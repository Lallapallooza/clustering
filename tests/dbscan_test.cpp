#include <gtest/gtest.h>

#include "clustering/dbscan.h"
#include "clustering/index/kdtree.h"
#include "clustering/index/range_query.h"

using clustering::DBSCAN;
using clustering::KDTree;
using clustering::KDTreeDistanceType;
using clustering::NDArray;

namespace {
struct MissingQueryIndex {
  explicit MissingQueryIndex(const NDArray<float, 2> & /*points*/) {}
};
} // namespace

static_assert(!clustering::index::RangeQuery<MissingQueryIndex, float>);

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

TEST(DBSCAN, PinnedKDTreeSpellingMatchesDefault) {
  NDArray<float, 2> points({10, 2});
  for (std::size_t i = 0; i < 5; ++i) {
    points[i][0] = static_cast<float>(i) * 0.1f;
    points[i][1] = 0.0f;
  }
  for (std::size_t i = 5; i < 10; ++i) {
    points[i][0] = 10.0f + (static_cast<float>(i - 5) * 0.1f);
    points[i][1] = 0.0f;
  }

  DBSCAN<float> defaultDbscan(points, 0.3f, 2, 1);
  defaultDbscan.run();

  DBSCAN<float, KDTree<float, KDTreeDistanceType::kEucledian>> pinnedDbscan(points, 0.3f, 2, 1);
  pinnedDbscan.run();

  ASSERT_EQ(defaultDbscan.nClusters(), pinnedDbscan.nClusters());
  for (std::size_t i = 0; i < points.dim(0); ++i) {
    EXPECT_EQ(defaultDbscan.labels()[i].load(std::memory_order_relaxed),
              pinnedDbscan.labels()[i].load(std::memory_order_relaxed));
  }
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
