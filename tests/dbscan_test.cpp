#include <gtest/gtest.h>

#include "clustering/dbscan.h"

TEST(DBSCAN, FindsTwoWellSeparatedClusters) {
  NDArray<float, 2> points({10, 2});
  for (std::size_t i = 0; i < 5; ++i) {
    points[i][0] = static_cast<float>(i) * 0.1f;
    points[i][1] = 0.0f;
  }
  for (std::size_t i = 5; i < 10; ++i) {
    points[i][0] = 10.0f + static_cast<float>(i - 5) * 0.1f;
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
