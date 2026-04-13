#include <gtest/gtest.h>

#include "clustering/kdtree.h"

using clustering::KDTree;
using clustering::NDArray;

TEST(KDTree, RadiusQueryFindsNearbyPoints) {
  NDArray<float, 2> points({4, 2});
  points[0][0] = 0.0f;
  points[0][1] = 0.0f;
  points[1][0] = 0.1f;
  points[1][1] = 0.0f;
  points[2][0] = 5.0f;
  points[2][1] = 5.0f;
  points[3][0] = 10.0f;
  points[3][1] = 0.0f;

  const KDTree<float> tree(points);

  NDArray<float, 1> query({2});
  query[0] = 0.0f;
  query[1] = 0.0f;

  auto neighbors = tree.query(query, 0.5f);
  EXPECT_EQ(neighbors.size(), 2u);
}

TEST(KDTree, RadiusQueryReturnsEmptyWhenNoneInRange) {
  NDArray<float, 2> points({2, 2});
  points[0][0] = 0.0f;
  points[0][1] = 0.0f;
  points[1][0] = 10.0f;
  points[1][1] = 10.0f;

  const KDTree<float> tree(points);

  NDArray<float, 1> query({2});
  query[0] = 100.0f;
  query[1] = 100.0f;

  auto neighbors = tree.query(query, 1.0f);
  EXPECT_TRUE(neighbors.empty());
}

TEST(KDTree, LimitCapsTheReturnedCount) {
  NDArray<float, 2> points({10, 2});
  for (std::size_t i = 0; i < 10; ++i) {
    points[i][0] = static_cast<float>(i) * 0.01f;
    points[i][1] = 0.0f;
  }

  const KDTree<float> tree(points);

  NDArray<float, 1> query({2});
  query[0] = 0.0f;
  query[1] = 0.0f;

  auto neighbors = tree.query(query, 1.0f, 3);
  EXPECT_EQ(neighbors.size(), 3u);
}
