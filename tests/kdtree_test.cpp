#include <gtest/gtest.h>

#include "clustering/index/kdtree.h"

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

TEST(KDTree, PoolBuiltTreeMatchesSerialBuild) {
  // Enough points that the build recursion forks above its serial floor.
  constexpr std::size_t kN = 4096;
  NDArray<float, 2> points({kN, 3});
  std::uint64_t state = 42;
  for (std::size_t i = 0; i < kN; ++i) {
    for (std::size_t j = 0; j < 3; ++j) {
      state = state * 6364136223846793005ULL + 1442695040888963407ULL;
      points[i][j] = static_cast<float>(state >> 40) / 1000.0f;
    }
  }

  const KDTree<float> serialTree(points);
  const clustering::math::Pool pool{&clustering::math::sharedPool(4)};
  const KDTree<float> poolTree(points, pool);

  ASSERT_EQ(serialTree.nodeCount(), poolTree.nodeCount());
  const auto serialPerm = serialTree.indexPermutation();
  const auto poolPerm = poolTree.indexPermutation();
  ASSERT_EQ(serialPerm.size(), poolPerm.size());
  for (std::size_t i = 0; i < serialPerm.size(); ++i) {
    ASSERT_EQ(serialPerm[i], poolPerm[i]) << "permutation diverges at slot " << i;
  }

  const auto serialAdj = serialTree.query(2.0f, clustering::math::Pool{});
  const auto poolAdj = poolTree.query(2.0f, pool);
  ASSERT_EQ(serialAdj.size(), poolAdj.size());
  for (std::size_t i = 0; i < serialAdj.size(); ++i) {
    ASSERT_EQ(serialAdj[i], poolAdj[i]) << "adjacency diverges at row " << i;
  }
}
