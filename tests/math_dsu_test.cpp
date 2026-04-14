#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "clustering/math/dsu.h"

using clustering::UnionFind;

TEST(UnionFind, SingletonComponentsOnConstruction) {
  UnionFind uf(5);
  EXPECT_EQ(uf.size(), 5u);
  EXPECT_EQ(uf.countComponents(), 5u);
  for (std::uint32_t i = 0; i < 5; ++i) {
    EXPECT_EQ(uf.find(i), i);
  }
}

TEST(UnionFind, UniteMergesComponents) {
  UnionFind uf(4);
  EXPECT_TRUE(uf.unite(0, 1));
  EXPECT_EQ(uf.countComponents(), 3u);
  EXPECT_TRUE(uf.sameComponent(0, 1));
  EXPECT_FALSE(uf.sameComponent(0, 2));
}

TEST(UnionFind, UniteIsIdempotentOnSameComponent) {
  UnionFind uf(3);
  EXPECT_TRUE(uf.unite(0, 1));
  EXPECT_FALSE(uf.unite(0, 1));
  EXPECT_FALSE(uf.unite(1, 0));
  EXPECT_EQ(uf.countComponents(), 2u);
}

TEST(UnionFind, TransitiveMergeReducesToOneComponent) {
  UnionFind uf(5);
  uf.unite(0, 1);
  uf.unite(1, 2);
  uf.unite(2, 3);
  uf.unite(3, 4);
  EXPECT_EQ(uf.countComponents(), 1u);
  const auto root = uf.find(0);
  for (std::uint32_t i = 1; i < 5; ++i) {
    EXPECT_EQ(uf.find(i), root);
  }
}

TEST(UnionFind, PathCompressionFlattensTree) {
  // Build a linear chain 0 -> 1 -> 2 -> 3 -> 4 (as components). After a single find(0), every
  // node on the path should point directly at the root.
  UnionFind uf(5);
  uf.unite(0, 1);
  uf.unite(2, 3);
  uf.unite(0, 2);
  uf.unite(0, 4);
  const auto root = uf.find(0);
  for (std::uint32_t i = 0; i < 5; ++i) {
    (void)uf.find(i);
    EXPECT_EQ(uf.find(i), root);
  }
}

TEST(UnionFind, KruskalMstOnKnownGraph) {
  // Graph with 7 nodes and 9 weighted edges; the MST picks 6 edges totaling weight 39 and leaves
  // one component. Edge list (u, v, w):
  //   (0,1,7) (0,3,5) (1,2,8) (1,3,9) (1,4,7) (2,4,5) (3,4,15) (3,5,6) (4,5,8) (4,6,9) (5,6,11)
  // Canonical MST weight = 39 via (0,3,5), (2,4,5), (3,5,6), (0,1,7), (1,4,7), (4,6,9).
  struct Edge {
    std::uint32_t u;
    std::uint32_t v;
    int w;
  };
  std::vector<Edge> edges{{0, 1, 7},  {0, 3, 5}, {1, 2, 8}, {1, 3, 9}, {1, 4, 7}, {2, 4, 5},
                          {3, 4, 15}, {3, 5, 6}, {4, 5, 8}, {4, 6, 9}, {5, 6, 11}};
  std::ranges::sort(edges, [](const Edge &a, const Edge &b) noexcept { return a.w < b.w; });
  UnionFind uf(7);
  int total = 0;
  std::size_t picked = 0;
  for (const auto &e : edges) {
    if (uf.unite(e.u, e.v)) {
      total += e.w;
      ++picked;
    }
  }
  EXPECT_EQ(picked, 6u);
  EXPECT_EQ(total, 39);
  EXPECT_EQ(uf.countComponents(), 1u);
}

TEST(UnionFind, LargeNDoesNotStackOverflow) {
  // Iterative path compression must survive a degenerate chain of 1M links. A recursive find()
  // would blow the default stack here.
  constexpr std::size_t n = 1'000'000;
  UnionFind<std::uint32_t> uf(n);
  for (std::uint32_t i = 1; i < n; ++i) {
    uf.unite(i - 1, i);
  }
  EXPECT_EQ(uf.countComponents(), 1u);
  const auto root = uf.find(0);
  EXPECT_EQ(uf.find(static_cast<std::uint32_t>(n - 1)), root);
}

TEST(UnionFind, UnitReturnsFalseOnAlreadyJoinedAfterCompression) {
  UnionFind uf(6);
  uf.unite(0, 1);
  uf.unite(1, 2);
  uf.unite(3, 4);
  (void)uf.find(0);
  (void)uf.find(2);
  EXPECT_FALSE(uf.unite(0, 2));
  EXPECT_FALSE(uf.unite(2, 0));
  EXPECT_TRUE(uf.unite(2, 3));
  EXPECT_EQ(uf.countComponents(), 2u);
  EXPECT_TRUE(uf.unite(0, 5));
  EXPECT_EQ(uf.countComponents(), 1u);
}

TEST(UnionFind, Uint64IndexTypeWorks) {
  UnionFind<std::uint64_t> uf(4);
  uf.unite(0, 3);
  uf.unite(1, 2);
  EXPECT_TRUE(uf.sameComponent(0, 3));
  EXPECT_FALSE(uf.sameComponent(0, 1));
  EXPECT_EQ(uf.countComponents(), 2u);
}
