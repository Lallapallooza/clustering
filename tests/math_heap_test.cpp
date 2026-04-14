#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include "clustering/math/heap.h"

using clustering::BinaryHeap;
using clustering::IndexedHeap;

TEST(BinaryHeap, EmptyOnConstruction) {
  const BinaryHeap<double, int> h;
  EXPECT_TRUE(h.empty());
  EXPECT_EQ(h.size(), 0u);
}

TEST(BinaryHeap, PushPopReturnsMinFirst) {
  BinaryHeap<int, int> h;
  h.push(5, 50);
  h.push(1, 10);
  h.push(4, 40);
  h.push(2, 20);
  h.push(3, 30);
  EXPECT_EQ(h.size(), 5u);
  for (const int expected : {1, 2, 3, 4, 5}) {
    ASSERT_FALSE(h.empty());
    EXPECT_EQ(h.top().first, expected);
    EXPECT_EQ(h.top().second, expected * 10);
    h.pop();
  }
  EXPECT_TRUE(h.empty());
}

TEST(BinaryHeap, DuplicateKeysBothPopped) {
  BinaryHeap<int, int> h;
  h.push(2, 1);
  h.push(2, 2);
  h.push(1, 3);
  EXPECT_EQ(h.top().first, 1);
  h.pop();
  EXPECT_EQ(h.top().first, 2);
  h.pop();
  EXPECT_EQ(h.top().first, 2);
  h.pop();
  EXPECT_TRUE(h.empty());
}

TEST(BinaryHeap, HoldsUpTo1kElementsInOrder) {
  BinaryHeap<int, int> h;
  for (int v = 1000; v > 0; --v) {
    h.push(v, v);
  }
  for (int expected = 1; expected <= 1000; ++expected) {
    ASSERT_EQ(h.top().first, expected);
    h.pop();
  }
  EXPECT_TRUE(h.empty());
}

TEST(IndexedHeap, EmptyOnConstruction) {
  const IndexedHeap<double, int> h(8);
  EXPECT_TRUE(h.empty());
  EXPECT_EQ(h.size(), 0u);
  EXPECT_EQ(h.capacity(), 8u);
  for (std::uint32_t i = 0; i < 8; ++i) {
    EXPECT_FALSE(h.contains(i));
  }
}

TEST(IndexedHeap, PushContainsPop) {
  IndexedHeap<int, int> h(5);
  h.push(3, 30, 300);
  h.push(1, 10, 100);
  h.push(4, 40, 400);
  h.push(0, 0, 0);
  h.push(2, 20, 200);
  EXPECT_EQ(h.size(), 5u);
  for (std::uint32_t i = 0; i < 5; ++i) {
    EXPECT_TRUE(h.contains(i));
  }
  const auto top = h.pop();
  EXPECT_EQ(top.handle, 0u);
  EXPECT_EQ(top.key, 0);
  EXPECT_EQ(top.val, 0);
  EXPECT_FALSE(h.contains(0));
  EXPECT_EQ(h.size(), 4u);

  for (const std::uint32_t expected : {1u, 2u, 3u, 4u}) {
    const auto e = h.pop();
    EXPECT_EQ(e.handle, expected);
  }
  EXPECT_TRUE(h.empty());
}

TEST(IndexedHeap, DecreaseKeyMovesEntryUp) {
  IndexedHeap<int, int> h(5);
  h.push(0, 100, 0);
  h.push(1, 50, 1);
  h.push(2, 75, 2);
  h.push(3, 25, 3);
  h.push(4, 10, 4);
  // Current min is handle 4 (key 10). Lower handle 0's key from 100 to 5 -- should now be top.
  h.decreaseKey(0, 5);
  EXPECT_EQ(h.top().handle, 0u);
  EXPECT_EQ(h.top().key, 5);
  h.pop();
  EXPECT_EQ(h.top().handle, 4u);
}

TEST(IndexedHeap, DecreaseKeyToSameKeyIsNoop) {
  IndexedHeap<int, int> h(3);
  h.push(0, 10, 0);
  h.push(1, 5, 1);
  h.push(2, 20, 2);
  h.decreaseKey(0, 10);
  EXPECT_EQ(h.top().handle, 1u);
}

TEST(IndexedHeap, PopClearsHandleSlot) {
  IndexedHeap<int, int> h(3);
  h.push(0, 1, 0);
  EXPECT_TRUE(h.contains(0));
  (void)h.pop();
  EXPECT_FALSE(h.contains(0));
  // Handle can now be reused.
  h.push(0, 99, 0);
  EXPECT_TRUE(h.contains(0));
  EXPECT_EQ(h.top().key, 99);
}

TEST(IndexedHeap, DijkstraShortestPathsOnKnownGraph) {
  // Graph with 5 nodes and 8 directed edges. Source = 0. Known shortest-path distances:
  //   d[0]=0, d[1]=7, d[2]=9, d[3]=20, d[4]=11.
  // Edge list (u, v, w):
  //   (0,1,7) (0,2,9) (0,4,14) (1,2,10) (1,3,15) (2,3,11) (2,4,2) (3,4,6)
  struct Edge {
    std::uint32_t to;
    int w;
  };
  constexpr std::size_t n = 5;
  std::vector<std::vector<Edge>> adj(n);
  adj[0].push_back({1, 7});
  adj[0].push_back({2, 9});
  adj[0].push_back({4, 14});
  adj[1].push_back({2, 10});
  adj[1].push_back({3, 15});
  adj[2].push_back({3, 11});
  adj[2].push_back({4, 2});
  adj[3].push_back({4, 6});

  constexpr int kInf = std::numeric_limits<int>::max();
  std::vector<int> dist(n, kInf);
  dist[0] = 0;
  IndexedHeap<int, std::uint32_t> pq(n);
  pq.push(0, 0, 0);

  while (!pq.empty()) {
    const auto e = pq.pop();
    const std::uint32_t u = e.handle;
    if (e.key > dist[u]) {
      continue;
    }
    for (const auto &edge : adj[u]) {
      const int alt = dist[u] + edge.w;
      if (alt < dist[edge.to]) {
        dist[edge.to] = alt;
        if (pq.contains(edge.to)) {
          pq.decreaseKey(edge.to, alt);
        } else {
          pq.push(edge.to, alt, edge.to);
        }
      }
    }
  }

  EXPECT_EQ(dist[0], 0);
  EXPECT_EQ(dist[1], 7);
  EXPECT_EQ(dist[2], 9);
  EXPECT_EQ(dist[3], 20);
  EXPECT_EQ(dist[4], 11);
}

TEST(IndexedHeap, RepeatedDecreaseKeyConverges) {
  // Decrease the key of one handle many times in succession; each should move it progressively
  // closer to the root. Final state: that handle is the top.
  IndexedHeap<int, int> h(10);
  for (std::uint32_t i = 0; i < 10; ++i) {
    h.push(i, 100 + static_cast<int>(i), static_cast<int>(i));
  }
  for (int k = 99; k >= 0; k -= 5) {
    h.decreaseKey(9, k);
  }
  // Loop visits 99, 94, 89, ..., 4. Last stored key is the final decreaseKey arg, which is 4.
  EXPECT_EQ(h.top().handle, 9u);
  EXPECT_EQ(h.top().key, 4);
}

#ifndef NDEBUG
TEST(IndexedHeapDeathTest, PushDuplicateHandleAsserts) {
  IndexedHeap<int, int> h(3);
  h.push(1, 5, 5);
  EXPECT_DEATH({ h.push(1, 10, 10); }, "");
}

TEST(IndexedHeapDeathTest, DecreaseKeyOnAbsentHandleAsserts) {
  IndexedHeap<int, int> h(3);
  EXPECT_DEATH({ h.decreaseKey(0, 5); }, "");
}

TEST(IndexedHeapDeathTest, DecreaseKeyRaisesAsserts) {
  IndexedHeap<int, int> h(3);
  h.push(0, 5, 0);
  EXPECT_DEATH({ h.decreaseKey(0, 10); }, "");
}
#endif
