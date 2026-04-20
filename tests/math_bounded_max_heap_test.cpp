#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <random>
#include <utility>
#include <vector>

#include "clustering/math/detail/bounded_max_heap.h"

using clustering::math::detail::BoundedMaxHeap;

namespace {

// Drain the heap into a vector of (key, val) pairs in decreasing-key-then-decreasing-val order
// (root out first).
template <class Key, class Val>
std::vector<std::pair<Key, Val>> drain(BoundedMaxHeap<Key, Val> &h) {
  std::vector<std::pair<Key, Val>> out;
  while (!h.empty()) {
    out.push_back(h.top());
    h.pop();
  }
  return out;
}

} // namespace

TEST(BoundedMaxHeap, EmptyOnConstruction) {
  const BoundedMaxHeap<int, int> h(4);
  EXPECT_TRUE(h.empty());
  EXPECT_EQ(h.size(), 0u);
  EXPECT_EQ(h.capacity(), 4u);
}

TEST(BoundedMaxHeap, RetainsAllUnderCapacity) {
  BoundedMaxHeap<int, int> h(5);
  h.push(3, 30);
  h.push(1, 10);
  h.push(2, 20);
  EXPECT_EQ(h.size(), 3u);
  EXPECT_EQ(h.top().first, 3);
  EXPECT_EQ(h.top().second, 30);
  const auto ordered = drain(h);
  ASSERT_EQ(ordered.size(), 3u);
  EXPECT_EQ(ordered[0].first, 3);
  EXPECT_EQ(ordered[1].first, 2);
  EXPECT_EQ(ordered[2].first, 1);
}

TEST(BoundedMaxHeap, RetainsKSmallestWhenStreamExceedsCapacity) {
  // Insert 10 keys into a capacity-3 heap; the retained set should be {1,2,3}.
  BoundedMaxHeap<int, int> h(3);
  for (const int k : {7, 4, 1, 9, 3, 6, 2, 5, 8, 10}) {
    h.push(k, k * 100);
  }
  EXPECT_EQ(h.size(), 3u);
  EXPECT_EQ(h.top().first, 3);
  const auto ordered = drain(h);
  ASSERT_EQ(ordered.size(), 3u);
  EXPECT_EQ(ordered[0].first, 3);
  EXPECT_EQ(ordered[0].second, 300);
  EXPECT_EQ(ordered[1].first, 2);
  EXPECT_EQ(ordered[1].second, 200);
  EXPECT_EQ(ordered[2].first, 1);
  EXPECT_EQ(ordered[2].second, 100);
}

TEST(BoundedMaxHeap, MatchesStdPartialSortOnRandomStream) {
  // Property-style: a capacity-k bounded-max-heap fed an arbitrary stream retains the same k
  // keys as std::partial_sort over the same stream, modulo tie-break on Val.
  constexpr std::size_t kCap = 7;
  std::mt19937 gen(0xC0FFEEu);
  std::uniform_int_distribution<int> pick(0, 999);
  std::vector<int> keys(128);
  for (int &k : keys) {
    k = pick(gen);
  }

  BoundedMaxHeap<int, int> h(kCap);
  for (std::size_t i = 0; i < keys.size(); ++i) {
    h.push(keys[i], static_cast<int>(i));
  }

  std::vector<int> sorted = keys;
  std::partial_sort(sorted.begin(), sorted.begin() + kCap, sorted.end());
  std::vector<int> expected(sorted.begin(), sorted.begin() + kCap);
  std::sort(expected.begin(), expected.end(), std::greater<>());

  const auto ordered = drain(h);
  ASSERT_EQ(ordered.size(), kCap);
  std::vector<int> drainedKeys;
  drainedKeys.reserve(ordered.size());
  for (const auto &p : ordered) {
    drainedKeys.push_back(p.first);
  }
  EXPECT_EQ(drainedKeys, expected);
}

TEST(BoundedMaxHeap, TieBreakRetainsSmallerVal) {
  // With equal keys, the heap must retain the entries with the smaller Val. Insert six entries
  // at the same key with distinct Vals into a capacity-3 heap; the retained Vals must be the
  // three smallest.
  BoundedMaxHeap<int, int> h(3);
  for (const int v : {5, 2, 4, 1, 3, 0}) {
    h.push(42, v);
  }
  EXPECT_EQ(h.size(), 3u);
  // Root is the largest-Val retained among the smallest three: {0, 1, 2} => root Val is 2.
  EXPECT_EQ(h.top().first, 42);
  EXPECT_EQ(h.top().second, 2);
  const auto ordered = drain(h);
  ASSERT_EQ(ordered.size(), 3u);
  EXPECT_EQ(ordered[0].second, 2);
  EXPECT_EQ(ordered[1].second, 1);
  EXPECT_EQ(ordered[2].second, 0);
}

TEST(BoundedMaxHeap, TieBreakIgnoresEqualKeyEqualOrLargerVal) {
  // Once the heap is full with a single (key, val), a subsequent push with the same key and a
  // larger Val must NOT replace the retained entry. Equal key AND equal Val is also a no-op.
  BoundedMaxHeap<int, int> h(1);
  h.push(10, 5);
  EXPECT_EQ(h.top().second, 5);
  h.push(10, 9);
  EXPECT_EQ(h.top().second, 5);
  h.push(10, 5);
  EXPECT_EQ(h.top().second, 5);
  h.push(10, 3);
  EXPECT_EQ(h.top().second, 3);
}

TEST(BoundedMaxHeap, ClearEmptiesButKeepsCapacity) {
  BoundedMaxHeap<int, int> h(4);
  h.push(1, 1);
  h.push(2, 2);
  h.push(3, 3);
  EXPECT_EQ(h.size(), 3u);
  h.clear();
  EXPECT_TRUE(h.empty());
  EXPECT_EQ(h.capacity(), 4u);
  h.push(99, 99);
  EXPECT_EQ(h.size(), 1u);
  EXPECT_EQ(h.top().first, 99);
}

TEST(BoundedMaxHeap, CapacityOneKeepsMinimum) {
  // With capacity 1, a stream of keys should leave the minimum key as the retained entry.
  BoundedMaxHeap<int, int> h(1);
  for (const int k : {5, 3, 7, 1, 9, 2}) {
    h.push(k, k);
  }
  EXPECT_EQ(h.size(), 1u);
  EXPECT_EQ(h.top().first, 1);
}

TEST(BoundedMaxHeap, CapacityZeroIgnoresPushes) {
  BoundedMaxHeap<int, int> h(0);
  h.push(1, 2);
  h.push(3, 4);
  EXPECT_TRUE(h.empty());
  EXPECT_EQ(h.capacity(), 0u);
}

TEST(BoundedMaxHeap, SameKeyNInsertions) {
  // Capacity 4, stream of 10 entries all at the same key but distinct Val: the four smallest
  // Vals are retained.
  BoundedMaxHeap<double, std::int32_t> h(4);
  for (std::int32_t i = 0; i < 10; ++i) {
    h.push(1.0, i);
  }
  EXPECT_EQ(h.size(), 4u);
  const auto ordered = drain(h);
  ASSERT_EQ(ordered.size(), 4u);
  EXPECT_EQ(ordered[0].second, 3);
  EXPECT_EQ(ordered[1].second, 2);
  EXPECT_EQ(ordered[2].second, 1);
  EXPECT_EQ(ordered[3].second, 0);
}

TEST(BoundedMaxHeap, DeterministicAcrossRuns) {
  // Same input stream must produce identical retained contents across two heap instances. This
  // exercises the reproducibility contract load-bearing for kNN.
  std::vector<std::pair<double, std::int32_t>> stream;
  stream.reserve(264);
  std::mt19937 gen(0xD1CEu);
  std::uniform_real_distribution<double> keyDist(0.0, 10.0);
  for (std::int32_t i = 0; i < 256; ++i) {
    stream.emplace_back(keyDist(gen), i);
  }
  // Inject a cluster of ties into the stream.
  for (std::int32_t i = 0; i < 8; ++i) {
    stream.emplace_back(3.0, 1000 + i);
  }

  BoundedMaxHeap<double, std::int32_t> a(5);
  BoundedMaxHeap<double, std::int32_t> b(5);
  for (const auto &p : stream) {
    a.push(p.first, p.second);
    b.push(p.first, p.second);
  }

  const auto drainedA = drain(a);
  const auto drainedB = drain(b);
  ASSERT_EQ(drainedA.size(), drainedB.size());
  for (std::size_t i = 0; i < drainedA.size(); ++i) {
    EXPECT_EQ(drainedA[i].first, drainedB[i].first) << "i=" << i;
    EXPECT_EQ(drainedA[i].second, drainedB[i].second) << "i=" << i;
  }
}
