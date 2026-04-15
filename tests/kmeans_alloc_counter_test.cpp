#include <gtest/gtest.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <random>

#include "clustering/kmeans.h"
#include "clustering/ndarray.h"

using clustering::KMeans;
using clustering::NDArray;

namespace {

NDArray<float, 2> makeData(std::size_t n, std::size_t d, std::uint32_t seed) {
  NDArray<float, 2> X({n, d});
  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> dist(-2.0F, 2.0F);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t t = 0; t < d; ++t) {
      X(i, t) = dist(gen);
    }
  }
  return X;
}

} // namespace

// First run lazy-resizes scratch at the top of run() -- those allocations are tolerated. Second
// run at the identical (n, d, k, nJobs) shape must not allocate anywhere along the Lloyd path:
// no reallocation, no seeder scratch, no convergence-check scratch.
TEST(KMeansAllocCounter, NoAllocationInWarmRun) {
  constexpr std::size_t n = 256;
  constexpr std::size_t d = 8;
  constexpr std::size_t k = 4;
  const NDArray<float, 2> X = makeData(n, d, 42U);

  KMeans<float> km(k, 1);
  // First run: allocates scratch lazily. Not gated.
  km.run(X, 50, 1e-4F, 13U);

  auto &counter = clustering::detail::alignedAllocCallCount();
  const std::uint64_t before = counter.load(std::memory_order_relaxed);
  km.run(X, 50, 1e-4F, 17U);
  const std::uint64_t after = counter.load(std::memory_order_relaxed);

  EXPECT_EQ(after, before) << "warm run allocated " << (after - before) << " times";
}

// Mid-dim shape that routes through the chunked materialized path. Warm iterations share the
// solver's pre-sized distance tile, so the iteration loop sees zero alignedAlloc events.
TEST(KMeansAllocCounter, NoAllocationInWarmRunAtMidDim) {
  constexpr std::size_t n = 1000;
  constexpr std::size_t d = 32;
  constexpr std::size_t k = 64;
  const NDArray<float, 2> X = makeData(n, d, 123U);

  KMeans<float> km(k, 1);
  km.run(X, 20, 1e-4F, 13U);

  auto &counter = clustering::detail::alignedAllocCallCount();
  const std::uint64_t before = counter.load(std::memory_order_relaxed);
  km.run(X, 20, 1e-4F, 17U);
  const std::uint64_t after = counter.load(std::memory_order_relaxed);

  EXPECT_EQ(after, before) << "warm run allocated " << (after - before) << " times";
}

// Bounding the first-run allocation count so a future regression that adds an unintended
// per-iteration alloc shows up as a linear-in-maxIter blowup.
TEST(KMeansAllocCounter, FirstRunAllocCountIndependentOfMaxIter) {
  constexpr std::size_t n = 256;
  constexpr std::size_t d = 8;
  constexpr std::size_t k = 4;
  const NDArray<float, 2> X = makeData(n, d, 7U);

  auto &counter = clustering::detail::alignedAllocCallCount();

  KMeans<float> kmLow(k, 1);
  const std::uint64_t beforeLow = counter.load(std::memory_order_relaxed);
  kmLow.run(X, 5, 1e-4F, 0U);
  const std::uint64_t deltaLow = counter.load(std::memory_order_relaxed) - beforeLow;

  KMeans<float> kmHigh(k, 1);
  const std::uint64_t beforeHigh = counter.load(std::memory_order_relaxed);
  kmHigh.run(X, 200, 1e-4F, 0U);
  const std::uint64_t deltaHigh = counter.load(std::memory_order_relaxed) - beforeHigh;

  // If a per-iteration allocation leaked in, deltaHigh would scale with 200/5 = 40x. A small
  // constant budget over the warm baseline is fine (seeder staging vectors, etc.).
  EXPECT_LE(deltaHigh, deltaLow + 16U) << "deltaLow=" << deltaLow << " deltaHigh=" << deltaHigh;
}
