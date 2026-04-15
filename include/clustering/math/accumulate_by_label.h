#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <utility>
#include <vector>

#include "clustering/always_assert.h"
#include "clustering/math/thread.h"
#include "clustering/ndarray.h"

namespace clustering::math {

namespace detail {

/**
 * @brief Deterministic partitioning that mirrors @c BS::blocks -- used to derive a stable
 *        block index from @p lo inside a scatter task body.
 *
 * BS pool schedules tasks across workers in an order that is not guaranteed deterministic
 * across runs; indexing per-worker slabs by @c Pool::workerIndex() therefore breaks
 * bit-identity. Indexing by block (the deterministic partition of @c [0, n)) fixes this.
 */
struct BlockPartition {
  std::size_t first_index = 0;
  std::size_t block_size = 0;
  std::size_t remainder = 0;
  std::size_t num_blocks = 0;

  BlockPartition(std::size_t first, std::size_t n, std::size_t desired) noexcept
      : first_index(first) {
    if (n == 0 || desired == 0) {
      num_blocks = 0;
      return;
    }
    num_blocks = std::min(desired, n);
    block_size = n / num_blocks;
    remainder = n % num_blocks;
    if (block_size == 0) {
      block_size = 1;
      num_blocks = n;
    }
  }

  [[nodiscard]] std::size_t blockIndexOf(std::size_t lo) const noexcept {
    const std::size_t rel = lo - first_index;
    const std::size_t big = remainder * (block_size + 1);
    if (rel < big) {
      return rel / (block_size + 1);
    }
    return remainder + ((rel - big) / block_size);
  }
};

} // namespace detail

/**
 * @brief Sum rows of @p X grouped by label and count points per cluster.
 *
 * For each row @c i, accumulates @c X(i, :) into the cluster @c labels(i) and increments that
 * cluster's count. Labels outside @c [0, k) are skipped, letting callers mark noise (or
 * deleted rows) with a negative sentinel.
 *
 * Work decomposition: each block of the pool's partitioning maintains a @c (k, d) partial-sum
 * slab and a length-@c k partial-count vector. Blocks are indexed deterministically from the
 * @c lo boundary (not by worker id, which varies across runs), so the ascending-block-index
 * fold at the tail is bit-identical across repeated runs at fixed @c (n, k, d, workerCount).
 * No atomic float-add: at (k=1000, n=1e6) atomics contend on @c k cache lines across
 * workerCount writers per point; per-block partials + serial fold is measurably cheaper.
 *
 * Per-block slab memory: @c numBlocks * k * d * sizeof(T) bytes, allocated once per call.
 * The solver's scratch arena in Slice 2 will take over this allocation; for now the primitive
 * owns its scratch so it can be tested standalone.
 *
 * @tparam T Element type (@c float or @c double).
 * @param X           Data matrix (n x d), contiguous.
 * @param labels      Row-1 labels of length n.
 * @param k           Number of clusters.
 * @param outSums     Output sums (k x d); contents are overwritten.
 * @param outCounts   Output counts (length k); contents are overwritten.
 * @param pool        Parallelism injection.
 */
template <class T>
void accumulateByLabel(const NDArray<T, 2, Layout::Contig> &X,
                       const NDArray<std::int32_t, 1> &labels, std::size_t k,
                       NDArray<T, 2> &outSums, NDArray<std::int32_t, 1> &outCounts, Pool pool) {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                "accumulateByLabel<T> requires T to be float or double");

  CLUSTERING_ALWAYS_ASSERT(outSums.isMutable());
  CLUSTERING_ALWAYS_ASSERT(outCounts.isMutable());
  CLUSTERING_ALWAYS_ASSERT(labels.dim(0) == X.dim(0));
  CLUSTERING_ALWAYS_ASSERT(outSums.dim(0) == k);
  CLUSTERING_ALWAYS_ASSERT(outSums.dim(1) == X.dim(1));
  CLUSTERING_ALWAYS_ASSERT(outCounts.dim(0) == k);

  const std::size_t n = X.dim(0);
  const std::size_t d = X.dim(1);

  // Zero outputs so the fold only writes additive contributions.
  for (std::size_t c = 0; c < k; ++c) {
    outCounts(c) = 0;
    for (std::size_t t = 0; t < d; ++t) {
      outSums(c, t) = T{0};
    }
  }

  if (n == 0 || k == 0 || d == 0) {
    return;
  }

  const bool willParallelize = pool.shouldParallelize(n, 64, 2) && pool.pool != nullptr;
  const std::size_t blocks = willParallelize ? pool.workerCount() : std::size_t{1};
  const detail::BlockPartition part(0, n, blocks);
  const std::size_t numBlocks = part.num_blocks == 0 ? std::size_t{1} : part.num_blocks;

  std::vector<T, ::clustering::detail::AlignedAllocator<T, 32>> partialSums(numBlocks * k * d,
                                                                            T{0});
  std::vector<std::int32_t, ::clustering::detail::AlignedAllocator<std::int32_t, 32>> partialCounts(
      numBlocks * k, 0);

  auto scatterRange = [&](std::size_t lo, std::size_t hi) noexcept {
    const std::size_t b = part.blockIndexOf(lo);
    T *slab = partialSums.data() + (b * k * d);
    std::int32_t *cslab = partialCounts.data() + (b * k);
    for (std::size_t i = lo; i < hi; ++i) {
      const std::int32_t lbl = labels(i);
      if (lbl < 0 || std::cmp_greater_equal(lbl, k)) {
        continue;
      }
      const auto row = static_cast<std::size_t>(lbl);
      const T *xRow = X.data() + (i * d);
      T *dst = slab + (row * d);
      for (std::size_t t = 0; t < d; ++t) {
        dst[t] += xRow[t];
      }
      cslab[row] += 1;
    }
  };

  if (willParallelize) {
    pool.pool
        ->submit_blocks(
            std::size_t{0}, n, [&](std::size_t lo, std::size_t hi) { scatterRange(lo, hi); },
            numBlocks)
        .wait();
  } else {
    scatterRange(0, n);
  }

  // Ascending-block-index fold, single-threaded. The loop order is load-bearing: changing it
  // changes the reduction tree and therefore the last-bit of the output sums, breaking the
  // bit-identity guarantee that identical (seed, nJobs) inputs produce identical outputs.
  for (std::size_t b = 0; b < numBlocks; ++b) {
    const T *slab = partialSums.data() + (b * k * d);
    const std::int32_t *cslab = partialCounts.data() + (b * k);
    for (std::size_t c = 0; c < k; ++c) {
      outCounts(c) += cslab[c];
      const T *src = slab + (c * d);
      for (std::size_t t = 0; t < d; ++t) {
        outSums(c, t) += src[t];
      }
    }
  }
}

/**
 * @brief Kahan-compensated variant of @ref accumulateByLabel.
 *
 * Carries a per-cluster per-d compensation array alongside the running sum. At every addend
 * step, @c c absorbs the low-order bits lost when a small addend combines with a large
 * running total. The compensation is carried through both the per-worker scatter and the
 * ascending-worker-index fold, so the final result is recoverable at f32 precision even
 * when a dominant addend is combined with many small ones.
 *
 * Fold order is the same ascending-worker-index serial loop as the plain variant; the Kahan
 * `c` is re-initialized fresh at the fold's outer loop and updated per added worker.
 *
 * @tparam T Element type (@c float or @c double).
 * @param X           Data matrix (n x d), contiguous.
 * @param labels      Row-1 labels of length n.
 * @param k           Number of clusters.
 * @param outSums     Output sums (k x d); contents are overwritten.
 * @param outCounts   Output counts (length k); contents are overwritten.
 * @param pool        Parallelism injection.
 */
template <class T>
void accumulateByLabelKahan(const NDArray<T, 2, Layout::Contig> &X,
                            const NDArray<std::int32_t, 1> &labels, std::size_t k,
                            NDArray<T, 2> &outSums, NDArray<std::int32_t, 1> &outCounts,
                            Pool pool) {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                "accumulateByLabelKahan<T> requires T to be float or double");

  CLUSTERING_ALWAYS_ASSERT(outSums.isMutable());
  CLUSTERING_ALWAYS_ASSERT(outCounts.isMutable());
  CLUSTERING_ALWAYS_ASSERT(labels.dim(0) == X.dim(0));
  CLUSTERING_ALWAYS_ASSERT(outSums.dim(0) == k);
  CLUSTERING_ALWAYS_ASSERT(outSums.dim(1) == X.dim(1));
  CLUSTERING_ALWAYS_ASSERT(outCounts.dim(0) == k);

  const std::size_t n = X.dim(0);
  const std::size_t d = X.dim(1);

  for (std::size_t c = 0; c < k; ++c) {
    outCounts(c) = 0;
    for (std::size_t t = 0; t < d; ++t) {
      outSums(c, t) = T{0};
    }
  }

  if (n == 0 || k == 0 || d == 0) {
    return;
  }

  const bool willParallelize = pool.shouldParallelize(n, 64, 2) && pool.pool != nullptr;
  const std::size_t desiredBlocks = willParallelize ? pool.workerCount() : std::size_t{1};
  const detail::BlockPartition part(0, n, desiredBlocks);
  const std::size_t numBlocks = part.num_blocks == 0 ? std::size_t{1} : part.num_blocks;

  std::vector<T, ::clustering::detail::AlignedAllocator<T, 32>> partialSums(numBlocks * k * d,
                                                                            T{0});
  std::vector<T, ::clustering::detail::AlignedAllocator<T, 32>> partialComps(numBlocks * k * d,
                                                                             T{0});
  std::vector<std::int32_t, ::clustering::detail::AlignedAllocator<std::int32_t, 32>> partialCounts(
      numBlocks * k, 0);

  auto scatterRange = [&](std::size_t lo, std::size_t hi) noexcept {
    const std::size_t b = part.blockIndexOf(lo);
    T *slab = partialSums.data() + (b * k * d);
    T *cslab = partialComps.data() + (b * k * d);
    std::int32_t *nslab = partialCounts.data() + (b * k);
    for (std::size_t i = lo; i < hi; ++i) {
      const std::int32_t lbl = labels(i);
      if (lbl < 0 || std::cmp_greater_equal(lbl, k)) {
        continue;
      }
      const auto row = static_cast<std::size_t>(lbl);
      const T *xRow = X.data() + (i * d);
      T *sumRow = slab + (row * d);
      T *compRow = cslab + (row * d);
      for (std::size_t t = 0; t < d; ++t) {
        const T y = xRow[t] - compRow[t];
        const T tVal = sumRow[t] + y;
        compRow[t] = (tVal - sumRow[t]) - y;
        sumRow[t] = tVal;
      }
      nslab[row] += 1;
    }
  };

  if (willParallelize) {
    pool.pool
        ->submit_blocks(
            std::size_t{0}, n, [&](std::size_t lo, std::size_t hi) { scatterRange(lo, hi); },
            numBlocks)
        .wait();
  } else {
    scatterRange(0, n);
  }

  // Ascending-block-index fold with Kahan compensation. The compensation `c` is fresh per
  // (cluster, dim) pair in the fold; each block contributes (partialSum - partialComp) to
  // the final sum, so the compensation computed in the scatter pass is rolled back into the
  // addend once to recover the bits lost before the fold begins.
  std::vector<T> foldComp(k * d, T{0});
  for (std::size_t b = 0; b < numBlocks; ++b) {
    const T *slab = partialSums.data() + (b * k * d);
    const T *cslab = partialComps.data() + (b * k * d);
    const std::int32_t *nslab = partialCounts.data() + (b * k);
    for (std::size_t c = 0; c < k; ++c) {
      outCounts(c) += nslab[c];
      const T *src = slab + (c * d);
      const T *comp = cslab + (c * d);
      T *dstRow = &outSums(c, 0);
      T *foldRow = foldComp.data() + (c * d);
      for (std::size_t t = 0; t < d; ++t) {
        const T addend = src[t] - comp[t];
        const T y = addend - foldRow[t];
        const T tVal = dstRow[t] + y;
        foldRow[t] = (tVal - dstRow[t]) - y;
        dstRow[t] = tVal;
      }
    }
  }
}

} // namespace clustering::math
