#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>

#include "clustering/always_assert.h"
#include "clustering/kmeans/detail/convergence.h"
#include "clustering/kmeans/detail/dispatch.h"
#include "clustering/kmeans/detail/empty_cluster.h"
#include "clustering/math/accumulate_by_label.h"
#include "clustering/math/centroid_shift.h"
#include "clustering/math/defaults.h"
#include "clustering/math/detail/pairwise_argmin_outer.h"
#include "clustering/math/pairwise.h"
#include "clustering/math/pairwise_argmin.h"
#include "clustering/math/thread.h"
#include "clustering/ndarray.h"

namespace clustering::kmeans::detail {

/**
 * @brief Contiguous views over scratch owned by @c Solver<T>.
 *
 * All buffers are sized against the current @c (n, d, k, workerCount) shape tuple. The Lloyd
 * driver in @ref runLloydFused treats them strictly as preallocated working space: no
 * allocation fires between the first assignment call and the convergence check of any
 * iteration.
 */
template <class T> struct LloydScratch {
  NDArray<T, 2, Layout::Contig> *centroids;    ///< Current centroids (k x d).
  NDArray<T, 2, Layout::Contig> *centroidsOld; ///< Previous-iteration centroids (k x d).
  NDArray<T, 1> *cSqNorms;                     ///< Centroid row norms squared (k).
  NDArray<T, 2, Layout::Contig> *sums;         ///< Per-cluster coordinate sums (k x d).
  NDArray<std::int32_t, 1> *counts;            ///< Per-cluster point counts (k).
  NDArray<std::int32_t, 1> *labels;            ///< Current assignment (n).
  NDArray<T, 1> *minDistSq;                    ///< Per-point squared distance to assignment (n).
  NDArray<T, 1> *shiftSq;                      ///< Per-centroid squared shift (k).

  /// Per-block partial sums for the label-grouped fold (@c numBlocks * k * d).
  NDArray<T, 1> *partialSums;
  /// Kahan compensation slab paired with @c partialSums; unused when the plain variant runs.
  NDArray<T, 1> *partialComps;
  /// Per-block partial counts (@c numBlocks * k).
  NDArray<std::int32_t, 1> *partialCounts;
  /// Kahan roll-up compensation for the serial fold pass (@c k * d).
  NDArray<T, 1> *foldComp;
  /// Packed-B scratch for the fused argmin hot path. Sized via @c packedBScratchSizeFloats.
  NDArray<T, 1> *packedB;
  /// Packed centroid-norms scratch paired with @c packedB.
  NDArray<T, 1> *packedCSqNorms;
};

/**
 * @brief Run the assignment step against caller-owned packed scratch when available.
 *
 * Routes to the AVX2 fused outer with caller-supplied scratch when the precondition set
 * matches (@c T == float, contiguous + 32-byte aligned, @c d <= @c pairwiseArgminMaxD).
 * Otherwise falls back to the public entry at @c math::pairwiseArgminSqEuclidean, which
 * picks its own path. The fallback is the only path that may allocate during the Lloyd
 * loop; it fires at @c d > 256 and is documented in the contract as out-of-envelope.
 */
template <class T>
void runAssignment(const NDArray<T, 2, Layout::Contig> &X, const LloydScratch<T> &scratch,
                   math::Pool pool) {
#ifdef CLUSTERING_USE_AVX2
  if constexpr (std::is_same_v<T, float>) {
    const auto &centroids = *scratch.centroids;
    const auto &cSqNorms = *scratch.cSqNorms;
    if (X.template isAligned<32>() && centroids.template isAligned<32>() && X.dim(1) != 0 &&
        X.dim(1) <= math::defaults::pairwiseArgminMaxD) {
      math::detail::pairwiseArgminOuterAvx2F32WithScratch(
          X, centroids, cSqNorms, *scratch.labels, *scratch.minDistSq, scratch.packedB->data(),
          scratch.packedCSqNorms->data(), pool);
      return;
    }
  }
#endif
  math::pairwiseArgminSqEuclidean<T>(X, *scratch.centroids, *scratch.cSqNorms, *scratch.labels,
                                     *scratch.minDistSq, pool);
}

/**
 * @brief Scatter rows of @p X into per-block partials keyed by @p labels, then fold.
 *
 * Logically equivalent to @c math::accumulateByLabel but reads its working arrays from the
 * caller-owned @p scratch so the Lloyd driver stays allocation-free across iterations. The
 * block-partition scheme is the same deterministic fold the standalone primitive uses, which
 * keeps per-nJobs bit-identity with tests that exercise the free-function form.
 */
template <class T>
void scatterAndFoldPlain(const NDArray<T, 2, Layout::Contig> &X, const LloydScratch<T> &scratch,
                         std::size_t k, math::Pool pool) {
  const std::size_t n = X.dim(0);
  const std::size_t d = X.dim(1);

  NDArray<T, 2, Layout::Contig> &sums = *scratch.sums;
  NDArray<std::int32_t, 1> &counts = *scratch.counts;
  const NDArray<std::int32_t, 1> &labels = *scratch.labels;
  T *partialSums = scratch.partialSums->data();
  std::int32_t *partialCounts = scratch.partialCounts->data();

  for (std::size_t c = 0; c < k; ++c) {
    counts(c) = 0;
    for (std::size_t t = 0; t < d; ++t) {
      sums(c, t) = T{0};
    }
  }
  if (n == 0 || d == 0) {
    return;
  }

  const bool willParallelize = pool.shouldParallelize(n, 64, 2) && pool.pool != nullptr;
  const std::size_t desiredBlocks = willParallelize ? pool.workerCount() : std::size_t{1};
  const math::detail::BlockPartition part(0, n, desiredBlocks);
  const std::size_t numBlocks = part.num_blocks == 0 ? std::size_t{1} : part.num_blocks;

  // Zero the per-block tiles we intend to use this pass. The slab is sized at shape-change
  // time to @c workerCount * k * d, so we only clear the blocks we will scatter into.
  for (std::size_t b = 0; b < numBlocks; ++b) {
    T *slab = partialSums + (b * k * d);
    std::int32_t *cslab = partialCounts + (b * k);
    for (std::size_t e = 0; e < k * d; ++e) {
      slab[e] = T{0};
    }
    for (std::size_t c = 0; c < k; ++c) {
      cslab[c] = 0;
    }
  }

  auto scatterRange = [&](std::size_t lo, std::size_t hi) noexcept {
    const std::size_t b = part.blockIndexOf(lo);
    T *slab = partialSums + (b * k * d);
    std::int32_t *cslab = partialCounts + (b * k);
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

  // Ascending-block-index fold. Deterministic at fixed (n, k, d, nJobs); changing this order
  // changes the last-bit of the per-cluster sum and breaks the bit-identity guarantee.
  for (std::size_t b = 0; b < numBlocks; ++b) {
    const T *slab = partialSums + (b * k * d);
    const std::int32_t *cslab = partialCounts + (b * k);
    for (std::size_t c = 0; c < k; ++c) {
      counts(c) += cslab[c];
      const T *src = slab + (c * d);
      T *dstRow = &sums(c, 0);
      for (std::size_t t = 0; t < d; ++t) {
        dstRow[t] += src[t];
      }
    }
  }
}

/**
 * @brief Kahan-compensated variant of @ref scatterAndFoldPlain.
 *
 * Carries a per-cluster per-d compensation slab alongside each block's running sum and
 * rolls it into the serial fold. Used when @c n >= @ref kahanNThreshold.
 */
template <class T>
void scatterAndFoldKahan(const NDArray<T, 2, Layout::Contig> &X, const LloydScratch<T> &scratch,
                         std::size_t k, math::Pool pool) {
  const std::size_t n = X.dim(0);
  const std::size_t d = X.dim(1);

  NDArray<T, 2, Layout::Contig> &sums = *scratch.sums;
  NDArray<std::int32_t, 1> &counts = *scratch.counts;
  const NDArray<std::int32_t, 1> &labels = *scratch.labels;
  T *partialSums = scratch.partialSums->data();
  T *partialComps = scratch.partialComps->data();
  std::int32_t *partialCounts = scratch.partialCounts->data();
  T *foldComp = scratch.foldComp->data();

  for (std::size_t c = 0; c < k; ++c) {
    counts(c) = 0;
    for (std::size_t t = 0; t < d; ++t) {
      sums(c, t) = T{0};
    }
  }
  for (std::size_t e = 0; e < k * d; ++e) {
    foldComp[e] = T{0};
  }
  if (n == 0 || d == 0) {
    return;
  }

  const bool willParallelize = pool.shouldParallelize(n, 64, 2) && pool.pool != nullptr;
  const std::size_t desiredBlocks = willParallelize ? pool.workerCount() : std::size_t{1};
  const math::detail::BlockPartition part(0, n, desiredBlocks);
  const std::size_t numBlocks = part.num_blocks == 0 ? std::size_t{1} : part.num_blocks;

  for (std::size_t b = 0; b < numBlocks; ++b) {
    T *slab = partialSums + (b * k * d);
    T *cslab = partialComps + (b * k * d);
    std::int32_t *nslab = partialCounts + (b * k);
    for (std::size_t e = 0; e < k * d; ++e) {
      slab[e] = T{0};
      cslab[e] = T{0};
    }
    for (std::size_t c = 0; c < k; ++c) {
      nslab[c] = 0;
    }
  }

  auto scatterRange = [&](std::size_t lo, std::size_t hi) noexcept {
    const std::size_t b = part.blockIndexOf(lo);
    T *slab = partialSums + (b * k * d);
    T *cslab = partialComps + (b * k * d);
    std::int32_t *nslab = partialCounts + (b * k);
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

  for (std::size_t b = 0; b < numBlocks; ++b) {
    const T *slab = partialSums + (b * k * d);
    const T *cslab = partialComps + (b * k * d);
    const std::int32_t *nslab = partialCounts + (b * k);
    for (std::size_t c = 0; c < k; ++c) {
      counts(c) += nslab[c];
      const T *src = slab + (c * d);
      const T *comp = cslab + (c * d);
      T *dstRow = &sums(c, 0);
      T *foldRow = foldComp + (c * d);
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

/**
 * @brief Divide per-cluster sums by their counts to recover mean-of-assignment centroids.
 *
 * Clusters with a count of zero are left alone; the caller is expected to have reseeded
 * any empty cluster before calling this routine so that no division by zero occurs.
 */
template <class T>
void finalizeMeans(NDArray<T, 2, Layout::Contig> &centroids,
                   const NDArray<T, 2, Layout::Contig> &sums,
                   const NDArray<std::int32_t, 1> &counts) noexcept {
  const std::size_t k = centroids.dim(0);
  const std::size_t d = centroids.dim(1);
  for (std::size_t c = 0; c < k; ++c) {
    const std::int32_t cnt = counts(c);
    if (cnt <= 0) {
      continue;
    }
    const T inv = T{1} / static_cast<T>(cnt);
    const T *src = sums.data() + (c * d);
    T *dst = centroids.data() + (c * d);
    for (std::size_t t = 0; t < d; ++t) {
      dst[t] = src[t] * inv;
    }
  }
}

/**
 * @brief Refresh @c cSqNorms against the current centroid matrix.
 */
template <class T>
void refreshCentroidSqNorms(const NDArray<T, 2, Layout::Contig> &centroids,
                            NDArray<T, 1> &cSqNorms) noexcept {
  const std::size_t k = centroids.dim(0);
  const std::size_t d = centroids.dim(1);
  for (std::size_t c = 0; c < k; ++c) {
    const T *row = centroids.data() + (c * d);
    T s = T{0};
    for (std::size_t t = 0; t < d; ++t) {
      s += row[t] * row[t];
    }
    cSqNorms(c) = s;
  }
}

/**
 * @brief Drive the Lloyd iteration to completion given preallocated scratch.
 *
 * Assignment uses @c math::pairwiseArgminSqEuclidean (fused AVX2 path at @c d <= 256). Label
 * accumulation uses @ref scatterAndFoldPlain or @ref scatterAndFoldKahan depending on @c n
 * vs @ref kahanNThreshold. Convergence is the Kahan-summed sum of per-centroid squared
 * shifts compared against @c tol * tol so the caller-facing tolerance reads as a linear shift
 * budget despite the squared-space comparison.
 *
 * @return Pair @c (nIter, converged). @c nIter counts completed iterations; @c converged is
 *         @c true iff the stopping predicate fired before @p maxIter.
 */
template <class T>
std::pair<std::size_t, bool> runLloydFused(const NDArray<T, 2, Layout::Contig> &X,
                                           const LloydScratch<T> &scratch, std::size_t k,
                                           std::size_t maxIter, T tol, math::Pool pool) {
  const std::size_t n = X.dim(0);
  const std::size_t d = X.dim(1);
  (void)d;

  NDArray<T, 2, Layout::Contig> &centroids = *scratch.centroids;
  NDArray<T, 2, Layout::Contig> &centroidsOld = *scratch.centroidsOld;
  NDArray<T, 1> &cSqNorms = *scratch.cSqNorms;
  NDArray<T, 1> &minDistSq = *scratch.minDistSq;
  NDArray<T, 1> &shiftSq = *scratch.shiftSq;

  const T tolSq = tol * tol;
  const bool useKahan = n >= kahanNThreshold;

  std::size_t iter = 0;
  bool converged = false;

  refreshCentroidSqNorms<T>(centroids, cSqNorms);

  while (iter < maxIter) {
    // Assignment: labels(i) = argmin_c ||X(i) - centroids(c)||^2. First call in the iteration
    // -- alloc counter tests snapshot here.
    runAssignment<T>(X, scratch, pool);

    // Save current centroids for the shift check once the mean step lands.
    std::memcpy(centroidsOld.data(), centroids.data(),
                centroids.dim(0) * centroids.dim(1) * sizeof(T));

    if (useKahan) {
      scatterAndFoldKahan<T>(X, scratch, k, pool);
    } else {
      scatterAndFoldPlain<T>(X, scratch, k, pool);
    }

    // Empty-cluster reseed: furthest-point, O(k) passes bounded by counts scan. The donor's
    // minDistSq is zeroed so successive empties cannot reseed to the same point.
    (void)reseedEmptyClusters<T>(X, centroids, *scratch.sums, *scratch.counts, minDistSq);

    finalizeMeans<T>(centroids, *scratch.sums, *scratch.counts);
    refreshCentroidSqNorms<T>(centroids, cSqNorms);

    math::centroidShift<T>(centroidsOld, centroids, shiftSq, pool);
    const T totalShift = totalShiftSqKahan<T>(shiftSq);

    ++iter;
    if (totalShift <= tolSq) {
      converged = true;
      break;
    }
  }

  // Re-assign labels against the final centroids so the exposed assignment matches the
  // centroid matrix callers see. No extra allocation; minDistSq becomes the inertia payload.
  runAssignment<T>(X, scratch, pool);

  return {iter, converged};
}

} // namespace clustering::kmeans::detail
