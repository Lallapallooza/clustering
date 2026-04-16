#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <utility>

#include "clustering/always_assert.h"
#include "clustering/kmeans/detail/convergence.h"
#include "clustering/kmeans/detail/dispatch.h"
#include "clustering/kmeans/detail/empty_cluster.h"
#include "clustering/math/centroid_shift.h"
#include "clustering/math/defaults.h"
#include "clustering/math/detail/gemm_outer_prepacked.h"
#include "clustering/math/detail/gemm_pack.h"
#include "clustering/math/detail/matrix_desc.h"
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
  /// Doubles as the Goto-packed centroid buffer the chunked materialized fallback reads out of
  /// (same @c packB layout), so the two assignment paths share one panel storage.
  NDArray<T, 1> *packedB;
  /// Packed centroid-norms scratch paired with @c packedB; fused-argmin-only.
  NDArray<T, 1> *packedCSqNorms;
  /// Chunk distance tile for the materialized assignment fallback (@c chunkRows x k).
  /// Sized via @c math::chunkedMaterializedScratchShape; unused on the fused-argmin path.
  NDArray<T, 2, Layout::Contig> *distsChunk;
  /// Per-worker A-pack arena for @c gemmRunPrepacked (@c blocks * kMc * kKc elements).
  NDArray<T, 1> *gemmApArena;
  /// Per-chunk row-squared-norm buffer used by the chunked broadcast step.
  NDArray<T, 1> *xChunkNormsSq;
};

/**
 * @brief Compute per-row argmin over 256-row chunks reading all scratch from @p scratch.
 *
 * Mirrors @c math::detail::pairwiseArgminMaterializedWithScratch shape-for-shape but replaces
 * the per-chunk @c pairwiseSqEuclidean (which allocates xNorms, yNorms, and GEMM arenas) with
 * an explicit @c packB + @c gemmRunPrepacked sequence over caller-owned scratch. Net effect:
 * a full Lloyd iteration at @c d > @c pairwiseArgminMaxD executes zero @c AlignedAllocator
 * calls, preserving the solver's zero-alloc iteration contract.
 */
template <class T>
void runChunkedMaterializedAssignment(const NDArray<T, 2, Layout::Contig> &X,
                                      const LloydScratch<T> &scratch, math::Pool pool) noexcept {
  const auto &centroids = *scratch.centroids;
  const auto &cSqNorms = *scratch.cSqNorms;
  NDArray<std::int32_t, 1> &labels = *scratch.labels;
  NDArray<T, 1> &minDistSq = *scratch.minDistSq;
  NDArray<T, 2, Layout::Contig> &distsChunk = *scratch.distsChunk;

  const std::size_t n = X.dim(0);
  const std::size_t k = centroids.dim(0);
  const std::size_t d = X.dim(1);
  if (n == 0 || k == 0) {
    return;
  }

  // The single-call @c packB emits one panel per row-strip; @c gemmRunPrepacked then reads that
  // panel assuming @c (jc, pc) tile walks match the packer's layout. That match only holds when
  // @c d <= kKc<T> and @c k <= kNc<T>. Outside the envelope the reader silently picks the wrong
  // centroid columns; guard loudly so a future envelope widening surfaces as an assert rather
  // than a numerical mystery.
  CLUSTERING_ALWAYS_ASSERT(d <= math::detail::kKc<T>);
  CLUSTERING_ALWAYS_ASSERT(k <= math::detail::kNc<T>);

  // Pack the full centroid matrix once per call into the shared packedB buffer. Layout matches
  // the fused path's @c packB output exactly so the two assignment paths can share storage.
  const auto cTransposed = centroids.t();
  const auto cDesc = ::clustering::detail::describeMatrix(cTransposed);
  math::detail::packB<T>(cDesc, 0, d, 0, k, scratch.packedB->data());

  const std::size_t chunkCap = math::pairwiseArgminChunkRows;
  T *apArena = scratch.gemmApArena->data();
  T *xChunkNorms = scratch.xChunkNormsSq->data();

  for (std::size_t iBase = 0; iBase < n; iBase += chunkCap) {
    const std::size_t chunkRows = (iBase + chunkCap <= n) ? chunkCap : (n - iBase);

    // Contiguous sub-view of X: row iBase through iBase + chunkRows - 1. Zero-copy borrow.
    auto xChunk = NDArray<T, 2, Layout::Contig>::borrow(X.data() + (iBase * d), {chunkRows, d});

    // Per-row squared norm for the broadcast epilogue. Kept in a caller-owned scratch array so
    // the chunk loop does not allocate.
    for (std::size_t r = 0; r < chunkRows; ++r) {
      xChunkNorms[r] = math::detail::sqNormRow<T, Layout::Contig>(xChunk, r);
    }

    // distsChunk = -2 * xChunk * centroids.t(). Using the pre-packed centroids buffer and
    // caller-owned A-pack arena keeps the GEMM alloc-free.
    auto distsView = NDArray<T, 2>::borrow(distsChunk.data(), {chunkRows, k});
    const auto xDesc = ::clustering::detail::describeMatrix(xChunk);
    auto distsDesc = ::clustering::detail::describeMatrixMut(distsView);
    math::detail::gemmRunPrepacked<T>(xDesc, scratch.packedB->data(), d, k, distsDesc, T{-2}, T{0},
                                      apArena, pool);

    auto scanRange = [&](std::size_t lo, std::size_t hi) noexcept {
      for (std::size_t i = lo; i < hi; ++i) {
        const T xn = xChunkNorms[i];
        const T *row = distsChunk.data() + (i * k);
        T bestVal = std::numeric_limits<T>::infinity();
        std::int32_t bestIdx = 0;
        for (std::size_t j = 0; j < k; ++j) {
          // Reconstruct ||x||^2 + ||c||^2 - 2 x . c, clamping cancellation artefacts to zero
          // the same way @c pairwiseSqEuclideanGemm's broadcast pass does.
          T v = row[j] + xn + cSqNorms(j);
          if (v < T{0}) {
            v = T{0};
          }
          if (j == 0 || v < bestVal) {
            bestVal = v;
            bestIdx = static_cast<std::int32_t>(j);
          }
        }
        minDistSq(iBase + i) = bestVal;
        labels(iBase + i) = bestIdx;
      }
    };

    if (pool.shouldParallelize(chunkRows, 4, 2) && pool.pool != nullptr) {
      pool.pool
          ->submit_blocks(std::size_t{0}, chunkRows,
                          [&](std::size_t lo, std::size_t hi) { scanRange(lo, hi); })
          .wait();
    } else {
      scanRange(0, chunkRows);
    }
  }
}

/**
 * @brief Maximum @c d for the direct-compute argmin hot path.
 *
 * At @c d <= this threshold the fused argmin-GEMM driver's @c packA + packB overhead dominates
 * the handful of FMAs the microkernel performs, so the direct @c ||x - c||^2 formula with 8-row
 * SIMD accumulators beats the packed-GEMM path. Measured on Zen5: crossover sits near
 * @c d == 8 where the two paths tie; below that the direct path wins by the pack cost.
 */
inline constexpr std::size_t kDirectArgminMaxD = 8;

/**
 * @brief Run the assignment step against caller-owned scratch, splitting on @c d.
 *
 * Three dispatch tiers:
 *   - @c d <= @ref kDirectArgminMaxD : direct small-@c d argmin (no GEMM packing).
 *   - @c d <= @c pairwiseArgminMaxD  : fused argmin-GEMM with pre-packed scratch.
 *   - otherwise                      : chunked materialized fallback.
 */
template <class T>
void runAssignment(const NDArray<T, 2, Layout::Contig> &X, const LloydScratch<T> &scratch,
                   math::Pool pool) {
#ifdef CLUSTERING_USE_AVX2
  if constexpr (std::is_same_v<T, float>) {
    const auto &centroids = *scratch.centroids;
    const auto &cSqNorms = *scratch.cSqNorms;
    const std::size_t d = X.dim(1);
    if (X.template isAligned<32>() && centroids.template isAligned<32>() && d != 0) {
      if (d <= kDirectArgminMaxD) {
        math::detail::pairwiseArgminDirectSmallDF32(X, centroids, *scratch.labels,
                                                    *scratch.minDistSq, pool);
        return;
      }
      if (d <= math::defaults::pairwiseArgminMaxD) {
        math::detail::pairwiseArgminOuterAvx2F32WithScratch(
            X, centroids, cSqNorms, *scratch.labels, *scratch.minDistSq, scratch.packedB->data(),
            scratch.packedCSqNorms->data(), pool);
        return;
      }
    }
  }
#endif
  runChunkedMaterializedAssignment<T>(X, scratch, pool);
}

/**
 * @brief Whether @ref runAssignment's direct small-@c d path will fire for this shape.
 *
 * Returns @c true iff the direct path handles @p d (no GEMM decomposition, so @c minDistSq
 * carries the true squared distance and downstream callers can skip the final
 * @ref recomputeMinDistSqDirect pass).
 */
template <class T>
[[nodiscard]] bool
assignmentProducesDirectMinDistSq(const NDArray<T, 2, Layout::Contig> &X,
                                  const NDArray<T, 2, Layout::Contig> &C) noexcept {
#ifdef CLUSTERING_USE_AVX2
  if constexpr (std::is_same_v<T, float>) {
    const std::size_t d = X.dim(1);
    return X.template isAligned<32>() && C.template isAligned<32>() && d != 0 &&
           d <= kDirectArgminMaxD;
  } else {
    (void)X;
    (void)C;
    return false;
  }
#else
  (void)X;
  (void)C;
  return false;
#endif
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
 * @brief Overwrite @p minDistSq(i) with the directly-computed @c ||X(i) - centroids[labels(i)]||^2.
 *
 * The fused argmin path tracks @c ||c||^2 - 2 x.c then adds @c ||x||^2 at the tile epilogue. At
 * large centroid magnitudes the f32 sum @c ||c||^2 + ||x||^2 - 2 x.c suffers catastrophic
 * cancellation (the two large terms partially cancel, the residual is dominated by rounding
 * noise). Argmin labels are unaffected because @c ||x||^2 is constant per row -- the same
 * additive shift applies to every candidate centroid. Inertia and the empty-cluster reseed
 * argmax both read the absolute distances and need the direct subtract-square-sum to be
 * numerically faithful at large coordinate magnitudes.
 */
template <class T>
void recomputeMinDistSqDirect(const NDArray<T, 2, Layout::Contig> &X,
                              const NDArray<T, 2, Layout::Contig> &centroids,
                              const NDArray<std::int32_t, 1> &labels, NDArray<T, 1> &minDistSq,
                              math::Pool pool) noexcept {
  const std::size_t n = X.dim(0);
  const std::size_t d = X.dim(1);
  const std::size_t k = centroids.dim(0);
  if (n == 0 || d == 0 || k == 0) {
    return;
  }

  auto runRowRange = [&](std::size_t lo, std::size_t hi) noexcept {
    for (std::size_t i = lo; i < hi; ++i) {
      const std::int32_t lbl = labels(i);
      if (lbl < 0 || std::cmp_greater_equal(lbl, k)) {
        minDistSq(i) = T{0};
        continue;
      }
      const T *xRow = X.data() + (i * d);
      const T *cRow = centroids.data() + (static_cast<std::size_t>(lbl) * d);
      T sum = T{0};
      for (std::size_t t = 0; t < d; ++t) {
        const T diff = xRow[t] - cRow[t];
        sum += diff * diff;
      }
      minDistSq(i) = sum;
    }
  };

  if (pool.shouldParallelize(n, 64, 2) && pool.pool != nullptr) {
    pool.pool
        ->submit_blocks(std::size_t{0}, n,
                        [&](std::size_t lo, std::size_t hi) { runRowRange(lo, hi); })
        .wait();
  } else {
    runRowRange(0, n);
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

    // Empty-cluster reseed: furthest-point, O(k) passes bounded by counts scan. minDistSq still
    // holds the decomposed-formula residual so it carries cancellation noise at large centroid
    // magnitudes; the noise tail is bounded by per-point @c ||c||^2 + ||x||^2 cancellation, much
    // smaller than the inter-blob distance the donor is selected against, so the argmax
    // selection is preserved in practice on benchmark data. The donor's minDistSq is zeroed so
    // successive empties cannot reseed to the same point.
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
  // centroid matrix callers see. The direct small-@c d path already writes true squared
  // distances into @c minDistSq, so the recompute-direct pass is only needed when the
  // decomposed @c ||c||^2 - 2 x.c + ||x||^2 formula could carry cancellation noise.
  runAssignment<T>(X, scratch, pool);
  if (!assignmentProducesDirectMinDistSq<T>(X, centroids)) {
    recomputeMinDistSqDirect<T>(X, centroids, *scratch.labels, minDistSq, pool);
  }

  return {iter, converged};
}

} // namespace clustering::kmeans::detail
