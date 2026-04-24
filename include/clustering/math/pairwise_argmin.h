#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "clustering/always_assert.h"
#include "clustering/math/defaults.h"
#include "clustering/math/detail/pairwise_argmin_outer.h"
#include "clustering/math/pairwise.h"
#include "clustering/math/thread.h"
#include "clustering/ndarray.h"

namespace clustering::math {

/**
 * @brief Chunk height used by the materialized argmin path when striping over @c n.
 *
 * Matches scikit-learn's @c CHUNK_SIZE = 256 for euclidean-distance assignment so the per-chunk
 * distance tile (@c chunkRows * k * sizeof(T)) stays L2-resident on targets with >=256 KiB L2.
 */
inline constexpr std::size_t pairwiseArgminChunkRows = 256;

/**
 * @brief Required shape for the chunked materialized argmin scratch buffer.
 *
 * The caller sizes an owning `NDArray<T, 2>` with these dimensions and forwards it as
 * @c distsScratch to @c pairwiseArgminMaterializedWithScratch. The height collapses to
 * @c n when @c n is smaller than the chunk row cap so small inputs do not overallocate.
 *
 * @param n Row count of the data matrix.
 * @param k Row count of the centroid matrix.
 */
[[nodiscard]] inline std::array<std::size_t, 2>
chunkedMaterializedScratchShape(std::size_t n, std::size_t k) noexcept {
  const std::size_t rows = (n < pairwiseArgminChunkRows) ? n : pairwiseArgminChunkRows;
  return {rows == 0 ? std::size_t{1} : rows, k == 0 ? std::size_t{1} : k};
}

namespace detail {

/**
 * @brief Tag identifying which outer driver executed for a @ref pairwiseArgminSqEuclidean
 *        request.
 *
 * Test surface only. Mirrors @ref PairwisePath so dispatch tests can pin the branch
 * crisply without resorting to timing or numerical fingerprinting.
 */
enum class ArgminPath : std::uint8_t { Fused, Materialized };

/**
 * @brief Compute per-row argmin + minimum squared distance over @p n in 256-row strips using a
 *        caller-owned distance tile.
 *
 * For each chunk of up to @c pairwiseArgminChunkRows rows of @p X, calls
 * @c pairwiseSqEuclidean against the full centroid matrix to fill the @p distsScratch tile,
 * then scalar-argmins each row into @p labels + @p outMinSq. The tile is sized so one full
 * chunk's worth of distances stays L2-resident across the GEMM and the argmin scan; keeping
 * the output matrix off DRAM is the load-bearing win over materializing `(n, k)` up front.
 *
 * Chunks are independent so the outer chunk loop parallelises freely; per-row argmin carries
 * no reduction across rows, so threaded and serial runs produce bit-identical labels and
 * within-reassociation-tolerance @c outMinSq.
 *
 * @tparam T  Element type (@c float or @c double).
 * @tparam LX Layout tag of @p X.
 * @tparam LC Layout tag of @p C.
 * @param X             Data matrix (n x d).
 * @param C             Centroid matrix (k x d).
 * @param labels        Output labels of length n; must be mutable.
 * @param outMinSq      Output minimum squared distances of length n; must be mutable.
 * @param distsScratch  Distance tile of shape @ref chunkedMaterializedScratchShape; the
 *                      function reuses it across chunks.
 * @param pool          Parallelism injection; forwarded to the per-chunk @c pairwiseSqEuclidean.
 */
template <class T, Layout LX, Layout LC>
void pairwiseArgminMaterializedWithScratch(const NDArray<T, 2, LX> &X, const NDArray<T, 2, LC> &C,
                                           NDArray<std::int32_t, 1> &labels,
                                           NDArray<T, 1> &outMinSq, NDArray<T, 2> &distsScratch,
                                           Pool pool) {
  const std::size_t n = X.dim(0);
  const std::size_t k = C.dim(0);
  const std::size_t d = X.dim(1);
  if (n == 0 || k == 0) {
    return;
  }
  const std::size_t chunkCap = pairwiseArgminChunkRows;
  CLUSTERING_ALWAYS_ASSERT(distsScratch.dim(1) >= k);
  CLUSTERING_ALWAYS_ASSERT(distsScratch.dim(0) >= (n < chunkCap ? n : chunkCap));

  auto runChunk = [&](std::size_t iBase, std::size_t chunkRows, const auto &xChunk) noexcept {
    // The scratch tile may be wider than k when the caller pre-sized with k padded up; the
    // dispatch view narrows it back to (chunkRows, k) so the callee sees exactly one chunk.
    NDArray<T, 2> distsView = NDArray<T, 2>::borrow(distsScratch.data(), {chunkRows, k});
    pairwiseSqEuclidean(xChunk, C, distsView, pool);

    auto scanRange = [&](std::size_t lo, std::size_t hi) noexcept {
      for (std::size_t i = lo; i < hi; ++i) {
        const T *row = distsView.data() + (i * k);
        T bestVal = row[0];
        std::int32_t bestIdx = 0;
        for (std::size_t j = 1; j < k; ++j) {
          const T v = row[j];
          if (v < bestVal) {
            bestVal = v;
            bestIdx = static_cast<std::int32_t>(j);
          }
        }
        outMinSq(iBase + i) = bestVal;
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
  };

  for (std::size_t iBase = 0; iBase < n; iBase += chunkCap) {
    const std::size_t chunkRows = (iBase + chunkCap <= n) ? chunkCap : (n - iBase);
    if constexpr (LX == Layout::Contig) {
      auto xChunk = NDArray<T, 2, Layout::Contig>::borrow(X.data() + (iBase * d), {chunkRows, d});
      runChunk(iBase, chunkRows, xChunk);
    } else {
      auto xChunk = X.slice(0, iBase, iBase + chunkRows);
      runChunk(iBase, chunkRows, xChunk);
    }
  }
}

/**
 * @brief Compute per-row argmin and minimum squared distance via the materialized two-step.
 *
 * Convenience wrapper that allocates the chunked distance scratch and forwards to
 * @ref pairwiseArgminMaterializedWithScratch. Used as the fall-through path when @c d exceeds
 * @c defaults::pairwiseArgminMaxD and as the correctness oracle for the fused path in tests.
 */
template <class T, Layout LX, Layout LC>
void pairwiseArgminMaterialized(const NDArray<T, 2, LX> &X, const NDArray<T, 2, LC> &C,
                                NDArray<std::int32_t, 1> &labels, NDArray<T, 1> &outMinSq,
                                Pool pool) {
  const std::size_t n = X.dim(0);
  const std::size_t k = C.dim(0);
  if (n == 0 || k == 0) {
    return;
  }

  const auto shape = chunkedMaterializedScratchShape(n, k);
  NDArray<T, 2> distsScratch({shape[0], shape[1]});
  pairwiseArgminMaterializedWithScratch(X, C, labels, outMinSq, distsScratch, pool);
}

/**
 * @brief Runtime predicate: true when the fused AVX2 path is eligible for this call.
 *
 * Dispatch criteria (all must hold):
 *   - @c CLUSTERING_USE_AVX2 is defined and @c T is @c float.
 *   - @p X and @p C are @c Layout::Contig.
 *   - @p X and @p C are runtime 32-byte aligned (the AVX2 microkernel issues aligned loads).
 *   - @p d is non-zero and at most @c defaults::pairwiseArgminMaxD.
 *   - @p n and @p k are non-zero.
 *
 * Any failure falls through to the materialized path.
 */
template <class T, Layout LX, Layout LC>
bool canUseFusedArgmin(const NDArray<T, 2, LX> &X, const NDArray<T, 2, LC> &C,
                       const NDArray<T, 1> &cSqNorms) noexcept {
#ifdef CLUSTERING_USE_AVX2
  if constexpr (std::is_same_v<T, float> && LX == Layout::Contig && LC == Layout::Contig) {
    const std::size_t n = X.dim(0);
    const std::size_t k = C.dim(0);
    const std::size_t d = X.dim(1);
    if (n == 0 || k == 0 || d == 0) {
      return false;
    }
    if (d > defaults::pairwiseArgminMaxD) {
      return false;
    }
    if (!X.template isAligned<32>() || !C.template isAligned<32>()) {
      return false;
    }
    if (!cSqNorms.isContiguous()) {
      return false;
    }
    return true;
  } else {
    (void)X;
    (void)C;
    (void)cSqNorms;
    return false;
  }
#else
  (void)X;
  (void)C;
  (void)cSqNorms;
  return false;
#endif
}

} // namespace detail

/**
 * @brief Per-row argmin and minimum squared distance of rows of @p X against rows of @p C.
 *
 * Writes `labels(i)` = argmin_j ||X(i) - C(j)||^2 and
 * `outMinDistSq(i)` = min_j ||X(i) - C(j)||^2 for every row of @p X. @p cSqNorms must hold
 * `||C(j)||^2` per row of @p C; callers typically produce it via @c rowNormsSq.
 *
 * Dispatches between a fused AVX2 outer driver (@c d <= @c defaults::pairwiseArgminMaxD,
 * float, contiguous, 32-byte aligned X and C) and a materialized two-step
 * (@c pairwiseSqEuclidean + per-row scan) otherwise. Both paths produce labels consistent
 * within float-reassociation tolerance; strict-less-than tie-break mirrors @c math::argmin.
 *
 * @tparam T  Element type (@c float or @c double).
 * @tparam LX Layout tag of @p X; CTAD-resolved.
 * @tparam LC Layout tag of @p C; CTAD-resolved.
 * @param X            Data matrix (n x d).
 * @param C            Centroid matrix (k x d); must have the same column count as @p X.
 * @param cSqNorms     Row-1 array of squared norms, length @c k.
 * @param labels       Mutable row-1 output of length @c n, holds the argmin per row.
 * @param outMinDistSq Mutable row-1 output of length @c n, holds the minimum distance per row.
 * @param pool         Parallelism injection; forwarded to the selected path.
 */
template <class T, Layout LX = Layout::Contig, Layout LC = Layout::Contig>
void pairwiseArgminSqEuclidean(const NDArray<T, 2, LX> &X, const NDArray<T, 2, LC> &C,
                               const NDArray<T, 1> &cSqNorms, NDArray<std::int32_t, 1> &labels,
                               NDArray<T, 1> &outMinDistSq, Pool pool) {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                "pairwiseArgminSqEuclidean<T> requires T to be float or double");

  CLUSTERING_ALWAYS_ASSERT(labels.isMutable());
  CLUSTERING_ALWAYS_ASSERT(outMinDistSq.isMutable());
  CLUSTERING_ALWAYS_ASSERT(X.dim(1) == C.dim(1));
  CLUSTERING_ALWAYS_ASSERT(labels.dim(0) == X.dim(0));
  CLUSTERING_ALWAYS_ASSERT(outMinDistSq.dim(0) == X.dim(0));
  CLUSTERING_ALWAYS_ASSERT(cSqNorms.dim(0) == C.dim(0));

  const std::size_t n = X.dim(0);
  const std::size_t k = C.dim(0);
  if (n == 0 || k == 0) {
    return;
  }

#ifdef CLUSTERING_USE_AVX2
  if constexpr (std::is_same_v<T, float> && LX == Layout::Contig && LC == Layout::Contig) {
    if (detail::canUseFusedArgmin(X, C, cSqNorms)) {
      detail::pairwiseArgminOuterAvx2F32(X, C, cSqNorms, labels, outMinDistSq, pool);
      return;
    }
  }
#endif

  detail::pairwiseArgminMaterialized(X, C, labels, outMinDistSq, pool);
}

namespace detail {

/**
 * @brief Test-only: runs the same dispatch as @ref pairwiseArgminSqEuclidean and reports
 *        which outer driver fired.
 *
 * Shares the public entry's preconditions and branch; the return on empty input is
 * @c ArgminPath::Materialized by convention (neither driver runs; the cheap path at
 * `(n, k)` == (0, 0) short-circuits before dispatch).
 */
template <class T, Layout LX = Layout::Contig, Layout LC = Layout::Contig>
ArgminPath pairwiseArgminSqEuclideanWithDispatchInfo(const NDArray<T, 2, LX> &X,
                                                     const NDArray<T, 2, LC> &C,
                                                     const NDArray<T, 1> &cSqNorms,
                                                     NDArray<std::int32_t, 1> &labels,
                                                     NDArray<T, 1> &outMinDistSq, Pool pool) {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                "pairwiseArgminSqEuclideanWithDispatchInfo<T> requires T to be float or double");

  CLUSTERING_ALWAYS_ASSERT(labels.isMutable());
  CLUSTERING_ALWAYS_ASSERT(outMinDistSq.isMutable());
  CLUSTERING_ALWAYS_ASSERT(X.dim(1) == C.dim(1));
  CLUSTERING_ALWAYS_ASSERT(labels.dim(0) == X.dim(0));
  CLUSTERING_ALWAYS_ASSERT(outMinDistSq.dim(0) == X.dim(0));
  CLUSTERING_ALWAYS_ASSERT(cSqNorms.dim(0) == C.dim(0));

  const std::size_t n = X.dim(0);
  const std::size_t k = C.dim(0);
  if (n == 0 || k == 0) {
    return ArgminPath::Materialized;
  }

#ifdef CLUSTERING_USE_AVX2
  if constexpr (std::is_same_v<T, float> && LX == Layout::Contig && LC == Layout::Contig) {
    if (canUseFusedArgmin(X, C, cSqNorms)) {
      pairwiseArgminOuterAvx2F32(X, C, cSqNorms, labels, outMinDistSq, pool);
      return ArgminPath::Fused;
    }
  }
#endif

  pairwiseArgminMaterialized(X, C, labels, outMinDistSq, pool);
  return ArgminPath::Materialized;
}

} // namespace detail

} // namespace clustering::math
