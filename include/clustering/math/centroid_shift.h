#pragma once

#include <cstddef>
#include <type_traits>

#include "clustering/always_assert.h"
#include "clustering/math/pairwise.h"
#include "clustering/math/thread.h"
#include "clustering/ndarray.h"

namespace clustering::math {

/**
 * @brief Per-row squared shift between two centroid matrices of identical shape.
 *
 * Writes @c outShiftSq(c) = sum_t (cNew(c, t) - cOld(c, t))^2 for every cluster @c c.
 * Composes from @c detail::sqEuclideanRow, which picks an AVX2 reduction on contiguous
 * 32-byte-aligned inputs and falls back to a scalar loop otherwise. The outer row loop fans
 * out over @p pool using the same @c shouldParallelize gate as @ref rowNormsSq.
 *
 * @tparam T Element type (@c float or @c double).
 * @param cOld       Previous centroids (k x d), contiguous.
 * @param cNew       Current centroids (k x d), contiguous.
 * @param outShiftSq Rank-1 output of length k; @c isMutable() must be true.
 * @param pool       Parallelism injection.
 */
template <class T>
void centroidShift(const NDArray<T, 2, Layout::Contig> &cOld,
                   const NDArray<T, 2, Layout::Contig> &cNew, NDArray<T, 1> &outShiftSq,
                   Pool pool) {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                "centroidShift<T> requires T to be float or double");

  CLUSTERING_ALWAYS_ASSERT(outShiftSq.isMutable());
  CLUSTERING_ALWAYS_ASSERT(cOld.dim(0) == cNew.dim(0));
  CLUSTERING_ALWAYS_ASSERT(cOld.dim(1) == cNew.dim(1));
  CLUSTERING_ALWAYS_ASSERT(outShiftSq.dim(0) == cOld.dim(0));

  const std::size_t k = cOld.dim(0);
  if (k == 0) {
    return;
  }

  auto runRowRange = [&](std::size_t lo, std::size_t hi) noexcept {
    for (std::size_t c = lo; c < hi; ++c) {
      outShiftSq(c) = detail::sqEuclideanRow<T, Layout::Contig, Layout::Contig>(cOld, c, cNew, c);
    }
  };

  if (pool.shouldParallelize(k, 4, 2) && pool.pool != nullptr) {
    pool.pool
        ->submit_blocks(std::size_t{0}, k,
                        [&](std::size_t lo, std::size_t hi) { runRowRange(lo, hi); })
        .wait();
  } else {
    runRowRange(0, k);
  }
}

} // namespace clustering::math
