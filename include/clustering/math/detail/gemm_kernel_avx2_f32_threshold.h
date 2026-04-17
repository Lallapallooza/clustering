#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#ifdef CLUSTERING_USE_AVX2
#include <immintrin.h>
#endif

#include "clustering/math/detail/gemm_kernel_scalar.h"

namespace clustering::math::detail {

#ifdef CLUSTERING_USE_AVX2

/**
 * @brief Fused 8x6 AVX2 f32 microkernel: accumulate @c X*Y^T over the full K, fold the
 *        @c ||x||^2 + ||y||^2 norms to recover squared distance, and emit every cell whose
 *        squared distance lies at or below @p radiusSq.
 *
 * Computes a column-by-column inner product of an 8-row A-panel against a 6-column B-panel.
 * After the @c K-loop, converts each accumulator to @c ||x_i||^2 + ||y_j||^2 - 2*x_i.y_j,
 * clamps the tiny-negative cancellation region up to zero, and compares against a broadcast
 * @c radiusSq.
 *
 * Buffer layouts match @c packA / @c packB from @c gemm_pack.h:
 *   - @p apPanel holds an @c Mr x @c kc A-panel; element @c (r, k) is at @c ap[k*Mr + r].
 *   - @p bpPanel holds a @c kc x @c Nr B-panel; element @c (k, c) is at @c bp[k*Nr + c].
 *
 * Output ordering contract: within a single tile call, @p emit fires in row-major order --
 * @c (rowBase+0, colBase+0), @c (rowBase+0, colBase+1), ... through @c (rowBase+validRows-1,
 * colBase+validCols-1). Rows past @p validRows and columns past @p validCols are never handed
 * to @p emit. The outer driver iterates tiles in its own natural order (row-chunks first, then
 * column-panels within a chunk), so the composite emit order is deterministic but not
 * globally (row, col)-sorted.
 *
 * @param apPanel    Packed A panel (32-byte aligned).
 * @param bpPanel    Packed B panel (32-byte aligned).
 * @param kc         Inner dimension count.
 * @param aRowNormsSq Per-row squared norms for the Mr rows of this tile, 32-byte aligned;
 *                   padded rows past @p validRows are unread.
 * @param bColNormsSq Per-column squared norms for the Nr columns of this panel; padded
 *                   columns past @p validCols are unread.
 * @param rowBase    Global row index of row 0 of this tile; contributes to @p emit via
 *                   @c rowBase + rowOffset.
 * @param colBase    Global column index of column 0 of this panel; contributes to @p emit
 *                   via @c colBase + columnOffset.
 * @param validRows  Count of valid rows in the tile; @c [0, validRows) rows are scanned.
 * @param validCols  Count of valid columns in the panel; @c [0, validCols) columns are
 *                   scanned.
 * @param radiusSq   Squared radius; cells whose squared distance is at most this value
 *                   trigger @p emit.
 * @param emit       @c void(std::size_t, std::size_t) callable; invoked once per surviving
 *                   cell in row-major order.
 */
template <class Emit>
inline void gemmKernel8x6Avx2F32Threshold(const float *apPanel, const float *bpPanel,
                                          std::size_t kc, const float *aRowNormsSq,
                                          const float *bColNormsSq, std::size_t rowBase,
                                          std::size_t colBase, std::size_t validRows,
                                          std::size_t validCols, float radiusSq, Emit &&emit) {
  constexpr std::size_t kMr = kKernelMr<float>;
  constexpr std::size_t kNr = kKernelNr<float>;

  // __restrict__ locals captured at the kernel top per the codegen-discipline convention: the
  // compiler must prove non-aliasing across the K-loop before it can pin the six accumulators
  // in YMM registers and fully unroll the inner body.
  const float *__restrict__ apLocal = apPanel;
  const float *__restrict__ bpLocal = bpPanel;
  const std::size_t kcLocal = kc;

  __m256 c0 = _mm256_setzero_ps();
  __m256 c1 = _mm256_setzero_ps();
  __m256 c2 = _mm256_setzero_ps();
  __m256 c3 = _mm256_setzero_ps();
  __m256 c4 = _mm256_setzero_ps();
  __m256 c5 = _mm256_setzero_ps();

  for (std::size_t k = 0; k < kcLocal; ++k) {
    const __m256 a = _mm256_load_ps(apLocal + (k * kMr));
    const float *bRow = bpLocal + (k * kNr);
    c0 = _mm256_fmadd_ps(a, _mm256_broadcast_ss(bRow + 0), c0);
    c1 = _mm256_fmadd_ps(a, _mm256_broadcast_ss(bRow + 1), c1);
    c2 = _mm256_fmadd_ps(a, _mm256_broadcast_ss(bRow + 2), c2);
    c3 = _mm256_fmadd_ps(a, _mm256_broadcast_ss(bRow + 3), c3);
    c4 = _mm256_fmadd_ps(a, _mm256_broadcast_ss(bRow + 4), c4);
    c5 = _mm256_fmadd_ps(a, _mm256_broadcast_ss(bRow + 5), c5);
  }

  // Fold @c dot(x_i, y_j) into @c ||x_i||^2 + ||y_j||^2 - 2*dot per column. Cancellation when
  // @c x_i ~= y_j can drop the result fractionally below zero; clamping at zero keeps the
  // contract that squared distances are non-negative, matching @ref pairwiseSqEuclideanGemm.
  const __m256 xNorms = _mm256_load_ps(aRowNormsSq);
  const __m256 neg2 = _mm256_set1_ps(-2.0F);
  const __m256 zero = _mm256_setzero_ps();

  alignas(32) std::array<std::array<float, kMr>, kNr> distCols{};
  auto fold = [&](__m256 acc, std::size_t colOffset, float *out) noexcept {
    const __m256 yNorm = _mm256_set1_ps(bColNormsSq[colOffset]);
    __m256 dist = _mm256_fmadd_ps(acc, neg2, _mm256_add_ps(xNorms, yNorm));
    dist = _mm256_max_ps(dist, zero);
    _mm256_store_ps(out, dist);
  };
  if (validCols > 0) {
    fold(c0, 0, distCols[0].data());
  }
  if (validCols > 1) {
    fold(c1, 1, distCols[1].data());
  }
  if (validCols > 2) {
    fold(c2, 2, distCols[2].data());
  }
  if (validCols > 3) {
    fold(c3, 3, distCols[3].data());
  }
  if (validCols > 4) {
    fold(c4, 4, distCols[4].data());
  }
  if (validCols > 5) {
    fold(c5, 5, distCols[5].data());
  }

  for (std::size_t r = 0; r < validRows; ++r) {
    for (std::size_t c = 0; c < validCols; ++c) {
      if (distCols[c][r] <= radiusSq) {
        emit(rowBase + r, colBase + c);
      }
    }
  }
}

#endif // CLUSTERING_USE_AVX2

} // namespace clustering::math::detail
