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
 *        `||x||^2 + ||y||^2` norms to recover squared distance, and emit every cell whose
 *        squared distance lies at or below @p radiusSq.
 *
 * Computes a column-by-column inner product of an 8-row A-panel against a 6-column B-panel.
 * After the @c K-loop, converts each accumulator to `||x_i||^2 + ||y_j||^2` - 2*x_i.y_j,
 * clamps the tiny-negative cancellation region up to zero, and compares against a broadcast
 * @c radiusSq.
 *
 * Buffer layouts match @c packA / @c packB from @c gemm_pack.h:
 *   - @p apPanel holds an @c Mr x @c kc A-panel; element `(r, k)` is at `ap[k*Mr + r]`.
 *   - @p bpPanel holds a @c kc x @c Nr B-panel; element `(k, c)` is at `bp[k*Nr + c]`.
 *
 * Output ordering contract: within a single tile call, @p emit fires in row-major order --
 * `(rowBase+0, colBase+0)`, `(rowBase+0, colBase+1)`, ... through `(rowBase+validRows-1,`
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
 * @param validRows  Count of valid rows in the tile; `[0, validRows)` rows are scanned.
 * @param validCols  Count of valid columns in the panel; `[0, validCols)` columns are
 *                   scanned.
 * @param radiusSq   Squared radius; cells whose squared distance is at most this value
 *                   trigger @p emit.
 * @param emit       `void(std::size_t, std::size_t)` callable; invoked once per surviving
 *                   cell in row-major order.
 */
template <class Emit>
[[gnu::always_inline]] inline void
gemmKernel8x6Avx2F32Threshold(const float *apPanel, const float *bpPanel, std::size_t kc,
                              const float *aRowNormsSq, const float *bColNormsSq,
                              std::size_t rowBase, std::size_t colBase, std::size_t validRows,
                              std::size_t validCols, float radiusSq, Emit &&emit) {
  constexpr std::size_t kMr = kKernelMr<float>;
  constexpr std::size_t kNr = kKernelNr<float>;

  // __restrict__ locals captured at the kernel top per the codegen-discipline convention: the
  // compiler must prove non-aliasing across the K-loop before it can pin the six accumulators
  // in YMM registers and fully unroll the inner body.
  const float *__restrict__ apLocal = apPanel;
  const float *__restrict__ bpLocal = bpPanel;
  const std::size_t kcLocal = kc;

  // Split each output column into two accumulators so even / odd K iterations walk disjoint
  // dependency chains. Zen's 4-cycle FMA latency otherwise caps a single-accumulator loop at
  // 0.25 FMAs/cycle per output -- the doubled chains lift that to the 2 FMA/cycle port ceiling.
  __m256 c0a = _mm256_setzero_ps();
  __m256 c0b = _mm256_setzero_ps();
  __m256 c1a = _mm256_setzero_ps();
  __m256 c1b = _mm256_setzero_ps();
  __m256 c2a = _mm256_setzero_ps();
  __m256 c2b = _mm256_setzero_ps();
  __m256 c3a = _mm256_setzero_ps();
  __m256 c3b = _mm256_setzero_ps();
  __m256 c4a = _mm256_setzero_ps();
  __m256 c4b = _mm256_setzero_ps();
  __m256 c5a = _mm256_setzero_ps();
  __m256 c5b = _mm256_setzero_ps();

  std::size_t k = 0;
  for (; k + 1 < kcLocal; k += 2) {
    const __m256 a0 = _mm256_load_ps(apLocal + (k * kMr));
    const __m256 a1 = _mm256_load_ps(apLocal + ((k + 1) * kMr));
    const float *bRow0 = bpLocal + (k * kNr);
    const float *bRow1 = bpLocal + ((k + 1) * kNr);
    c0a = _mm256_fmadd_ps(a0, _mm256_broadcast_ss(bRow0 + 0), c0a);
    c0b = _mm256_fmadd_ps(a1, _mm256_broadcast_ss(bRow1 + 0), c0b);
    c1a = _mm256_fmadd_ps(a0, _mm256_broadcast_ss(bRow0 + 1), c1a);
    c1b = _mm256_fmadd_ps(a1, _mm256_broadcast_ss(bRow1 + 1), c1b);
    c2a = _mm256_fmadd_ps(a0, _mm256_broadcast_ss(bRow0 + 2), c2a);
    c2b = _mm256_fmadd_ps(a1, _mm256_broadcast_ss(bRow1 + 2), c2b);
    c3a = _mm256_fmadd_ps(a0, _mm256_broadcast_ss(bRow0 + 3), c3a);
    c3b = _mm256_fmadd_ps(a1, _mm256_broadcast_ss(bRow1 + 3), c3b);
    c4a = _mm256_fmadd_ps(a0, _mm256_broadcast_ss(bRow0 + 4), c4a);
    c4b = _mm256_fmadd_ps(a1, _mm256_broadcast_ss(bRow1 + 4), c4b);
    c5a = _mm256_fmadd_ps(a0, _mm256_broadcast_ss(bRow0 + 5), c5a);
    c5b = _mm256_fmadd_ps(a1, _mm256_broadcast_ss(bRow1 + 5), c5b);
  }
  if (k < kcLocal) {
    const __m256 a = _mm256_load_ps(apLocal + (k * kMr));
    const float *bRow = bpLocal + (k * kNr);
    c0a = _mm256_fmadd_ps(a, _mm256_broadcast_ss(bRow + 0), c0a);
    c1a = _mm256_fmadd_ps(a, _mm256_broadcast_ss(bRow + 1), c1a);
    c2a = _mm256_fmadd_ps(a, _mm256_broadcast_ss(bRow + 2), c2a);
    c3a = _mm256_fmadd_ps(a, _mm256_broadcast_ss(bRow + 3), c3a);
    c4a = _mm256_fmadd_ps(a, _mm256_broadcast_ss(bRow + 4), c4a);
    c5a = _mm256_fmadd_ps(a, _mm256_broadcast_ss(bRow + 5), c5a);
  }
  const __m256 c0 = _mm256_add_ps(c0a, c0b);
  const __m256 c1 = _mm256_add_ps(c1a, c1b);
  const __m256 c2 = _mm256_add_ps(c2a, c2b);
  const __m256 c3 = _mm256_add_ps(c3a, c3b);
  const __m256 c4 = _mm256_add_ps(c4a, c4b);
  const __m256 c5 = _mm256_add_ps(c5a, c5b);

  // Fold `dot(x_i, y_j)` into `||x_i||^2 + ||y_j||^2` - 2*dot per column, threshold-compare
  // the 8 lanes in SIMD, then pack the result into an 8-bit column mask. Walking the mask is
  // O(survivors) instead of the old scalar O(Mr * Nr) double loop. Cancellation when
  // @c x_i ~= y_j can drop the dist fractionally below zero; the `max(.,0)` clamp preserves
  // the non-negative squared-distance contract and never flips the sign of the comparison.
  const __m256 xNorms = _mm256_load_ps(aRowNormsSq);
  const __m256 neg2 = _mm256_set1_ps(-2.0F);
  const __m256 zero = _mm256_setzero_ps();
  const __m256 radiusVec = _mm256_set1_ps(radiusSq);

  auto foldAndMask = [&](__m256 acc, std::size_t colOffset) noexcept -> std::uint8_t {
    const __m256 yNorm = _mm256_set1_ps(bColNormsSq[colOffset]);
    __m256 dist = _mm256_fmadd_ps(acc, neg2, _mm256_add_ps(xNorms, yNorm));
    dist = _mm256_max_ps(dist, zero);
    const __m256 survive = _mm256_cmp_ps(dist, radiusVec, _CMP_LE_OQ);
    return static_cast<std::uint8_t>(_mm256_movemask_ps(survive));
  };

  std::array<std::uint8_t, kNr> masks{};
  if (validCols > 0) {
    masks[0] = foldAndMask(c0, 0);
  }
  if (validCols > 1) {
    masks[1] = foldAndMask(c1, 1);
  }
  if (validCols > 2) {
    masks[2] = foldAndMask(c2, 2);
  }
  if (validCols > 3) {
    masks[3] = foldAndMask(c3, 3);
  }
  if (validCols > 4) {
    masks[4] = foldAndMask(c4, 4);
  }
  if (validCols > 5) {
    masks[5] = foldAndMask(c5, 5);
  }

  // Clamp mask bits beyond validRows so padding rows in the last M-tile never surface.
  const auto rowMask =
      (validRows >= kMr) ? std::uint8_t{0xFF} : static_cast<std::uint8_t>((1U << validRows) - 1U);
  for (std::size_t c = 0; c < validCols; ++c) {
    auto m = static_cast<std::uint8_t>(masks[c] & rowMask);
    while (m != 0) {
      const int bit = __builtin_ctz(static_cast<unsigned>(m));
      emit(rowBase + static_cast<std::size_t>(bit), colBase + c);
      m = static_cast<std::uint8_t>(m & (m - 1));
    }
  }
}

/**
 * @brief Fused 16x6 AVX2 f32 threshold microkernel over a pair of stacked 8-row A panels.
 *
 * Same contract as @ref gemmKernel8x6Avx2F32Threshold but sixteen output rows per tile, fed by
 * two consecutive @c packA panels. The per-pair kernel is bounded by the two load ports: a
 * broadcast per B column plus one A load per accumulator row-half. Doubling the row count
 * amortises the six broadcasts across two row vectors, so each K step issues eight loads for
 * twelve multiply-adds and the multiply-add ports become the binding resource. Twelve
 * independent accumulator chains cover the multiply-add latency at two issues per cycle.
 *
 * @param apPanelLo Packed A panel holding tile rows `[0, 8)` (32-byte aligned).
 * @param apPanelHi Packed A panel holding tile rows `[8, 16)` (32-byte aligned).
 * @param aRowNormsSq Per-row squared norms for the 16 rows of this tile, 32-byte aligned.
 */
template <class Emit>
[[gnu::always_inline]] inline void
gemmKernel16x6Avx2F32Threshold(const float *apPanelLo, const float *apPanelHi, const float *bpPanel,
                               std::size_t kc, const float *aRowNormsSq, const float *bColNormsSq,
                               std::size_t rowBase, std::size_t colBase, std::size_t validRows,
                               std::size_t validCols, float radiusSq, Emit &&emit) {
  constexpr std::size_t kMr = 8;
  constexpr std::size_t kNr = kKernelNr<float>;
  constexpr std::size_t kRows16 = 16;

  const float *__restrict__ apLo = apPanelLo;
  const float *__restrict__ apHi = apPanelHi;
  const float *__restrict__ bpLocal = bpPanel;
  const std::size_t kcLocal = kc;

  __m256 c0lo = _mm256_setzero_ps();
  __m256 c0hi = _mm256_setzero_ps();
  __m256 c1lo = _mm256_setzero_ps();
  __m256 c1hi = _mm256_setzero_ps();
  __m256 c2lo = _mm256_setzero_ps();
  __m256 c2hi = _mm256_setzero_ps();
  __m256 c3lo = _mm256_setzero_ps();
  __m256 c3hi = _mm256_setzero_ps();
  __m256 c4lo = _mm256_setzero_ps();
  __m256 c4hi = _mm256_setzero_ps();
  __m256 c5lo = _mm256_setzero_ps();
  __m256 c5hi = _mm256_setzero_ps();

  for (std::size_t k = 0; k < kcLocal; ++k) {
    const __m256 aLo = _mm256_load_ps(apLo + (k * kMr));
    const __m256 aHi = _mm256_load_ps(apHi + (k * kMr));
    const float *bRow = bpLocal + (k * kNr);
    const __m256 b0 = _mm256_broadcast_ss(bRow + 0);
    c0lo = _mm256_fmadd_ps(aLo, b0, c0lo);
    c0hi = _mm256_fmadd_ps(aHi, b0, c0hi);
    const __m256 b1 = _mm256_broadcast_ss(bRow + 1);
    c1lo = _mm256_fmadd_ps(aLo, b1, c1lo);
    c1hi = _mm256_fmadd_ps(aHi, b1, c1hi);
    const __m256 b2 = _mm256_broadcast_ss(bRow + 2);
    c2lo = _mm256_fmadd_ps(aLo, b2, c2lo);
    c2hi = _mm256_fmadd_ps(aHi, b2, c2hi);
    const __m256 b3 = _mm256_broadcast_ss(bRow + 3);
    c3lo = _mm256_fmadd_ps(aLo, b3, c3lo);
    c3hi = _mm256_fmadd_ps(aHi, b3, c3hi);
    const __m256 b4 = _mm256_broadcast_ss(bRow + 4);
    c4lo = _mm256_fmadd_ps(aLo, b4, c4lo);
    c4hi = _mm256_fmadd_ps(aHi, b4, c4hi);
    const __m256 b5 = _mm256_broadcast_ss(bRow + 5);
    c5lo = _mm256_fmadd_ps(aLo, b5, c5lo);
    c5hi = _mm256_fmadd_ps(aHi, b5, c5hi);
  }

  const __m256 xNormsLo = _mm256_load_ps(aRowNormsSq);
  const __m256 xNormsHi = _mm256_load_ps(aRowNormsSq + kMr);
  const __m256 neg2 = _mm256_set1_ps(-2.0F);
  const __m256 zero = _mm256_setzero_ps();
  const __m256 radiusVec = _mm256_set1_ps(radiusSq);

  auto foldAndMask16 = [&](__m256 accLo, __m256 accHi,
                           std::size_t colOffset) noexcept -> std::uint16_t {
    const __m256 yNorm = _mm256_set1_ps(bColNormsSq[colOffset]);
    __m256 distLo = _mm256_fmadd_ps(accLo, neg2, _mm256_add_ps(xNormsLo, yNorm));
    __m256 distHi = _mm256_fmadd_ps(accHi, neg2, _mm256_add_ps(xNormsHi, yNorm));
    distLo = _mm256_max_ps(distLo, zero);
    distHi = _mm256_max_ps(distHi, zero);
    const auto maskLo =
        static_cast<unsigned>(_mm256_movemask_ps(_mm256_cmp_ps(distLo, radiusVec, _CMP_LE_OQ)));
    const auto maskHi =
        static_cast<unsigned>(_mm256_movemask_ps(_mm256_cmp_ps(distHi, radiusVec, _CMP_LE_OQ)));
    return static_cast<std::uint16_t>(maskLo | (maskHi << kMr));
  };

  std::array<std::uint16_t, kNr> masks{};
  if (validCols > 0) {
    masks[0] = foldAndMask16(c0lo, c0hi, 0);
  }
  if (validCols > 1) {
    masks[1] = foldAndMask16(c1lo, c1hi, 1);
  }
  if (validCols > 2) {
    masks[2] = foldAndMask16(c2lo, c2hi, 2);
  }
  if (validCols > 3) {
    masks[3] = foldAndMask16(c3lo, c3hi, 3);
  }
  if (validCols > 4) {
    masks[4] = foldAndMask16(c4lo, c4hi, 4);
  }
  if (validCols > 5) {
    masks[5] = foldAndMask16(c5lo, c5hi, 5);
  }

  const auto rowMask = (validRows >= kRows16) ? std::uint16_t{0xFFFF}
                                              : static_cast<std::uint16_t>((1U << validRows) - 1U);
  for (std::size_t c = 0; c < validCols; ++c) {
    auto m = static_cast<std::uint16_t>(masks[c] & rowMask);
    while (m != 0) {
      const int bit = __builtin_ctz(static_cast<unsigned>(m));
      emit(rowBase + static_cast<std::size_t>(bit), colBase + c);
      m = static_cast<std::uint16_t>(m & (m - 1));
    }
  }
}

#endif // CLUSTERING_USE_AVX2

} // namespace clustering::math::detail
