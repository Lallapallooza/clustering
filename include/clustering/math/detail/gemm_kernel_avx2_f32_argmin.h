#pragma once

#include <cstddef>
#include <cstdint>

#ifdef CLUSTERING_USE_AVX2
#include <immintrin.h>
#endif

#include "clustering/math/detail/gemm_kernel_scalar.h"

namespace clustering::math::detail {

#ifdef CLUSTERING_USE_AVX2

/**
 * @brief Fused 8x6 AVX2 f32 microkernel: accumulate @c -2*X*C^T over the full K and fold
 *        running @c (minDistSq, argmin) per M-row.
 *
 * Computes a column-by-column inner product of an 8-row A-panel against a 6-column B-panel.
 * After the @c K-loop, folds @c +cSqNorms[j] into each column's accumulator (giving
 * @c ||c_j||^2 - 2*x_i.c_j, which is @c ||x_i - c_j||^2 - ||x_i||^2 -- shifting every column
 * by the same constant per row is safe for the running argmin comparison) and updates a pair
 * of @c (bestMin, bestArg) YMM registers via @c _mm256_cmp_ps + @c _mm256_blendv_ps with
 * @c _CMP_LT_OQ. Strict less-than mirrors @c math::argmin's earliest-index-on-tie semantics.
 *
 * Buffer layouts match @c packA / @c packB from @c gemm_pack.h:
 *   - @p apPanel holds an @c Mr x @c kc A-panel; element @c (r, k) is at @c ap[k*Mr + r].
 *   - @p bpPanel holds a @c kc x @c Nr B-panel; element @c (k, c) is at @c bp[k*Nr + c].
 *
 * @p bestMin and @p bestArg hold per-row running state across successive N-panel invocations
 * of this kernel; the outer driver resets them per M-tile and reads them back at the M-tile
 * epilogue to emit labels and minimum distances.
 *
 * @param apPanel          Packed A panel (32-byte aligned).
 * @param bpPanel          Packed B panel (32-byte aligned).
 * @param cSqNormsPanel    Per-column squared norms for the Nr columns of this panel; padded
 *                         entries must be @c FLT_MAX so they never win the argmin contest.
 * @param kc               Inner dimension count.
 * @param jBase            Global column index of column 0 of this B-panel; contributes to
 *                         @p bestArg via @c jBase + columnOffset.
 * @param bestMin          In/out: per-row running minimum of
 *                         @c ||c_j||^2 - 2*x_i.c_j across all columns scanned so far.
 * @param bestArg          In/out: per-row argmin of the columns scanned so far.
 */
inline void gemmKernel8x6Avx2F32FusedArgmin(const float *apPanel, const float *bpPanel,
                                            const float *cSqNormsPanel, std::size_t kc,
                                            std::int32_t jBase, __m256 &bestMin,
                                            __m256i &bestArg) noexcept {
  constexpr std::size_t kMr = kKernelMr<float>;
  constexpr std::size_t kNr = kKernelNr<float>;

  // __restrict__ locals captured at the kernel top per the codegen-discipline convention. The
  // compiler must prove non-aliasing across the K-loop before it can pin the six accumulators
  // in YMM registers and fully unroll the inner body.
  const float *__restrict__ apLocal = apPanel;
  const float *__restrict__ bpLocal = bpPanel;
  const float *__restrict__ npLocal = cSqNormsPanel;
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

  // Fold -2 * dot product + ||c_j||^2 per column. ||x_i||^2 is a per-row constant across all
  // columns and is added once at the M-tile epilogue; its absence here does not affect the
  // relative ordering of candidates for the argmin contest.
  const __m256 neg2 = _mm256_set1_ps(-2.0F);
  c0 = _mm256_fmadd_ps(c0, neg2, _mm256_set1_ps(npLocal[0]));
  c1 = _mm256_fmadd_ps(c1, neg2, _mm256_set1_ps(npLocal[1]));
  c2 = _mm256_fmadd_ps(c2, neg2, _mm256_set1_ps(npLocal[2]));
  c3 = _mm256_fmadd_ps(c3, neg2, _mm256_set1_ps(npLocal[3]));
  c4 = _mm256_fmadd_ps(c4, neg2, _mm256_set1_ps(npLocal[4]));
  c5 = _mm256_fmadd_ps(c5, neg2, _mm256_set1_ps(npLocal[5]));

  // Per-column running-min / argmin. Strict less-than so the earliest column wins on ties.
  auto updateBest = [&](__m256 cand, std::int32_t jOffset) {
    const __m256 mask = _mm256_cmp_ps(cand, bestMin, _CMP_LT_OQ);
    bestMin = _mm256_blendv_ps(bestMin, cand, mask);
    const __m256i jVec = _mm256_set1_epi32(jBase + jOffset);
    bestArg = _mm256_blendv_epi8(bestArg, jVec, _mm256_castps_si256(mask));
  };
  updateBest(c0, 0);
  updateBest(c1, 1);
  updateBest(c2, 2);
  updateBest(c3, 3);
  updateBest(c4, 4);
  updateBest(c5, 5);
}

#endif // CLUSTERING_USE_AVX2

} // namespace clustering::math::detail
