#pragma once

#include <cstddef>

#ifdef CLUSTERING_USE_AVX2
#include <immintrin.h>
#endif

#include "clustering/math/detail/gemm_kernel_scalar.h"

namespace clustering::math::detail {

#ifdef CLUSTERING_USE_AVX2

/**
 * @brief AVX2 f32 8x6 microkernel; produces the same column-major @c Mr x @c Nr output tile as
 *        @c gemmKernelMrNrScalar<float, Beta>.
 *
 * Six @c __m256 accumulators (one per output column) each carry all eight row-elements of that
 * column in a single YMM register. The K-step issues one aligned 32-byte load from the packed A
 * panel, six scalar broadcasts from the packed B panel, and six FMAs that accumulate
 * @c a * broadcast(b_c) into @c acc[c]. The column-major tile lets the epilogue flush each
 * accumulator with a single aligned @c _mm256_store_ps at @c tile + c*Mr.
 *
 * Buffer layouts are identical to @c gemmKernelMrNrScalar<float, Beta>; see that declaration
 * for the @c ap / @c bp / @c tile contracts and the tail-protocol invariant. The AVX2 kernel
 * requires @p ap, @p bp, and @p tile to be 32-byte aligned (arena + @c alignas(32) scratch).
 *
 * @tparam Beta Compile-time BetaKind selecting the epilogue. @c kZero writes the tile from the
 *              alpha-scaled product only (never reads @p tile); @c kGeneral reads @p tile, scales
 *              by @p beta, and fuses the alpha-scaled product on top.
 * @param ap    Packed A panel (32-byte aligned).
 * @param bp    Packed B panel (32-byte aligned).
 * @param tile  @c Mr x @c Nr column-major scratch tile (32-byte aligned).
 * @param kc    Inner-dimension count.
 * @param alpha Scalar multiplier on @c A*B.
 * @param beta  Scalar multiplier on the prior tile content; ignored on @c kZero.
 */
template <BetaKind Beta>
void gemmKernel8x6Avx2F32(const float *ap, const float *bp, float *tile, std::size_t kc,
                          float alpha, float beta) noexcept {
  constexpr std::size_t kMr = kKernelMr<float>;
  constexpr std::size_t kNr = kKernelNr<float>;

  // __restrict__ locals captured at the kernel top per the codegen-discipline convention. The
  // compiler must prove non-aliasing across the K-loop before it can pin the six accumulators
  // in YMM registers and fully unroll the inner body; any descriptor-field read inside the
  // loop body breaks that proof.
  const float *__restrict__ apLocal = ap;
  const float *__restrict__ bpLocal = bp;
  float *__restrict__ tLocal = tile;
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

  const __m256 alphaV = _mm256_broadcast_ss(&alpha);

  if constexpr (Beta == BetaKind::kZero) {
    _mm256_store_ps(tLocal + (0 * kMr), _mm256_mul_ps(alphaV, c0));
    _mm256_store_ps(tLocal + (1 * kMr), _mm256_mul_ps(alphaV, c1));
    _mm256_store_ps(tLocal + (2 * kMr), _mm256_mul_ps(alphaV, c2));
    _mm256_store_ps(tLocal + (3 * kMr), _mm256_mul_ps(alphaV, c3));
    _mm256_store_ps(tLocal + (4 * kMr), _mm256_mul_ps(alphaV, c4));
    _mm256_store_ps(tLocal + (5 * kMr), _mm256_mul_ps(alphaV, c5));
  } else {
    const __m256 betaV = _mm256_broadcast_ss(&beta);
    const __m256 t0 = _mm256_mul_ps(betaV, _mm256_load_ps(tLocal + (0 * kMr)));
    const __m256 t1 = _mm256_mul_ps(betaV, _mm256_load_ps(tLocal + (1 * kMr)));
    const __m256 t2 = _mm256_mul_ps(betaV, _mm256_load_ps(tLocal + (2 * kMr)));
    const __m256 t3 = _mm256_mul_ps(betaV, _mm256_load_ps(tLocal + (3 * kMr)));
    const __m256 t4 = _mm256_mul_ps(betaV, _mm256_load_ps(tLocal + (4 * kMr)));
    const __m256 t5 = _mm256_mul_ps(betaV, _mm256_load_ps(tLocal + (5 * kMr)));
    _mm256_store_ps(tLocal + (0 * kMr), _mm256_fmadd_ps(alphaV, c0, t0));
    _mm256_store_ps(tLocal + (1 * kMr), _mm256_fmadd_ps(alphaV, c1, t1));
    _mm256_store_ps(tLocal + (2 * kMr), _mm256_fmadd_ps(alphaV, c2, t2));
    _mm256_store_ps(tLocal + (3 * kMr), _mm256_fmadd_ps(alphaV, c3, t3));
    _mm256_store_ps(tLocal + (4 * kMr), _mm256_fmadd_ps(alphaV, c4, t4));
    _mm256_store_ps(tLocal + (5 * kMr), _mm256_fmadd_ps(alphaV, c5, t5));
  }
}

#endif // CLUSTERING_USE_AVX2

} // namespace clustering::math::detail
