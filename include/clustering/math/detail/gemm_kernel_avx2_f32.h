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
 *        `gemmKernelMrNrScalar<float, Beta>`.
 *
 * Two independent accumulator chains per output column carry even and odd K steps, then fold at
 * the epilogue. The column-major tile lets the epilogue flush each accumulator with a single
 * aligned @c _mm256_store_ps at @c tile + c*Mr.
 *
 * Buffer layouts are identical to `gemmKernelMrNrScalar<float, Beta>`; see that declaration
 * for the @c ap / @c bp / @c tile contracts and the tail-protocol invariant. The AVX2 kernel
 * requires @p ap, @p bp, and @p tile to be 32-byte aligned (arena + `alignas(32)` scratch).
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
    const __m256 a = _mm256_load_ps(apLocal + (k * kMr));
    const __m256 b = _mm256_load_ps(apLocal + ((k + 1) * kMr));
    const float *bRow = bpLocal + (k * kNr);
    const float *bRowNext = bpLocal + ((k + 1) * kNr);
    c0a = _mm256_fmadd_ps(a, _mm256_broadcast_ss(bRow + 0), c0a);
    c0b = _mm256_fmadd_ps(b, _mm256_broadcast_ss(bRowNext + 0), c0b);
    c1a = _mm256_fmadd_ps(a, _mm256_broadcast_ss(bRow + 1), c1a);
    c1b = _mm256_fmadd_ps(b, _mm256_broadcast_ss(bRowNext + 1), c1b);
    c2a = _mm256_fmadd_ps(a, _mm256_broadcast_ss(bRow + 2), c2a);
    c2b = _mm256_fmadd_ps(b, _mm256_broadcast_ss(bRowNext + 2), c2b);
    c3a = _mm256_fmadd_ps(a, _mm256_broadcast_ss(bRow + 3), c3a);
    c3b = _mm256_fmadd_ps(b, _mm256_broadcast_ss(bRowNext + 3), c3b);
    c4a = _mm256_fmadd_ps(a, _mm256_broadcast_ss(bRow + 4), c4a);
    c4b = _mm256_fmadd_ps(b, _mm256_broadcast_ss(bRowNext + 4), c4b);
    c5a = _mm256_fmadd_ps(a, _mm256_broadcast_ss(bRow + 5), c5a);
    c5b = _mm256_fmadd_ps(b, _mm256_broadcast_ss(bRowNext + 5), c5b);
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

/**
 * @brief AVX2 f32 tail microkernel for a packed 8x6 B panel with four live columns.
 *
 * The packed-B stride remains `kKernelNr<float>`; only columns 0 through 3 are accumulated and
 * written to @p tile. The caller owns the tail dispatch and writes back only the live columns.
 */
template <BetaKind Beta>
void gemmKernel8x4Avx2F32(const float *ap, const float *bp, float *tile, std::size_t kc,
                          float alpha, float beta) noexcept {
  constexpr std::size_t kMr = kKernelMr<float>;
  constexpr std::size_t kNr = kKernelNr<float>;

  const float *__restrict__ apLocal = ap;
  const float *__restrict__ bpLocal = bp;
  float *__restrict__ tLocal = tile;
  const std::size_t kcLocal = kc;

  __m256 c0a = _mm256_setzero_ps();
  __m256 c0b = _mm256_setzero_ps();
  __m256 c1a = _mm256_setzero_ps();
  __m256 c1b = _mm256_setzero_ps();
  __m256 c2a = _mm256_setzero_ps();
  __m256 c2b = _mm256_setzero_ps();
  __m256 c3a = _mm256_setzero_ps();
  __m256 c3b = _mm256_setzero_ps();

  std::size_t k = 0;
  for (; k + 1 < kcLocal; k += 2) {
    const __m256 a = _mm256_load_ps(apLocal + (k * kMr));
    const __m256 b = _mm256_load_ps(apLocal + ((k + 1) * kMr));
    const float *bRow = bpLocal + (k * kNr);
    const float *bRowNext = bpLocal + ((k + 1) * kNr);
    c0a = _mm256_fmadd_ps(a, _mm256_broadcast_ss(bRow + 0), c0a);
    c0b = _mm256_fmadd_ps(b, _mm256_broadcast_ss(bRowNext + 0), c0b);
    c1a = _mm256_fmadd_ps(a, _mm256_broadcast_ss(bRow + 1), c1a);
    c1b = _mm256_fmadd_ps(b, _mm256_broadcast_ss(bRowNext + 1), c1b);
    c2a = _mm256_fmadd_ps(a, _mm256_broadcast_ss(bRow + 2), c2a);
    c2b = _mm256_fmadd_ps(b, _mm256_broadcast_ss(bRowNext + 2), c2b);
    c3a = _mm256_fmadd_ps(a, _mm256_broadcast_ss(bRow + 3), c3a);
    c3b = _mm256_fmadd_ps(b, _mm256_broadcast_ss(bRowNext + 3), c3b);
  }
  if (k < kcLocal) {
    const __m256 a = _mm256_load_ps(apLocal + (k * kMr));
    const float *bRow = bpLocal + (k * kNr);
    c0a = _mm256_fmadd_ps(a, _mm256_broadcast_ss(bRow + 0), c0a);
    c1a = _mm256_fmadd_ps(a, _mm256_broadcast_ss(bRow + 1), c1a);
    c2a = _mm256_fmadd_ps(a, _mm256_broadcast_ss(bRow + 2), c2a);
    c3a = _mm256_fmadd_ps(a, _mm256_broadcast_ss(bRow + 3), c3a);
  }

  const __m256 c0 = _mm256_add_ps(c0a, c0b);
  const __m256 c1 = _mm256_add_ps(c1a, c1b);
  const __m256 c2 = _mm256_add_ps(c2a, c2b);
  const __m256 c3 = _mm256_add_ps(c3a, c3b);
  const __m256 alphaV = _mm256_broadcast_ss(&alpha);

  if constexpr (Beta == BetaKind::kZero) {
    _mm256_store_ps(tLocal + (0 * kMr), _mm256_mul_ps(alphaV, c0));
    _mm256_store_ps(tLocal + (1 * kMr), _mm256_mul_ps(alphaV, c1));
    _mm256_store_ps(tLocal + (2 * kMr), _mm256_mul_ps(alphaV, c2));
    _mm256_store_ps(tLocal + (3 * kMr), _mm256_mul_ps(alphaV, c3));
  } else {
    const __m256 betaV = _mm256_broadcast_ss(&beta);
    const __m256 t0 = _mm256_mul_ps(betaV, _mm256_load_ps(tLocal + (0 * kMr)));
    const __m256 t1 = _mm256_mul_ps(betaV, _mm256_load_ps(tLocal + (1 * kMr)));
    const __m256 t2 = _mm256_mul_ps(betaV, _mm256_load_ps(tLocal + (2 * kMr)));
    const __m256 t3 = _mm256_mul_ps(betaV, _mm256_load_ps(tLocal + (3 * kMr)));
    _mm256_store_ps(tLocal + (0 * kMr), _mm256_fmadd_ps(alphaV, c0, t0));
    _mm256_store_ps(tLocal + (1 * kMr), _mm256_fmadd_ps(alphaV, c1, t1));
    _mm256_store_ps(tLocal + (2 * kMr), _mm256_fmadd_ps(alphaV, c2, t2));
    _mm256_store_ps(tLocal + (3 * kMr), _mm256_fmadd_ps(alphaV, c3, t3));
  }
}

#endif // CLUSTERING_USE_AVX2

} // namespace clustering::math::detail
