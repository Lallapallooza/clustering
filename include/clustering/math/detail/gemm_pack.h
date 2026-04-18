#pragma once

#include <cstddef>
#include <type_traits>

#include "clustering/math/detail/gemm_kernel_scalar.h"
#include "clustering/math/detail/matrix_desc.h"

#ifdef CLUSTERING_USE_AVX2
#include <immintrin.h>
#endif

namespace clustering::math::detail {

#ifdef CLUSTERING_USE_AVX2

/**
 * @brief In-register 8x8 f32 transpose; loads eight unaligned 8-lane rows and stores eight
 *        aligned 8-lane columns, column-major with destination stride 32B.
 *
 * Each source pointer addresses a contiguous 8-float row; @p dst must be 32-byte aligned and
 * has capacity for 8 x 32B stores. Callers place the eight rows at row-stride @p srcRowStride
 * apart. The destination layout is the @c Ap panel layout the 8x6 microkernel consumes:
 * @c dst[k*8 + r] holds @c src[r][k].
 */
[[gnu::always_inline]] inline void
transpose8x8Avx2F32(const float *src, std::ptrdiff_t srcRowStride, float *dst) noexcept {
  const __m256 v0 = _mm256_loadu_ps(src + (0 * srcRowStride));
  const __m256 v1 = _mm256_loadu_ps(src + (1 * srcRowStride));
  const __m256 v2 = _mm256_loadu_ps(src + (2 * srcRowStride));
  const __m256 v3 = _mm256_loadu_ps(src + (3 * srcRowStride));
  const __m256 v4 = _mm256_loadu_ps(src + (4 * srcRowStride));
  const __m256 v5 = _mm256_loadu_ps(src + (5 * srcRowStride));
  const __m256 v6 = _mm256_loadu_ps(src + (6 * srcRowStride));
  const __m256 v7 = _mm256_loadu_ps(src + (7 * srcRowStride));

  const __m256 t0 = _mm256_unpacklo_ps(v0, v1);
  const __m256 t1 = _mm256_unpackhi_ps(v0, v1);
  const __m256 t2 = _mm256_unpacklo_ps(v2, v3);
  const __m256 t3 = _mm256_unpackhi_ps(v2, v3);
  const __m256 t4 = _mm256_unpacklo_ps(v4, v5);
  const __m256 t5 = _mm256_unpackhi_ps(v4, v5);
  const __m256 t6 = _mm256_unpacklo_ps(v6, v7);
  const __m256 t7 = _mm256_unpackhi_ps(v6, v7);

  const __m256 s0 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(1, 0, 1, 0));
  const __m256 s1 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(3, 2, 3, 2));
  const __m256 s2 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(1, 0, 1, 0));
  const __m256 s3 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(3, 2, 3, 2));
  const __m256 s4 = _mm256_shuffle_ps(t4, t6, _MM_SHUFFLE(1, 0, 1, 0));
  const __m256 s5 = _mm256_shuffle_ps(t4, t6, _MM_SHUFFLE(3, 2, 3, 2));
  const __m256 s6 = _mm256_shuffle_ps(t5, t7, _MM_SHUFFLE(1, 0, 1, 0));
  const __m256 s7 = _mm256_shuffle_ps(t5, t7, _MM_SHUFFLE(3, 2, 3, 2));

  _mm256_store_ps(dst + 0U, _mm256_permute2f128_ps(s0, s4, 0x20));
  _mm256_store_ps(dst + 8U, _mm256_permute2f128_ps(s1, s5, 0x20));
  _mm256_store_ps(dst + 16U, _mm256_permute2f128_ps(s2, s6, 0x20));
  _mm256_store_ps(dst + 24U, _mm256_permute2f128_ps(s3, s7, 0x20));
  _mm256_store_ps(dst + 32U, _mm256_permute2f128_ps(s0, s4, 0x31));
  _mm256_store_ps(dst + 40U, _mm256_permute2f128_ps(s1, s5, 0x31));
  _mm256_store_ps(dst + 48U, _mm256_permute2f128_ps(s2, s6, 0x31));
  _mm256_store_ps(dst + 56U, _mm256_permute2f128_ps(s3, s7, 0x31));
}

/**
 * @brief Pack an @c mc x @c kc block of a dense row-major f32 source into the 8-row panel
 *        layout the 8x6 microkernel consumes, via 8x8 AVX2 transpose tiles.
 *
 * Assumes @c aColStride == 1 (caller dispatches on the descriptor). The @p apOut arena is
 * 32-byte aligned by construction; each 32B destination store inside the transpose tile is
 * aligned. Unaligned AVX2 loads absorb the @c pc-offset.
 */
inline void packA_f32_RowMajorAvx2(const float *aBase, std::ptrdiff_t aRowStride, std::size_t ic,
                                   std::size_t mc, std::size_t pc, std::size_t kc,
                                   float *apOut) noexcept {
  constexpr std::size_t kMr = 8;
  const std::size_t panels = (mc + kMr - 1) / kMr;

  for (std::size_t p = 0; p < panels; ++p) {
    float *panelOut = apOut + (p * kMr * kc);
    const std::size_t rBase = p * kMr;
    const std::size_t rValid = (rBase + kMr <= mc) ? kMr : (mc - rBase);

    if (rValid == kMr) {
      const float *rowBase = aBase + (static_cast<std::ptrdiff_t>(ic + rBase) * aRowStride) +
                             static_cast<std::ptrdiff_t>(pc);

      std::size_t k = 0;
      for (; k + 8 <= kc; k += 8) {
        transpose8x8Avx2F32(rowBase + k, aRowStride, panelOut + (k * kMr));
      }
      for (; k < kc; ++k) {
        const float *src = rowBase + k;
        float *col = panelOut + (k * kMr);
        for (std::size_t r = 0; r < kMr; ++r) {
          col[r] = src[static_cast<std::ptrdiff_t>(r) * aRowStride];
        }
      }
    } else {
      for (std::size_t k = 0; k < kc; ++k) {
        float *col = panelOut + (k * kMr);
        for (std::size_t r = 0; r < rValid; ++r) {
          const std::ptrdiff_t srcRow = static_cast<std::ptrdiff_t>(ic + rBase + r) * aRowStride;
          col[r] = aBase[srcRow + static_cast<std::ptrdiff_t>(pc + k)];
        }
        for (std::size_t r = rValid; r < kMr; ++r) {
          col[r] = 0.0F;
        }
      }
    }
  }
}

#endif // CLUSTERING_USE_AVX2

/**
 * @brief Pack an @c Mc x @c Kc block of @c A into the microkernel-friendly @c Ap layout.
 *
 * Output is a sequence of @c ceil(mc / Mr) panels. Panel @c p is @c Mr rows by @c kc columns;
 * element @c (r, k) (with @c r the panel-local row and @c k the inner-dim index) lands at
 * @c apOut[p*Mr*kc + k*Mr + r]. The last panel zero-pads any rows beyond @c (mc - p*Mr) for
 * every @c k in @c [0, kc), so the kernel's accumulator math remains valid for tail tiles.
 *
 * @tparam T  Element type.
 * @param Ad  Source matrix descriptor (@c MatrixDescC<T>); arbitrary strides are supported.
 * @param ic  Starting row in @c Ad of the block to pack.
 * @param mc  Row count of the block (last panel zero-pads if @c mc % Mr != 0).
 * @param pc  Starting column in @c Ad of the block.
 * @param kc  Column count of the block.
 * @param apOut Destination buffer of capacity @c ceil(mc/Mr) * Mr * kc, 32-byte aligned.
 */
template <class T>
void packA(const ::clustering::detail::MatrixDescC<T> &Ad, std::size_t ic, std::size_t mc,
           std::size_t pc, std::size_t kc, T *apOut) noexcept {
  constexpr std::size_t kMr = kKernelMr<T>;

  const std::ptrdiff_t aRowStride = Ad.rowStride;
  const std::ptrdiff_t aColStride = Ad.colStride;
  const T *aBase = Ad.ptr;

#ifdef CLUSTERING_USE_AVX2
  // Dense-row fast path: contiguous k-direction collapses the (r, k) transpose-pack into 8x8
  // AVX2 tiles so each source row is consumed by one aligned store group instead of Mr scalar
  // stores per k-step.
  if constexpr (std::is_same_v<T, float>) {
    if (aColStride == 1) {
      packA_f32_RowMajorAvx2(aBase, aRowStride, ic, mc, pc, kc, apOut);
      return;
    }
  }
#endif

  const std::size_t panels = (mc + kMr - 1) / kMr;
  for (std::size_t p = 0; p < panels; ++p) {
    T *panelOut = apOut + (p * kMr * kc);
    const std::size_t rBase = p * kMr;
    const std::size_t rValid = (rBase + kMr <= mc) ? kMr : (mc - rBase);

    for (std::size_t k = 0; k < kc; ++k) {
      T *col = panelOut + (k * kMr);
      const std::ptrdiff_t srcCol = (static_cast<std::ptrdiff_t>(pc + k)) * aColStride;
      for (std::size_t r = 0; r < rValid; ++r) {
        const std::ptrdiff_t srcRow = (static_cast<std::ptrdiff_t>(ic + rBase + r)) * aRowStride;
        col[r] = aBase[srcRow + srcCol];
      }
      for (std::size_t r = rValid; r < kMr; ++r) {
        col[r] = T{0};
      }
    }
  }
}

/**
 * @brief Pack a @c Kc x @c Nc block of @c B into the microkernel-friendly @c Bp layout.
 *
 * Output is a sequence of @c ceil(nc / Nr) panels. Panel @c p is @c kc rows by @c Nr columns;
 * element @c (k, c) (with @c c the panel-local column) lands at
 * @c bpOut[p*kc*Nr + k*Nr + c]. The last panel zero-pads any columns beyond @c (nc - p*Nr).
 *
 * @tparam T  Element type.
 * @param Bd  Source matrix descriptor (@c MatrixDescC<T>); arbitrary strides are supported.
 * @param pc  Starting row in @c Bd of the block to pack.
 * @param kc  Row count of the block.
 * @param jc  Starting column in @c Bd of the block.
 * @param nc  Column count of the block (last panel zero-pads if @c nc % Nr != 0).
 * @param bpOut Destination buffer of capacity @c ceil(nc/Nr) * kc * Nr, 32-byte aligned.
 */
template <class T>
void packB(const ::clustering::detail::MatrixDescC<T> &Bd, std::size_t pc, std::size_t kc,
           std::size_t jc, std::size_t nc, T *bpOut) noexcept {
  constexpr std::size_t kNr = kKernelNr<T>;

  const std::ptrdiff_t bRowStride = Bd.rowStride;
  const std::ptrdiff_t bColStride = Bd.colStride;
  const T *bBase = Bd.ptr;

  const std::size_t panels = (nc + kNr - 1) / kNr;
  for (std::size_t p = 0; p < panels; ++p) {
    T *panelOut = bpOut + (p * kc * kNr);
    const std::size_t cBase = p * kNr;
    const std::size_t cValid = (cBase + kNr <= nc) ? kNr : (nc - cBase);

    for (std::size_t k = 0; k < kc; ++k) {
      T *row = panelOut + (k * kNr);
      const std::ptrdiff_t srcRow = (static_cast<std::ptrdiff_t>(pc + k)) * bRowStride;
      for (std::size_t c = 0; c < cValid; ++c) {
        const std::ptrdiff_t srcCol = (static_cast<std::ptrdiff_t>(jc + cBase + c)) * bColStride;
        row[c] = bBase[srcRow + srcCol];
      }
      for (std::size_t c = cValid; c < kNr; ++c) {
        row[c] = T{0};
      }
    }
  }
}

} // namespace clustering::math::detail
