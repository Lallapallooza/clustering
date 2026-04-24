#pragma once

#include <array>
#include <cstddef>
#include <type_traits>

#include "clustering/math/detail/avx2_helpers.h"

#ifdef CLUSTERING_USE_AVX2
#include <immintrin.h>
#endif

namespace clustering::math::detail {

/**
 * @brief Squared Euclidean distances from a single query to a contiguous AoS block of points.
 *
 * Writes `out[0..count-1]` = @c sum_k (query[k] - leaf[i*d + k])^2 for each @c i. Designed as
 * the hot kernel of KDTree leaf brute-force scans (Boruvka per-round traversal, kNN walk).
 * Compared to calling @ref sqEuclideanRowPtr in a loop, this amortises the horizontal-sum
 * epilogue across four neighbours at a time: four independent @c ymm accumulators collapse
 * through two paired @c hadd_ps plus a cross-lane add into an @c xmm holding four finished
 * distances with a single reduction tree per block of four.
 *
 * The batched hsum cuts the per-distance cost nearly in half at @c d=8 (ten-op scalar reduction
 * replaced by a five-op shared reduction) and stays beneficial up to a few dozen dimensions,
 * where the per-dim load/sub/fmadd work dominates again. Scalar fallback exactly mirrors
 * @ref sqEuclideanRowPtr.
 *
 * @tparam T Element type (@c float or @c double).
 * @param query Contiguous @p d-element query row.
 * @param leaf  Base of @c count rows, each @p d contiguous elements.
 * @param count Number of leaf rows to process.
 * @param d     Per-row element count; stride inside @p leaf.
 * @param out   Length-@p count output; each entry is the squared distance to the matching row.
 */
template <class T>
inline void sqDistancesAosBlock(const T *query, const T *leaf, std::size_t count, std::size_t d,
                                T *out) noexcept {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                "sqDistancesAosBlock: T must be float or double.");
#ifdef CLUSTERING_USE_AVX2
  if constexpr (std::is_same_v<T, float>) {
    if (d >= kAvx2Lanes<T>) {
      std::size_t i = 0;
      for (; i + 4 <= count; i += 4) {
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps();
        __m256 acc3 = _mm256_setzero_ps();
        const float *p0 = leaf + ((i + 0) * d);
        const float *p1 = leaf + ((i + 1) * d);
        const float *p2 = leaf + ((i + 2) * d);
        const float *p3 = leaf + ((i + 3) * d);
        std::size_t k = 0;
        for (; k + 8 <= d; k += 8) {
          const __m256 q = _mm256_loadu_ps(query + k);
          const __m256 r0 = _mm256_loadu_ps(p0 + k);
          const __m256 r1 = _mm256_loadu_ps(p1 + k);
          const __m256 r2 = _mm256_loadu_ps(p2 + k);
          const __m256 r3 = _mm256_loadu_ps(p3 + k);
          const __m256 d0 = _mm256_sub_ps(r0, q);
          const __m256 d1 = _mm256_sub_ps(r1, q);
          const __m256 d2 = _mm256_sub_ps(r2, q);
          const __m256 d3 = _mm256_sub_ps(r3, q);
          acc0 = _mm256_fmadd_ps(d0, d0, acc0);
          acc1 = _mm256_fmadd_ps(d1, d1, acc1);
          acc2 = _mm256_fmadd_ps(d2, d2, acc2);
          acc3 = _mm256_fmadd_ps(d3, d3, acc3);
        }
        // Collapse four 8-lane accumulators into a single xmm of four distances. The pair of
        // hadds across (acc0, acc1) and (acc2, acc3) brings each accumulator down to 4 partial
        // sums per row and interleaves pairs into one ymm; a third hadd plus a cross-lane add
        // finishes the reduction without per-distance movs or extracts.
        const __m256 h01 = _mm256_hadd_ps(acc0, acc1);
        const __m256 h23 = _mm256_hadd_ps(acc2, acc3);
        const __m256 h = _mm256_hadd_ps(h01, h23);
        const __m128 lo = _mm256_castps256_ps128(h);
        const __m128 hi = _mm256_extractf128_ps(h, 1);
        __m128 dists = _mm_add_ps(lo, hi);
        // Tail dims (d % 8 != 0): accumulate into a scalar per row and fold into the xmm.
        if (k < d) {
          alignas(16) std::array<float, 4> tail{0.0F, 0.0F, 0.0F, 0.0F};
          for (std::size_t j = k; j < d; ++j) {
            const float q = query[j];
            const float e0 = p0[j] - q;
            const float e1 = p1[j] - q;
            const float e2 = p2[j] - q;
            const float e3 = p3[j] - q;
            tail[0] += e0 * e0;
            tail[1] += e1 * e1;
            tail[2] += e2 * e2;
            tail[3] += e3 * e3;
          }
          dists = _mm_add_ps(dists, _mm_load_ps(tail.data()));
        }
        _mm_storeu_ps(out + i, dists);
      }
      // Scalar tail for the last (count % 4) rows.
      for (; i < count; ++i) {
        out[i] = sqEuclideanRowPtr(query, leaf + (i * d), d);
      }
      return;
    }
  }
#endif
  for (std::size_t i = 0; i < count; ++i) {
    out[i] = sqEuclideanRowPtr(query, leaf + (i * d), d);
  }
}

} // namespace clustering::math::detail
