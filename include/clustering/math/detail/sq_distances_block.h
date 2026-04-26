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
 * the hot kernel of KDTree leaf brute-force scans (Boruvka per-round traversal, kNN walk) and
 * the AFK-MC2 chain steps. Compared to calling @ref sqEuclideanRowPtr in a loop, this kernel
 * amortises the horizontal-sum epilogue across four neighbours at a time and breaks the
 * per-leaf FMA chain into two interleaved chains so the inner loop runs throughput-bound on
 * Zen rather than latency-bound.
 *
 * Per d=16 step (two 8-feature chunks), each leaf consumes one even-chunk FMA (into @c ae) and
 * one odd-chunk FMA (into @c ao). The two chains are independent so multiple FMAs can issue
 * before the prior FMA's result is back, halving the latency-bound critical path versus a
 * single-accumulator chain. The 4-leaf hsum tree (two paired @c hadd_ps plus a cross-lane add)
 * collapses the eight ymms into one xmm holding four distances with a single reduction per
 * block.
 *
 * Falls back to a single-chain 8-feature step for the trailing @c d%16 in `[8, 15]`, then a
 * scalar tail for @c d%8 != 0. Scalar fallback for `d < 8` mirrors @ref sqEuclideanRowPtr.
 *
 * @tparam T Element type (@c float or @c double).
 * @param query Contiguous @p d-element query row.
 * @param leaf  Base of @c count rows, each @p d contiguous elements.
 * @param count Number of leaf rows to process.
 * @param d     Per-row element count; stride inside @p leaf.
 * @param out   Length-@p count output; each entry is the squared distance to the matching row.
 */
template <class T>
[[gnu::always_inline]] inline void sqDistancesAosBlock(const T *query, const T *leaf,
                                                       std::size_t count, std::size_t d,
                                                       T *out) noexcept {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                "sqDistancesAosBlock: T must be float or double.");
#ifdef CLUSTERING_USE_AVX2
  if constexpr (std::is_same_v<T, float>) {
    if (d >= kAvx2Lanes<T>) {
      std::size_t i = 0;
      for (; i + 4 <= count; i += 4) {
        __m256 ae0 = _mm256_setzero_ps();
        __m256 ae1 = _mm256_setzero_ps();
        __m256 ae2 = _mm256_setzero_ps();
        __m256 ae3 = _mm256_setzero_ps();
        __m256 ao0 = _mm256_setzero_ps();
        __m256 ao1 = _mm256_setzero_ps();
        __m256 ao2 = _mm256_setzero_ps();
        __m256 ao3 = _mm256_setzero_ps();
        const float *p0 = leaf + ((i + 0) * d);
        const float *p1 = leaf + ((i + 1) * d);
        const float *p2 = leaf + ((i + 2) * d);
        const float *p3 = leaf + ((i + 3) * d);
        std::size_t k = 0;
        for (; k + 16 <= d; k += 16) {
          const __m256 q0 = _mm256_loadu_ps(query + k);
          const __m256 d0 = _mm256_sub_ps(_mm256_loadu_ps(p0 + k), q0);
          const __m256 d1 = _mm256_sub_ps(_mm256_loadu_ps(p1 + k), q0);
          const __m256 d2 = _mm256_sub_ps(_mm256_loadu_ps(p2 + k), q0);
          const __m256 d3 = _mm256_sub_ps(_mm256_loadu_ps(p3 + k), q0);
          ae0 = _mm256_fmadd_ps(d0, d0, ae0);
          ae1 = _mm256_fmadd_ps(d1, d1, ae1);
          ae2 = _mm256_fmadd_ps(d2, d2, ae2);
          ae3 = _mm256_fmadd_ps(d3, d3, ae3);
          const __m256 q1 = _mm256_loadu_ps(query + k + 8);
          const __m256 e0 = _mm256_sub_ps(_mm256_loadu_ps(p0 + k + 8), q1);
          const __m256 e1 = _mm256_sub_ps(_mm256_loadu_ps(p1 + k + 8), q1);
          const __m256 e2 = _mm256_sub_ps(_mm256_loadu_ps(p2 + k + 8), q1);
          const __m256 e3 = _mm256_sub_ps(_mm256_loadu_ps(p3 + k + 8), q1);
          ao0 = _mm256_fmadd_ps(e0, e0, ao0);
          ao1 = _mm256_fmadd_ps(e1, e1, ao1);
          ao2 = _mm256_fmadd_ps(e2, e2, ao2);
          ao3 = _mm256_fmadd_ps(e3, e3, ao3);
        }
        if (k + 8 <= d) {
          const __m256 q = _mm256_loadu_ps(query + k);
          const __m256 d0 = _mm256_sub_ps(_mm256_loadu_ps(p0 + k), q);
          const __m256 d1 = _mm256_sub_ps(_mm256_loadu_ps(p1 + k), q);
          const __m256 d2 = _mm256_sub_ps(_mm256_loadu_ps(p2 + k), q);
          const __m256 d3 = _mm256_sub_ps(_mm256_loadu_ps(p3 + k), q);
          ae0 = _mm256_fmadd_ps(d0, d0, ae0);
          ae1 = _mm256_fmadd_ps(d1, d1, ae1);
          ae2 = _mm256_fmadd_ps(d2, d2, ae2);
          ae3 = _mm256_fmadd_ps(d3, d3, ae3);
          k += 8;
        }
        const __m256 acc0 = _mm256_add_ps(ae0, ao0);
        const __m256 acc1 = _mm256_add_ps(ae1, ao1);
        const __m256 acc2 = _mm256_add_ps(ae2, ao2);
        const __m256 acc3 = _mm256_add_ps(ae3, ao3);
        // Amortised 4-distance hsum: two paired hadds collapse each accumulator's 8 lanes into
        // 4 partial sums, a third hadd interleaves the four pairs into one ymm whose lo/hi
        // halves are then summed cross-lane to four finished distances in xmm.
        const __m256 h01 = _mm256_hadd_ps(acc0, acc1);
        const __m256 h23 = _mm256_hadd_ps(acc2, acc3);
        const __m256 h = _mm256_hadd_ps(h01, h23);
        const __m128 lo = _mm256_castps256_ps128(h);
        const __m128 hi = _mm256_extractf128_ps(h, 1);
        __m128 dists = _mm_add_ps(lo, hi);
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
