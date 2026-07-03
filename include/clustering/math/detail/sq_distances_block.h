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

/**
 * @brief Fold `minSq[i] = min(minSq[i], ||points_i - row||^2)` over a contiguous AoS block.
 *
 * The seeder's per-round refresh sweeps are bound by the scalar load-compare-branch chain on
 * `minSq[i]`, so this kernel processes eight points per step with a branchless @c vminps and
 * an unconditional store. At `d == 2` the point pairs are deinterleaved in-register; at
 * `d >= 8` the distances come from @ref sqDistancesAosBlock through a stack staging buffer.
 * Remaining shapes keep the scalar walk.
 *
 * @param row    Contiguous @p d-element row every point is measured against.
 * @param points Base of @p count rows, each @p d contiguous elements.
 * @param minSq  Length-@p count running minima, updated in place.
 */
template <class T>
inline void refreshMinSqAgainstRow(const T *row, const T *points, std::size_t count, std::size_t d,
                                   T *minSq) noexcept {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                "refreshMinSqAgainstRow: T must be float or double.");
  std::size_t i = 0;
#ifdef CLUSTERING_USE_AVX2
  if constexpr (std::is_same_v<T, float>) {
    if (d == 2) {
      const __m256 cx = _mm256_set1_ps(row[0]);
      const __m256 cy = _mm256_set1_ps(row[1]);
      // shufps keeps its picks inside each 128-bit lane, so the distances land in
      // [p0 p1 p4 p5 | p2 p3 p6 p7] order; vpermps restores point order before the fold.
      const __m256i order = _mm256_setr_epi32(0, 1, 4, 5, 2, 3, 6, 7);
      for (; i + 8 <= count; i += 8) {
        const __m256 a = _mm256_loadu_ps(points + (2 * i));
        const __m256 b = _mm256_loadu_ps(points + (2 * i) + 8);
        const __m256 xs = _mm256_shuffle_ps(a, b, 0x88);
        const __m256 ys = _mm256_shuffle_ps(a, b, 0xDD);
        const __m256 dx = _mm256_sub_ps(xs, cx);
        const __m256 dy = _mm256_sub_ps(ys, cy);
        const __m256 lane = _mm256_fmadd_ps(dy, dy, _mm256_mul_ps(dx, dx));
        const __m256 dist = _mm256_permutevar8x32_ps(lane, order);
        const __m256 m = _mm256_loadu_ps(minSq + i);
        _mm256_storeu_ps(minSq + i, _mm256_min_ps(dist, m));
      }
    } else if (d >= kAvx2Lanes<T>) {
      constexpr std::size_t kStage = 64;
      alignas(32) std::array<float, kStage> stage;
      for (; i + 8 <= count; i += kStage) {
        const std::size_t blk = (count - i < kStage) ? (count - i) : kStage;
        sqDistancesAosBlock(row, points + (i * d), blk, d, stage.data());
        std::size_t j = 0;
        for (; j + 8 <= blk; j += 8) {
          const __m256 dist = _mm256_load_ps(stage.data() + j);
          const __m256 m = _mm256_loadu_ps(minSq + i + j);
          _mm256_storeu_ps(minSq + i + j, _mm256_min_ps(dist, m));
        }
        for (; j < blk; ++j) {
          const float cand = stage[j];
          if (cand < minSq[i + j]) {
            minSq[i + j] = cand;
          }
        }
        if (blk < kStage) {
          i += blk;
          break;
        }
      }
    }
  }
#endif
  for (; i < count; ++i) {
    const T cand = sqEuclideanRowPtr(points + (i * d), row, d);
    if (cand < minSq[i]) {
      minSq[i] = cand;
    }
  }
}

} // namespace clustering::math::detail
