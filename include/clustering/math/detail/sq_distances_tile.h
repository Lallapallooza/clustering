#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>

#include "clustering/math/detail/avx2_helpers.h"
#include "clustering/math/detail/sq_distances_block.h"

#ifdef CLUSTERING_USE_AVX2
#include <immintrin.h>
#endif

namespace clustering::math::detail {

#ifdef CLUSTERING_USE_AVX2

/**
 * @brief 4-query x 2-centroid AVX2 tile kernel for AFK-MC2 chain-step distance scans.
 *
 * Computes `out[0..3]` = squared distance from each of four query rows to centroid `c0`
 * and `out[4..7]` = squared distance to centroid `c1`. Both centroids share the inner loop
 * so each centroid feature load amortises across four query subtractions, and each query
 * feature load amortises across two centroid subtractions; per 8-feature chunk the kernel
 * issues 4 query loads + 2 centroid loads + 8 subs + 8 fmadds = 22 ops, of which 16 are FP
 * pipe issues. The 8-accumulator hsum tree finishes the eight distances with two
 * 4-accumulator hadd trees plus a cross-lane add, amortising the reduction across the tile.
 *
 * Compared with calling @ref sqDistancesAosBlock once per query, this tile cuts the centroid
 * load traffic in half and finishes the eight distances with a single per-tile reduction.
 *
 * @param q0 First query row; @p d contiguous floats.
 * @param q1 Second query row.
 * @param q2 Third query row.
 * @param q3 Fourth query row.
 * @param c0 First centroid row; @p d contiguous floats.
 * @param c1 Second centroid row.
 * @param d  Per-row element count.
 * @param out Length-8 output: `[d(q0,c0), d(q1,c0), d(q2,c0), d(q3,c0), d(q0,c1), ..., d(q3,c1)]`.
 */
[[gnu::always_inline]] inline void sqDistTile4q2cAvx2F32(const float *q0, const float *q1,
                                                         const float *q2, const float *q3,
                                                         const float *c0, const float *c1,
                                                         std::size_t d, float *out) noexcept {
  __m256 a00 = _mm256_setzero_ps();
  __m256 a10 = _mm256_setzero_ps();
  __m256 a20 = _mm256_setzero_ps();
  __m256 a30 = _mm256_setzero_ps();
  __m256 a01 = _mm256_setzero_ps();
  __m256 a11 = _mm256_setzero_ps();
  __m256 a21 = _mm256_setzero_ps();
  __m256 a31 = _mm256_setzero_ps();

  std::size_t k = 0;
  for (; k + 8 <= d; k += 8) {
    const __m256 vc0 = _mm256_loadu_ps(c0 + k);
    const __m256 vc1 = _mm256_loadu_ps(c1 + k);
    {
      const __m256 vq = _mm256_loadu_ps(q0 + k);
      const __m256 e0 = _mm256_sub_ps(vq, vc0);
      a00 = _mm256_fmadd_ps(e0, e0, a00);
      const __m256 e1 = _mm256_sub_ps(vq, vc1);
      a01 = _mm256_fmadd_ps(e1, e1, a01);
    }
    {
      const __m256 vq = _mm256_loadu_ps(q1 + k);
      const __m256 e0 = _mm256_sub_ps(vq, vc0);
      a10 = _mm256_fmadd_ps(e0, e0, a10);
      const __m256 e1 = _mm256_sub_ps(vq, vc1);
      a11 = _mm256_fmadd_ps(e1, e1, a11);
    }
    {
      const __m256 vq = _mm256_loadu_ps(q2 + k);
      const __m256 e0 = _mm256_sub_ps(vq, vc0);
      a20 = _mm256_fmadd_ps(e0, e0, a20);
      const __m256 e1 = _mm256_sub_ps(vq, vc1);
      a21 = _mm256_fmadd_ps(e1, e1, a21);
    }
    {
      const __m256 vq = _mm256_loadu_ps(q3 + k);
      const __m256 e0 = _mm256_sub_ps(vq, vc0);
      a30 = _mm256_fmadd_ps(e0, e0, a30);
      const __m256 e1 = _mm256_sub_ps(vq, vc1);
      a31 = _mm256_fmadd_ps(e1, e1, a31);
    }
  }

  // 4-accumulator hsum tree for centroid c0: a00, a10, a20, a30 -> 4 distances in xmm.
  const __m256 h00 = _mm256_hadd_ps(a00, a10);
  const __m256 h10 = _mm256_hadd_ps(a20, a30);
  const __m256 h0 = _mm256_hadd_ps(h00, h10);
  __m128 d_c0 = _mm_add_ps(_mm256_castps256_ps128(h0), _mm256_extractf128_ps(h0, 1));

  // Same tree for c1.
  const __m256 h01 = _mm256_hadd_ps(a01, a11);
  const __m256 h11 = _mm256_hadd_ps(a21, a31);
  const __m256 h1 = _mm256_hadd_ps(h01, h11);
  __m128 d_c1 = _mm_add_ps(_mm256_castps256_ps128(h1), _mm256_extractf128_ps(h1, 1));

  if (k < d) {
    alignas(16) std::array<float, 4> tail0{0.0F, 0.0F, 0.0F, 0.0F};
    alignas(16) std::array<float, 4> tail1{0.0F, 0.0F, 0.0F, 0.0F};
    for (std::size_t j = k; j < d; ++j) {
      const float qj0 = q0[j];
      const float qj1 = q1[j];
      const float qj2 = q2[j];
      const float qj3 = q3[j];
      const float cj0 = c0[j];
      const float cj1 = c1[j];
      const float e00 = qj0 - cj0;
      tail0[0] += e00 * e00;
      const float e10 = qj1 - cj0;
      tail0[1] += e10 * e10;
      const float e20 = qj2 - cj0;
      tail0[2] += e20 * e20;
      const float e30 = qj3 - cj0;
      tail0[3] += e30 * e30;
      const float e01 = qj0 - cj1;
      tail1[0] += e01 * e01;
      const float e11 = qj1 - cj1;
      tail1[1] += e11 * e11;
      const float e21 = qj2 - cj1;
      tail1[2] += e21 * e21;
      const float e31 = qj3 - cj1;
      tail1[3] += e31 * e31;
    }
    d_c0 = _mm_add_ps(d_c0, _mm_load_ps(tail0.data()));
    d_c1 = _mm_add_ps(d_c1, _mm_load_ps(tail1.data()));
  }

  _mm_storeu_ps(out + 0, d_c0);
  _mm_storeu_ps(out + 4, d_c1);
}

/**
 * @brief 4-query x 1-centroid AVX2 tile kernel; tail variant of @ref sqDistTile4q2cAvx2F32.
 *
 * Used when the centroid count is odd: the last centroid is processed with four query rows
 * but only one centroid in registers, freeing the second centroid lane to run a longer FMA
 * chain (two chains per query) that drives the per-distance critical path back below the FP
 * throughput ceiling.
 */
[[gnu::always_inline]] inline void sqDistTile4q1cAvx2F32(const float *q0, const float *q1,
                                                         const float *q2, const float *q3,
                                                         const float *c0, std::size_t d,
                                                         float *out) noexcept {
  __m256 ae0 = _mm256_setzero_ps();
  __m256 ae1 = _mm256_setzero_ps();
  __m256 ae2 = _mm256_setzero_ps();
  __m256 ae3 = _mm256_setzero_ps();
  __m256 ao0 = _mm256_setzero_ps();
  __m256 ao1 = _mm256_setzero_ps();
  __m256 ao2 = _mm256_setzero_ps();
  __m256 ao3 = _mm256_setzero_ps();

  std::size_t k = 0;
  for (; k + 16 <= d; k += 16) {
    const __m256 vc0e = _mm256_loadu_ps(c0 + k);
    {
      const __m256 e = _mm256_sub_ps(_mm256_loadu_ps(q0 + k), vc0e);
      ae0 = _mm256_fmadd_ps(e, e, ae0);
    }
    {
      const __m256 e = _mm256_sub_ps(_mm256_loadu_ps(q1 + k), vc0e);
      ae1 = _mm256_fmadd_ps(e, e, ae1);
    }
    {
      const __m256 e = _mm256_sub_ps(_mm256_loadu_ps(q2 + k), vc0e);
      ae2 = _mm256_fmadd_ps(e, e, ae2);
    }
    {
      const __m256 e = _mm256_sub_ps(_mm256_loadu_ps(q3 + k), vc0e);
      ae3 = _mm256_fmadd_ps(e, e, ae3);
    }
    const __m256 vc0o = _mm256_loadu_ps(c0 + k + 8);
    {
      const __m256 e = _mm256_sub_ps(_mm256_loadu_ps(q0 + k + 8), vc0o);
      ao0 = _mm256_fmadd_ps(e, e, ao0);
    }
    {
      const __m256 e = _mm256_sub_ps(_mm256_loadu_ps(q1 + k + 8), vc0o);
      ao1 = _mm256_fmadd_ps(e, e, ao1);
    }
    {
      const __m256 e = _mm256_sub_ps(_mm256_loadu_ps(q2 + k + 8), vc0o);
      ao2 = _mm256_fmadd_ps(e, e, ao2);
    }
    {
      const __m256 e = _mm256_sub_ps(_mm256_loadu_ps(q3 + k + 8), vc0o);
      ao3 = _mm256_fmadd_ps(e, e, ao3);
    }
  }
  if (k + 8 <= d) {
    const __m256 vc = _mm256_loadu_ps(c0 + k);
    {
      const __m256 e = _mm256_sub_ps(_mm256_loadu_ps(q0 + k), vc);
      ae0 = _mm256_fmadd_ps(e, e, ae0);
    }
    {
      const __m256 e = _mm256_sub_ps(_mm256_loadu_ps(q1 + k), vc);
      ae1 = _mm256_fmadd_ps(e, e, ae1);
    }
    {
      const __m256 e = _mm256_sub_ps(_mm256_loadu_ps(q2 + k), vc);
      ae2 = _mm256_fmadd_ps(e, e, ae2);
    }
    {
      const __m256 e = _mm256_sub_ps(_mm256_loadu_ps(q3 + k), vc);
      ae3 = _mm256_fmadd_ps(e, e, ae3);
    }
    k += 8;
  }
  const __m256 acc0 = _mm256_add_ps(ae0, ao0);
  const __m256 acc1 = _mm256_add_ps(ae1, ao1);
  const __m256 acc2 = _mm256_add_ps(ae2, ao2);
  const __m256 acc3 = _mm256_add_ps(ae3, ao3);
  const __m256 h01 = _mm256_hadd_ps(acc0, acc1);
  const __m256 h23 = _mm256_hadd_ps(acc2, acc3);
  const __m256 h = _mm256_hadd_ps(h01, h23);
  __m128 dists = _mm_add_ps(_mm256_castps256_ps128(h), _mm256_extractf128_ps(h, 1));
  if (k < d) {
    alignas(16) std::array<float, 4> tail{0.0F, 0.0F, 0.0F, 0.0F};
    for (std::size_t j = k; j < d; ++j) {
      const float cj = c0[j];
      const float e0 = q0[j] - cj;
      tail[0] += e0 * e0;
      const float e1 = q1[j] - cj;
      tail[1] += e1 * e1;
      const float e2 = q2[j] - cj;
      tail[2] += e2 * e2;
      const float e3 = q3[j] - cj;
      tail[3] += e3 * e3;
    }
    dists = _mm_add_ps(dists, _mm_load_ps(tail.data()));
  }
  _mm_storeu_ps(out, dists);
}

/**
 * @brief Min-distance scan from a batch of indexed query rows to a contiguous centroid block.
 *
 * For each `t` in `[0, qCount)`, computes `out[t] = min over j in [0, cCount)` of
 * `dist^2(xData[queryIdx[t]*d : (queryIdx[t]+1)*d], centroids[j*d : (j+1)*d])`. Outer loop
 * walks the centroid block once per 4-query tile, so each centroid row is loaded `qCount/4`
 * times instead of `qCount` times -- a `>= 4x` cut in centroid-side load traffic versus a
 * naive per-query loop, plus the per-tile hsum amortisation from @ref sqDistTile4q2cAvx2F32.
 *
 * @param xData    `n x d` AoS row-major input.
 * @param d        Per-row element count.
 * @param queryIdx Length-@p qCount indices into `[0, n)` selecting the query rows.
 * @param qCount   Number of queries.
 * @param centroids  `cCount x d` AoS row-major centroid block.
 * @param cCount   Number of centroid rows.
 * @param out      Length-@p qCount output; populated with the per-query min squared distance.
 */
[[gnu::always_inline]] inline void minDistBatchedAvx2F32(const float *xData, std::size_t d,
                                                         const std::size_t *queryIdx,
                                                         std::size_t qCount, const float *centroids,
                                                         std::size_t cCount, float *out) noexcept {
  constexpr float kInf = std::numeric_limits<float>::infinity();
  const __m128 vInf = _mm_set1_ps(kInf);

  std::size_t t = 0;
  for (; t + 4 <= qCount; t += 4) {
    const float *q0 = xData + (queryIdx[t + 0] * d);
    const float *q1 = xData + (queryIdx[t + 1] * d);
    const float *q2 = xData + (queryIdx[t + 2] * d);
    const float *q3 = xData + (queryIdx[t + 3] * d);
    __m128 best = vInf;
    alignas(16) std::array<float, 8> tile{};
    std::size_t j = 0;
    for (; j + 2 <= cCount; j += 2) {
      sqDistTile4q2cAvx2F32(q0, q1, q2, q3, centroids + (j * d), centroids + ((j + 1) * d), d,
                            tile.data());
      const __m128 dc0 = _mm_load_ps(tile.data() + 0);
      const __m128 dc1 = _mm_load_ps(tile.data() + 4);
      best = _mm_min_ps(best, dc0);
      best = _mm_min_ps(best, dc1);
    }
    if (j < cCount) {
      alignas(16) std::array<float, 4> tail{};
      sqDistTile4q1cAvx2F32(q0, q1, q2, q3, centroids + (j * d), d, tail.data());
      best = _mm_min_ps(best, _mm_load_ps(tail.data()));
    }
    _mm_storeu_ps(out + t, best);
  }
  // q-tail (0..3 queries): per-query block-of-4 against the centroids.
  for (; t < qCount; ++t) {
    const float *qrow = xData + (queryIdx[t] * d);
    alignas(16) std::array<float, 4> blockOut{};
    float minVal = kInf;
    std::size_t j = 0;
    for (; j + 4 <= cCount; j += 4) {
      sqDistancesAosBlock<float>(qrow, centroids + (j * d), 4, d, blockOut.data());
      const __m128 v = _mm_load_ps(blockOut.data());
      const __m128 perm0 = _mm_movehl_ps(v, v);
      const __m128 m0 = _mm_min_ps(v, perm0);
      const __m128 perm1 = _mm_shuffle_ps(m0, m0, 0x55);
      const __m128 m1 = _mm_min_ss(m0, perm1);
      const float blockMin = _mm_cvtss_f32(m1);
      minVal = std::min(blockMin, minVal);
    }
    for (; j < cCount; ++j) {
      const float dsq = sqEuclideanRowPtr(qrow, centroids + (j * d), d);
      minVal = std::min(dsq, minVal);
    }
    out[t] = minVal;
  }
}

#endif // CLUSTERING_USE_AVX2

} // namespace clustering::math::detail
