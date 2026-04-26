#pragma once

#ifdef CLUSTERING_USE_AVX2

#include <immintrin.h>

#include <array>
#include <cstddef>

namespace clustering::math::detail {

/**
 * @brief Score @c L candidate rows against all @p n rows of @p xData.
 *
 * For each row @c i and each candidate @c t in `[0, L)`:
 *   `dist(i, t)` = ||x_i - cand_t||^2
 *   `outDist(i, t)` = dist(i, t)
 *   `scoresOut[t]` += min(dist(i, t), minSq[i])
 *
 * Rows are processed in 8-row M-tiles. Each tile reads an 8 * d-float window of @p xData in AoS
 * order and an 8x8 in-register transpose hoists each feature chunk into an 8-lane YMM holding
 * the 8 rows' feature at that position. The inner candidate loop broadcasts each candidate's
 * feature, subtracts, and FMAs into one YMM accumulator per candidate; after the K loop those
 * accumulators hold the 8 per-row distances in their 8 lanes so a final min-vs-minSq +
 * horizontal-sum folds them into @p scoresOut without any per-row reductions. A scatter writes
 * the per-(row, candidate) distances into @p outDist for the seeder's winner pickup.
 *
 * Tail rows past the last full 8-tile fall through to a scalar epilogue so the kernel handles
 * non-multiple-of-8 @p n.
 *
 * @tparam L Compile-time candidate batch; 1..6.
 * @param xData     (n, d) AoS row-major matrix of input points.
 * @param n         Row count of @p xData.
 * @param d         Feature dimension. Must be >= 1; no lane-width minimum.
 * @param candData  (L, d) row-major candidate rows.
 * @param minSq     Per-row running min-distance-squared, length @p n.
 * @param outDist   (n, outStride) per-row per-candidate distance output; only columns
 *                  `[0, L)` are written.
 * @param outStride Row stride of @p outDist; must be `>= L`.
 * @param scoresOut Per-candidate score accumulator, length @c L; accumulated into.
 */
template <std::size_t L>
[[gnu::always_inline]] inline void
kmppScoreSoaRowsAvx2F32(const float *xData, std::size_t n, std::size_t d, const float *candData,
                        const float *minSq, float *outDist, std::size_t outStride,
                        float *scoresOut) noexcept {
  static_assert(L >= 1 && L <= 6, "L must be in [1, 6]");

  const std::size_t nTiles = n / 8;

  std::array<__m256, L> scoreAcc;
  for (std::size_t t = 0; t < L; ++t) {
    scoreAcc[t] = _mm256_setzero_ps();
  }

  for (std::size_t tile = 0; tile < nTiles; ++tile) {
    const std::size_t iBase = tile * 8;
    const float *xTileBase = xData + (iBase * d);

    std::array<__m256, L> acc;
    for (std::size_t t = 0; t < L; ++t) {
      acc[t] = _mm256_setzero_ps();
    }

    std::size_t k = 0;
    for (; k + 8 <= d; k += 8) {
      const __m256 r0 = _mm256_loadu_ps(xTileBase + (0 * d) + k);
      const __m256 r1 = _mm256_loadu_ps(xTileBase + (1 * d) + k);
      const __m256 r2 = _mm256_loadu_ps(xTileBase + (2 * d) + k);
      const __m256 r3 = _mm256_loadu_ps(xTileBase + (3 * d) + k);
      const __m256 r4 = _mm256_loadu_ps(xTileBase + (4 * d) + k);
      const __m256 r5 = _mm256_loadu_ps(xTileBase + (5 * d) + k);
      const __m256 r6 = _mm256_loadu_ps(xTileBase + (6 * d) + k);
      const __m256 r7 = _mm256_loadu_ps(xTileBase + (7 * d) + k);

      const __m256 u0 = _mm256_unpacklo_ps(r0, r1);
      const __m256 u1 = _mm256_unpackhi_ps(r0, r1);
      const __m256 u2 = _mm256_unpacklo_ps(r2, r3);
      const __m256 u3 = _mm256_unpackhi_ps(r2, r3);
      const __m256 u4 = _mm256_unpacklo_ps(r4, r5);
      const __m256 u5 = _mm256_unpackhi_ps(r4, r5);
      const __m256 u6 = _mm256_unpacklo_ps(r6, r7);
      const __m256 u7 = _mm256_unpackhi_ps(r6, r7);

      const __m256 s0 = _mm256_shuffle_ps(u0, u2, _MM_SHUFFLE(1, 0, 1, 0));
      const __m256 s1 = _mm256_shuffle_ps(u0, u2, _MM_SHUFFLE(3, 2, 3, 2));
      const __m256 s2 = _mm256_shuffle_ps(u1, u3, _MM_SHUFFLE(1, 0, 1, 0));
      const __m256 s3 = _mm256_shuffle_ps(u1, u3, _MM_SHUFFLE(3, 2, 3, 2));
      const __m256 s4 = _mm256_shuffle_ps(u4, u6, _MM_SHUFFLE(1, 0, 1, 0));
      const __m256 s5 = _mm256_shuffle_ps(u4, u6, _MM_SHUFFLE(3, 2, 3, 2));
      const __m256 s6 = _mm256_shuffle_ps(u5, u7, _MM_SHUFFLE(1, 0, 1, 0));
      const __m256 s7 = _mm256_shuffle_ps(u5, u7, _MM_SHUFFLE(3, 2, 3, 2));

      const __m256 xK0 = _mm256_permute2f128_ps(s0, s4, 0x20);
      const __m256 xK1 = _mm256_permute2f128_ps(s1, s5, 0x20);
      const __m256 xK2 = _mm256_permute2f128_ps(s2, s6, 0x20);
      const __m256 xK3 = _mm256_permute2f128_ps(s3, s7, 0x20);
      const __m256 xK4 = _mm256_permute2f128_ps(s0, s4, 0x31);
      const __m256 xK5 = _mm256_permute2f128_ps(s1, s5, 0x31);
      const __m256 xK6 = _mm256_permute2f128_ps(s2, s6, 0x31);
      const __m256 xK7 = _mm256_permute2f128_ps(s3, s7, 0x31);

      for (std::size_t t = 0; t < L; ++t) {
        const float *cand = candData + (t * d) + k;
        // Two independent FMA chains per candidate: even features into @c ae (seeded with the
        // prior k_chunk's acc), odd features into @c ao (starts at zero, merged at k_chunk end).
        // Halves the critical-path dependency depth at the cost of one VADDPS per k_chunk; fits
        // at L <= 6 without spilling (8 xK + L acc + 2 chain temps + alias on c/diff).
        __m256 ae = acc[t];
        __m256 ao = _mm256_setzero_ps();
        __m256 c = _mm256_broadcast_ss(cand + 0);
        __m256 diff = _mm256_sub_ps(xK0, c);
        ae = _mm256_fmadd_ps(diff, diff, ae);
        c = _mm256_broadcast_ss(cand + 1);
        diff = _mm256_sub_ps(xK1, c);
        ao = _mm256_fmadd_ps(diff, diff, ao);
        c = _mm256_broadcast_ss(cand + 2);
        diff = _mm256_sub_ps(xK2, c);
        ae = _mm256_fmadd_ps(diff, diff, ae);
        c = _mm256_broadcast_ss(cand + 3);
        diff = _mm256_sub_ps(xK3, c);
        ao = _mm256_fmadd_ps(diff, diff, ao);
        c = _mm256_broadcast_ss(cand + 4);
        diff = _mm256_sub_ps(xK4, c);
        ae = _mm256_fmadd_ps(diff, diff, ae);
        c = _mm256_broadcast_ss(cand + 5);
        diff = _mm256_sub_ps(xK5, c);
        ao = _mm256_fmadd_ps(diff, diff, ao);
        c = _mm256_broadcast_ss(cand + 6);
        diff = _mm256_sub_ps(xK6, c);
        ae = _mm256_fmadd_ps(diff, diff, ae);
        c = _mm256_broadcast_ss(cand + 7);
        diff = _mm256_sub_ps(xK7, c);
        ao = _mm256_fmadd_ps(diff, diff, ao);
        acc[t] = _mm256_add_ps(ae, ao);
      }
    }

    // K tail: features past the last multiple of 8.
    if (k < d) {
      alignas(32) std::array<float, 8 * L> tailScalars{};
      for (std::size_t r = 0; r < 8; ++r) {
        const float *xr = xTileBase + (r * d);
        for (std::size_t t = 0; t < L; ++t) {
          const float *cand = candData + (t * d);
          float s = 0.0F;
          for (std::size_t kk = k; kk < d; ++kk) {
            const float diff = xr[kk] - cand[kk];
            s += diff * diff;
          }
          tailScalars[(t * 8) + r] = s;
        }
      }
      for (std::size_t t = 0; t < L; ++t) {
        const __m256 tailVec = _mm256_load_ps(tailScalars.data() + (t * 8));
        acc[t] = _mm256_add_ps(acc[t], tailVec);
      }
    }

    const __m256 vMin = _mm256_loadu_ps(minSq + iBase);
    alignas(32) std::array<float, L * 8> distScatter{};
    for (std::size_t t = 0; t < L; ++t) {
      const __m256 capped = _mm256_min_ps(acc[t], vMin);
      scoreAcc[t] = _mm256_add_ps(scoreAcc[t], capped);
      _mm256_store_ps(distScatter.data() + (t * 8), acc[t]);
    }
    for (std::size_t r = 0; r < 8; ++r) {
      float *dstRow = outDist + ((iBase + r) * outStride);
      for (std::size_t t = 0; t < L; ++t) {
        dstRow[t] = distScatter[(t * 8) + r];
      }
    }
  }

  for (std::size_t t = 0; t < L; ++t) {
    const __m256 perm = _mm256_permute2f128_ps(scoreAcc[t], scoreAcc[t], 1);
    const __m256 s1 = _mm256_add_ps(scoreAcc[t], perm);
    const __m256 s2 = _mm256_hadd_ps(s1, s1);
    const __m256 s3 = _mm256_hadd_ps(s2, s2);
    scoresOut[t] += _mm_cvtss_f32(_mm256_castps256_ps128(s3));
  }

  for (std::size_t i = nTiles * 8; i < n; ++i) {
    const float *xi = xData + (i * d);
    const float mi = minSq[i];
    for (std::size_t t = 0; t < L; ++t) {
      const float *cand = candData + (t * d);
      float s = 0.0F;
      for (std::size_t k = 0; k < d; ++k) {
        const float diff = xi[k] - cand[k];
        s += diff * diff;
      }
      outDist[(i * outStride) + t] = s;
      scoresOut[t] += (s < mi) ? s : mi;
    }
  }
}

} // namespace clustering::math::detail

#endif // CLUSTERING_USE_AVX2
