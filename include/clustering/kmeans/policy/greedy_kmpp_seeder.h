#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <vector>

#include "clustering/always_assert.h"
#include "clustering/math/detail/avx2_helpers.h"
#include "clustering/math/detail/gemm_outer.h"
#include "clustering/math/detail/matrix_desc.h"
#include "clustering/math/pairwise.h"
#include "clustering/math/rng.h"
#include "clustering/math/thread.h"
#include "clustering/ndarray.h"

#ifdef CLUSTERING_USE_AVX2
#include <immintrin.h>

#include "clustering/math/detail/kmpp_score_avx2.h"
#endif

namespace clustering::kmeans {

namespace detail {

using math::detail::sqEuclideanRowPtr;

/**
 * @brief Compute the local-trials count used by greedy k-means++.
 *
 * Matches sklearn's convention @c 2 + floor(ln(k)). Gives k=8 -> L=4, k=64 -> L=6, k=256 ->
 * L=7, k=1000 -> L=8. The natural-log form keeps inertia within the per-pick scoring envelope
 * while trimming ~30% off the seeder's candidate work at high @c k.
 *
 * @return Local-trials count; always at least 1.
 */
[[nodiscard]] inline std::size_t greedyKmppLocalTrials(std::size_t k) noexcept {
  if (k <= 1) {
    return 1;
  }
  const auto lnK = std::log(static_cast<double>(k));
  return 2 + static_cast<std::size_t>(lnK);
}

/**
 * @brief Round @p L up to the nearest multiple of 8 used by the transposed scoring layout.
 *
 * The transposed kernel operates on chunks of 8 candidates; the candidate pack and the
 * per-(point, candidate) distance cache are padded to this width so the chunked scoring path
 * can index with a fixed row stride. The commit-step minSq refresh only reads @p L lanes.
 */
[[nodiscard]] constexpr std::size_t greedyKmppTransposedWidth(std::size_t L) noexcept {
  constexpr std::size_t kChunk = 8;
  return ((L + kChunk - 1) / kChunk) * kChunk;
}

#ifdef CLUSTERING_USE_AVX2

/**
 * @brief Compile-time batched scoring kernel: stream @p x once across @c B parallel AVX2
 *        accumulators to compute @c B squared distances against @p candData rows.
 *
 * Templating on @c B unblocks the compiler's full unroll of the inner candidate loop, the
 * load-bearing optimisation for the seeder's candidate scoring. The runtime entry
 * @ref sqEuclideanRowToBatchAvx2 dispatches to the @c B=8 specialisation for the common
 * @c L=8 case (k in [16, 31]) and to other compile-time @c B values for adjacent batches.
 */
template <std::size_t B>
[[gnu::always_inline]] inline void
sqEuclideanRowToBatchAvx2Fixed(const float *x, const float *candData, std::size_t d,
                               float *out) noexcept {
  static_assert(B >= 1 && B <= 8, "B must lie in [1, 8] -- 8 ymm regs hold the batch");
  // Double accumulator set (2 * B YMMs) over a 2x-unrolled K loop. Halves the per-iter fmadd
  // dependency chain so Zen5's 4-FMA-per-cycle throughput isn't latency-bound on the 4-cycle
  // fmadd round-trip; also gives the register allocator enough explicit live ranges to keep
  // accumulators in YMM registers rather than spilling to the stack (measured: 8 GFLOPS with
  // the original 1x loop, ~2x post-unroll on the seeder's B=4 hot path).
  std::array<__m256, B> acc0{};
  std::array<__m256, B> acc1{};
  for (std::size_t t = 0; t < B; ++t) {
    acc0[t] = _mm256_setzero_ps();
    acc1[t] = _mm256_setzero_ps();
  }
  std::size_t k = 0;
  for (; k + 16 <= d; k += 16) {
    const __m256 vx0 = _mm256_loadu_ps(x + k);
    const __m256 vx1 = _mm256_loadu_ps(x + k + 8);
    for (std::size_t t = 0; t < B; ++t) {
      const __m256 vc0 = _mm256_loadu_ps(candData + (t * d) + k);
      const __m256 vc1 = _mm256_loadu_ps(candData + (t * d) + k + 8);
      const __m256 diff0 = _mm256_sub_ps(vx0, vc0);
      const __m256 diff1 = _mm256_sub_ps(vx1, vc1);
      acc0[t] = _mm256_fmadd_ps(diff0, diff0, acc0[t]);
      acc1[t] = _mm256_fmadd_ps(diff1, diff1, acc1[t]);
    }
  }
  // 8-lane tail.
  for (; k + 8 <= d; k += 8) {
    const __m256 vx = _mm256_loadu_ps(x + k);
    for (std::size_t t = 0; t < B; ++t) {
      const __m256 vc = _mm256_loadu_ps(candData + (t * d) + k);
      const __m256 diff = _mm256_sub_ps(vx, vc);
      acc0[t] = _mm256_fmadd_ps(diff, diff, acc0[t]);
    }
  }
  std::array<float, B> tail{};
  for (std::size_t t = 0; t < B; ++t) {
    tail[t] = 0.0F;
  }
  for (std::size_t kt = k; kt < d; ++kt) {
    const float xk = x[kt];
    for (std::size_t t = 0; t < B; ++t) {
      const float diff = xk - candData[(t * d) + kt];
      tail[t] += diff * diff;
    }
  }
  for (std::size_t t = 0; t < B; ++t) {
    const __m256 sum = _mm256_add_ps(acc0[t], acc1[t]);
    out[t] = math::detail::horizontalSumAvx2(sum) + tail[t];
  }
}

template <std::size_t B>
inline void sqEuclideanRowToBatchAvx2Fixed(const double *x, const double *candData, std::size_t d,
                                           double *out) noexcept {
  static_assert(B >= 1 && B <= 8, "B must lie in [1, 8] -- 8 ymm regs hold the batch");
  std::array<__m256d, B> acc{};
  for (std::size_t t = 0; t < B; ++t) {
    acc[t] = _mm256_setzero_pd();
  }
  std::size_t k = 0;
  for (; k + 4 <= d; k += 4) {
    const __m256d vx = _mm256_loadu_pd(x + k);
    for (std::size_t t = 0; t < B; ++t) {
      const __m256d vc = _mm256_loadu_pd(candData + (t * d) + k);
      const __m256d diff = _mm256_sub_pd(vx, vc);
      acc[t] = _mm256_fmadd_pd(diff, diff, acc[t]);
    }
  }
  std::array<double, B> tail{};
  for (std::size_t t = 0; t < B; ++t) {
    tail[t] = 0.0;
  }
  for (std::size_t kt = k; kt < d; ++kt) {
    const double xk = x[kt];
    for (std::size_t t = 0; t < B; ++t) {
      const double diff = xk - candData[(t * d) + kt];
      tail[t] += diff * diff;
    }
  }
  for (std::size_t t = 0; t < B; ++t) {
    out[t] = math::detail::horizontalSumAvx2(acc[t]) + tail[t];
  }
}

/**
 * @brief Compute @p L squared Euclidean distances against an @c (L, d) row-batched candidate
 *        layout in a single streaming pass over the @p x row.
 *
 * Streams the @p x row through L parallel @c fmadd accumulators so each x byte is read from
 * memory once for all L distances. Dispatches to @ref sqEuclideanRowToBatchAvx2Fixed at
 * compile-time-bound @c B for the common batch sizes.
 */
template <class T>
inline void sqEuclideanRowToBatchAvx2(const T *x, const T *candData, std::size_t L, std::size_t d,
                                      T *out) noexcept {
  std::size_t base = 0;
  while (base + 8 <= L) {
    sqEuclideanRowToBatchAvx2Fixed<8>(x, candData + (base * d), d, out + base);
    base += 8;
  }
  switch (L - base) {
  case 0:
    break;
  case 1:
    sqEuclideanRowToBatchAvx2Fixed<1>(x, candData + (base * d), d, out + base);
    break;
  case 2:
    sqEuclideanRowToBatchAvx2Fixed<2>(x, candData + (base * d), d, out + base);
    break;
  case 3:
    sqEuclideanRowToBatchAvx2Fixed<3>(x, candData + (base * d), d, out + base);
    break;
  case 4:
    sqEuclideanRowToBatchAvx2Fixed<4>(x, candData + (base * d), d, out + base);
    break;
  case 5:
    sqEuclideanRowToBatchAvx2Fixed<5>(x, candData + (base * d), d, out + base);
    break;
  case 6:
    sqEuclideanRowToBatchAvx2Fixed<6>(x, candData + (base * d), d, out + base);
    break;
  case 7:
    sqEuclideanRowToBatchAvx2Fixed<7>(x, candData + (base * d), d, out + base);
    break;
  default:
    break;
  }
}

/**
 * @brief Compute @p L squared distances against an @c (d, 8) transposed candidate layout with
 *        one streaming pass over the @p x row.
 *
 * The (d, 8) layout puts the k-th feature of every candidate at @c cand[k*8 .. k*8 + 8), so a
 * single @c _mm256_load_ps fetches all 8 candidates' k-th component; broadcasting @c x[k] then
 * folds 8 squared-distance contributions in one FMA. At @c d < 8 this collapses the per-row
 * scoring from @c L*d scalar ops to @c d SIMD ops.
 */
inline void sqEuclideanRowAgainst8Transposed(const float *x, const float *candData, std::size_t d,
                                             float *out) noexcept {
  __m256 acc = _mm256_setzero_ps();
  for (std::size_t k = 0; k < d; ++k) {
    const __m256 cv = _mm256_load_ps(candData + (k * 8));
    const __m256 xv = _mm256_set1_ps(x[k]);
    const __m256 diff = _mm256_sub_ps(xv, cv);
    acc = _mm256_fmadd_ps(diff, diff, acc);
  }
  _mm256_storeu_ps(out, acc);
}

/**
 * @brief Compute two 8-way squared distance slabs against an @c (d, 16) transposed candidate
 *        layout in one streaming pass over the @p x row.
 *
 * Unrolls @ref sqEuclideanRowAgainst8Transposed across two adjacent lane groups so each @c x[k]
 * broadcast folds 16 candidate distances per FMA pair. At @c L in @c (8, 16] with @c d <= 8
 * this shaves half the broadcast + load traffic versus looping the 8-wide kernel twice.
 */
inline void sqEuclideanRowAgainst16Transposed(const float *x, const float *candData, std::size_t d,
                                              float *out) noexcept {
  __m256 accLo = _mm256_setzero_ps();
  __m256 accHi = _mm256_setzero_ps();
  for (std::size_t k = 0; k < d; ++k) {
    const __m256 cLo = _mm256_load_ps(candData + (k * 16));
    const __m256 cHi = _mm256_load_ps(candData + (k * 16) + 8);
    const __m256 xv = _mm256_set1_ps(x[k]);
    const __m256 diffLo = _mm256_sub_ps(xv, cLo);
    const __m256 diffHi = _mm256_sub_ps(xv, cHi);
    accLo = _mm256_fmadd_ps(diffLo, diffLo, accLo);
    accHi = _mm256_fmadd_ps(diffHi, diffHi, accHi);
  }
  _mm256_storeu_ps(out, accLo);
  _mm256_storeu_ps(out + 8, accHi);
}

/**
 * @brief Compute one 8-way squared distance slab against an @c (d, W) transposed candidate
 *        layout with an explicit row stride @p W.
 *
 * Generalizes @ref sqEuclideanRowAgainst8Transposed to the L > 16 regime where the transposed
 * pack keeps @c W = ceil(L/8) * 8 columns so chunked scoring can slide an 8-wide window across
 * it.
 */
inline void sqEuclideanRowAgainst8TransposedStrided(const float *x, const float *candData,
                                                    std::size_t d, std::size_t rowStride,
                                                    float *out) noexcept {
  __m256 acc = _mm256_setzero_ps();
  for (std::size_t k = 0; k < d; ++k) {
    const __m256 cv = _mm256_loadu_ps(candData + (k * rowStride));
    const __m256 xv = _mm256_set1_ps(x[k]);
    const __m256 diff = _mm256_sub_ps(xv, cv);
    acc = _mm256_fmadd_ps(diff, diff, acc);
  }
  _mm256_storeu_ps(out, acc);
}

#endif // CLUSTERING_USE_AVX2

/**
 * @brief Squared Euclidean distance from one @p x row to a batch of @p L candidate rows.
 *
 * Routes to the AVX2 batched kernel when the build is AVX2-enabled and @p d clears one lane
 * width; otherwise dispatches @p L scalar @ref sqEuclideanRowPtr calls. The batched AVX2 path
 * streams the @p x row through @p L parallel accumulators so the inner loop reads each x byte
 * once across the whole batch.
 */
template <class T>
inline void sqEuclideanRowToBatch(const T *x, const T *candData, std::size_t L, std::size_t d,
                                  T *out) noexcept {
#ifdef CLUSTERING_USE_AVX2
  if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
    if (d >= math::detail::kAvx2Lanes<T>) {
      sqEuclideanRowToBatchAvx2(x, candData, L, d, out);
      return;
    }
  }
#endif
  for (std::size_t t = 0; t < L; ++t) {
    out[t] = sqEuclideanRowPtr(x, candData + (t * d), d);
  }
}

} // namespace detail

/**
 * @brief Greedy k-means++ seeder.
 *
 * Picks @c k initial centroid rows from the dataset. The first centroid is drawn uniformly;
 * each subsequent centroid is the best of @c L = 2 + floor(ln(k)) candidates sampled with
 * probability proportional to @c D(x)^2 -- the squared distance from each point to its nearest
 * already-chosen centroid. The candidate that yields the smallest resulting sum of squared
 * minimum distances wins.
 *
 * Scratch is private: the candidate pack, the transposed candidate layout, the per-point
 * per-candidate distance cache, the cumulative-distance array, and the per-point running
 * min-squared-distance all live inside the policy. Repeated @c run calls at a stable
 * @c (n, d, k) shape pay no reallocation.
 *
 * @tparam T Element type; @c float or @c double.
 */
template <class T> class GreedyKmppSeeder {
public:
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                "GreedyKmppSeeder<T> requires T to be float or double");

  GreedyKmppSeeder()
      : m_candRows({0, 0}), m_candRowsT({0, 0}), m_candDistSq({0, 0}), m_cumDistSq({0}),
        m_minSq({0}), m_distsFlat({0, 0}), m_xNormsSq({0}), m_candNormsSq({0}), m_gemmApArena({0}),
        m_gemmBpArena({0}), m_localScores({0}) {}

  /**
   * @brief Seed @c k centroids from @p X into @p outCentroids.
   *
   * @param X            Data matrix (n x d), contiguous.
   * @param k            Number of centroids to seed (@c >= 1).
   * @param seed         RNG seed; identical seed + @c (X, k) produces identical centroids.
   * @param pool         Parallelism injection. Reserved for a future per-chunk fan-out of the
   *                     scoring loop.
   * @param outCentroids Output centroid matrix (k x d), contiguous; populated in row order.
   */
  void run(const NDArray<T, 2, Layout::Contig> &X, std::size_t k, std::uint64_t seed,
           math::Pool pool, NDArray<T, 2, Layout::Contig> &outCentroids) {
    const std::size_t n = X.dim(0);
    const std::size_t d = X.dim(1);

    CLUSTERING_ALWAYS_ASSERT(outCentroids.isMutable());
    CLUSTERING_ALWAYS_ASSERT(outCentroids.dim(0) == k);
    CLUSTERING_ALWAYS_ASSERT(outCentroids.dim(1) == d);
    CLUSTERING_ALWAYS_ASSERT(k >= 1);
    CLUSTERING_ALWAYS_ASSERT(n >= k);

    (void)pool;

    const std::size_t nLocalTrials = detail::greedyKmppLocalTrials(k);
    ensureShape(n, d, nLocalTrials, pool.workerCount());

    math::pcg64 rng;
    rng.seed(seed);

    const T *xData = X.data();
    T *centroidsData = outCentroids.data();
    T *minSq = m_minSq.data();
    T *candRowsData = m_candRows.data();
    T *cumDistSq = m_cumDistSq.data();
    T *candDistSqData = m_candDistSq.data();
#ifdef CLUSTERING_USE_AVX2
    T *candRowsTData = m_candRowsT.data();
#endif

    // GEMM scoring wins only when the candidate width L is >= one kNr panel (6). Below that
    // the 8x6 kernel's fixed 48-FMA body over-computes the 8xL useful tile; the per-row
    // streaming kernel with L parallel accumulators is tighter. Gate on L >= kNr<float>.
    constexpr std::size_t kNrF = math::detail::kKernelNr<float>;
    const bool useGemmScoring = (d >= 32) && (nLocalTrials >= kNrF);
    if (useGemmScoring) {
      T *xNormsData = m_xNormsSq.data();
      for (std::size_t i = 0; i < n; ++i) {
        xNormsData[i] = math::detail::sqNormRow<T, Layout::Contig>(X, i);
      }
    }

    // Step 1: first centroid uniformly. randUniformU64 is the deterministic primitive; the
    // modulo map carries a tiny bias for very large n but is the standard sklearn convention.
    const auto first = static_cast<std::size_t>(math::randUniformU64(rng) % n);
    std::memcpy(centroidsData, xData + (first * d), d * sizeof(T));

    for (std::size_t i = 0; i < n; ++i) {
      minSq[i] = detail::sqEuclideanRowPtr(xData + (i * d), centroidsData, d);
    }

    if (k == 1) {
      return;
    }

    std::vector<std::size_t> candidates(nLocalTrials, 0);
    std::vector<T> scores(nLocalTrials, T{0});

    for (std::size_t c = 1; c < k; ++c) {
      // Build the cumulative-distance array in a single pass. The cum array is only used as a
      // probability normalizer, so no Kahan compensation.
      T runningSum = T{0};
      for (std::size_t i = 0; i < n; ++i) {
        runningSum += minSq[i];
        cumDistSq[i] = runningSum;
      }
      const T total = runningSum;

      // Degenerate guard: when every chosen centroid coincides with every remaining point the
      // total collapses to ~0; pick the next centroid uniformly so the routine cannot stall.
      if (!(total > T{0})) {
        const auto pick = static_cast<std::size_t>(math::randUniformU64(rng) % n);
        std::memcpy(centroidsData + (c * d), xData + (pick * d), d * sizeof(T));
        for (std::size_t i = 0; i < n; ++i) {
          const T cand = detail::sqEuclideanRowPtr(xData + (i * d), centroidsData + (c * d), d);
          if (cand < minSq[i]) {
            minSq[i] = cand;
          }
        }
        continue;
      }

      // Draw nLocalTrials candidates by inverse-CDF sampling on the cumulative array via
      // std::upper_bound. Determinism: identical seed + identical n produces identical candidate
      // sets because the @c randUnit draw sequence is the same and the cum array is identical.
      const T *cumBegin = cumDistSq;
      const T *cumEnd = cumDistSq + n;
      for (std::size_t t = 0; t < nLocalTrials; ++t) {
        const T u = math::randUnit<T>(rng) * total;
        const T *it = std::upper_bound(cumBegin, cumEnd, u);
        const std::size_t pick = (it == cumEnd) ? (n - 1) : static_cast<std::size_t>(it - cumBegin);
        candidates[t] = pick;
      }

      // Pack the L candidate rows into a contiguous (L, d) buffer so the batched scoring kernel
      // can stream x once across L accumulators. The L*d pack is negligible against the n-pass
      // scoring it amortizes.
      for (std::size_t t = 0; t < nLocalTrials; ++t) {
        std::memcpy(candRowsData + (t * d), xData + (candidates[t] * d), d * sizeof(T));
      }

      for (std::size_t t = 0; t < nLocalTrials; ++t) {
        scores[t] = T{0};
      }
      constexpr std::size_t kMaxLocalTrials = 32;
      CLUSTERING_ALWAYS_ASSERT(nLocalTrials <= kMaxLocalTrials);

      const std::size_t transposedWidth = detail::greedyKmppTransposedWidth(nLocalTrials);
      bool scoredViaTransposed = false;
#ifdef CLUSTERING_USE_AVX2
      // Low-d hot path: at d <= kAvx2Lanes the (L, d) row-batched kernel either falls into the
      // scalar K-tail (d < 8) or pays @c L horizontal-sum reductions for one K-iter of work
      // (d == 8). The transposed @c (d, W) layout puts the same-feature components of every
      // candidate in consecutive 8-lane YMM registers, so each broadcast-of-x[k] + FMA pair
      // folds 8 (or 16, for the 16-lane unroll) distances at once.
      if constexpr (std::is_same_v<T, float>) {
        if (d > 0 && d <= math::detail::kAvx2Lanes<float>) {
          for (std::size_t kk = 0; kk < d; ++kk) {
            float *dstK = candRowsTData + (kk * transposedWidth);
            for (std::size_t t = 0; t < nLocalTrials; ++t) {
              dstK[t] = candRowsData[(t * d) + kk];
            }
            for (std::size_t t = nLocalTrials; t < transposedWidth; ++t) {
              dstK[t] = 0.0F;
            }
          }
          if (transposedWidth == 16) {
            __m256 scoresLoAcc = _mm256_setzero_ps();
            __m256 scoresHiAcc = _mm256_setzero_ps();
            for (std::size_t i = 0; i < n; ++i) {
              const float *xi = xData + (i * d);
              const __m256 miVec = _mm256_set1_ps(minSq[i]);
              float *dstRow = candDistSqData + (i * transposedWidth);
              detail::sqEuclideanRowAgainst16Transposed(xi, candRowsTData, d, dstRow);
              const __m256 dLo = _mm256_loadu_ps(dstRow);
              const __m256 dHi = _mm256_loadu_ps(dstRow + 8);
              scoresLoAcc = _mm256_add_ps(scoresLoAcc, _mm256_min_ps(dLo, miVec));
              scoresHiAcc = _mm256_add_ps(scoresHiAcc, _mm256_min_ps(dHi, miVec));
            }
            std::array<float, 16> tmp{};
            _mm256_storeu_ps(tmp.data(), scoresLoAcc);
            _mm256_storeu_ps(tmp.data() + 8, scoresHiAcc);
            for (std::size_t t = 0; t < nLocalTrials; ++t) {
              scores[t] = tmp[t];
            }
          } else if (transposedWidth == 8) {
            __m256 scoresAcc = _mm256_setzero_ps();
            for (std::size_t i = 0; i < n; ++i) {
              const float *xi = xData + (i * d);
              const __m256 miVec = _mm256_set1_ps(minSq[i]);
              float *dstRow = candDistSqData + (i * transposedWidth);
              detail::sqEuclideanRowAgainst8Transposed(xi, candRowsTData, d, dstRow);
              const __m256 dv = _mm256_loadu_ps(dstRow);
              scoresAcc = _mm256_add_ps(scoresAcc, _mm256_min_ps(dv, miVec));
            }
            std::array<float, 8> tmp{};
            _mm256_storeu_ps(tmp.data(), scoresAcc);
            for (std::size_t t = 0; t < nLocalTrials; ++t) {
              scores[t] = tmp[t];
            }
          } else {
            // Generic chunked path for L > 16 (very high k). Walk the transposed layout 8 lanes
            // at a time so each chunk stays on the fully unrolled 8-wide kernel.
            for (std::size_t i = 0; i < n; ++i) {
              const float *xi = xData + (i * d);
              const float mi = minSq[i];
              float *dstRow = candDistSqData + (i * transposedWidth);
              for (std::size_t base = 0; base < transposedWidth; base += 8) {
                detail::sqEuclideanRowAgainst8TransposedStrided(xi, candRowsTData + base, d,
                                                                transposedWidth, dstRow + base);
              }
              for (std::size_t t = 0; t < nLocalTrials; ++t) {
                scores[t] += (dstRow[t] < mi) ? dstRow[t] : mi;
              }
            }
          }
          scoredViaTransposed = true;
        }
      }
#endif

      if (!scoredViaTransposed) {
        // GEMM-based batch distance for moderate-to-high d: compute X * cand^T via the core
        // GEMM (alpha=-2, beta=0), then add pre-computed per-row ||x||^2 and per-candidate
        // ||c||^2 in one min+sum fold. BLAS-style GEMM is the decisive win at d >= ~16 where
        // the per-row streaming kernel bottlenecks on L1/L2 bandwidth.
        if (useGemmScoring) {
          auto candView = NDArray<T, 2, Layout::Contig>::borrow(candRowsData, {nLocalTrials, d});
          auto xView = NDArray<T, 2, Layout::Contig>::borrow(const_cast<T *>(xData), {n, d});
          auto distsView = NDArray<T, 2>::borrow(m_distsFlat.data(), {n, nLocalTrials});
          auto candT = candView.t();
          // Direct gemmRunReference with caller-owned scratch so the seeder's per-pick GEMM
          // leaves the shape-stable allocation footprint in place (no per-call arena alloc).
          const auto xDesc = ::clustering::detail::describeMatrix(xView);
          const auto candDesc = ::clustering::detail::describeMatrix(candT);
          auto distsDesc = ::clustering::detail::describeMatrixMut(distsView);
          math::detail::gemmRunReference<T>(xDesc, candDesc, distsDesc, T{-2}, T{0},
                                            m_gemmApArena.data(), m_gemmBpArena.data(), pool);
          // Candidate norms once per pick.
          T *candNorms = m_candNormsSq.data();
          for (std::size_t t = 0; t < nLocalTrials; ++t) {
            candNorms[t] = math::detail::sqNormRow<T, Layout::Contig>(candView, t);
          }
          const T *xNorms = m_xNormsSq.data();
          const T *distsFlat = m_distsFlat.data();
          for (std::size_t i = 0; i < n; ++i) {
            const T mi = minSq[i];
            const T xn = xNorms[i];
            const T *distRowI = distsFlat + (i * nLocalTrials);
            T *dstRow = candDistSqData + (i * transposedWidth);
            for (std::size_t t = 0; t < nLocalTrials; ++t) {
              T v = distRowI[t] + xn + candNorms[t];
              if (v < T{0}) {
                v = T{0};
              }
              dstRow[t] = v;
              scores[t] += (v < mi) ? v : mi;
            }
          }
        } else {
          // Fused scoring: for each x row, compute L distances against the candidate pack and
          // update L parallel running sums in one pass. The single-x-stream path is the load-
          // bearing win at envelope shapes where n*d far exceeds L2 -- one stream is the
          // difference between bandwidth-bound and bandwidth-bound times L. Parallelized over
          // X rows via per-worker score slabs reduced at the end; candDistSqData writes are
          // row-local so no aliasing across workers.
          const bool willParallelize = pool.shouldParallelize(n, 1024, 2) && pool.pool != nullptr;
          bool scoredViaSoa = false;
#ifdef CLUSTERING_USE_AVX2
          if constexpr (std::is_same_v<T, float>) {
            // SoA 8-row M-tile kernel: streams X AoS through an in-register 8x8 transpose so 8
            // rows' features land in feature-major YMM accumulators, folds L distances per row
            // without per-row horizontal reductions, writes the per-(row, cand) distances to
            // @c outDist, and accumulates min-capped scores. The kernel handles arbitrary row
            // counts, so per-worker row ranges slot in under the same parallel fan-out that
            // feeds the fallback path.
            const bool soaEligible = (d >= 8) && (nLocalTrials >= 1) && (nLocalTrials <= 6);
            if (soaEligible) {
              auto soaRange = [&](std::size_t lo, std::size_t hi, T *localScores) noexcept {
                const std::size_t rangeN = hi - lo;
                const float *xSlice = xData + (lo * d);
                const float *minSlice = minSq + lo;
                float *distSlice = candDistSqData + (lo * transposedWidth);
                switch (nLocalTrials) {
                case 1:
                  math::detail::kmppScoreSoaRowsAvx2F32<1>(xSlice, rangeN, d, candRowsData,
                                                           minSlice, distSlice, transposedWidth,
                                                           localScores);
                  break;
                case 2:
                  math::detail::kmppScoreSoaRowsAvx2F32<2>(xSlice, rangeN, d, candRowsData,
                                                           minSlice, distSlice, transposedWidth,
                                                           localScores);
                  break;
                case 3:
                  math::detail::kmppScoreSoaRowsAvx2F32<3>(xSlice, rangeN, d, candRowsData,
                                                           minSlice, distSlice, transposedWidth,
                                                           localScores);
                  break;
                case 4:
                  math::detail::kmppScoreSoaRowsAvx2F32<4>(xSlice, rangeN, d, candRowsData,
                                                           minSlice, distSlice, transposedWidth,
                                                           localScores);
                  break;
                case 5:
                  math::detail::kmppScoreSoaRowsAvx2F32<5>(xSlice, rangeN, d, candRowsData,
                                                           minSlice, distSlice, transposedWidth,
                                                           localScores);
                  break;
                case 6:
                  math::detail::kmppScoreSoaRowsAvx2F32<6>(xSlice, rangeN, d, candRowsData,
                                                           minSlice, distSlice, transposedWidth,
                                                           localScores);
                  break;
                default:
                  break;
                }
              };

              if (willParallelize) {
                const std::size_t workers = pool.workerCount();
                T *localScores = m_localScores.data();
                for (std::size_t e = 0; e < workers * nLocalTrials; ++e) {
                  localScores[e] = T{0};
                }
                pool.pool
                    ->submit_blocks(
                        std::size_t{0}, n,
                        [&](std::size_t lo, std::size_t hi) {
                          const std::size_t w = math::Pool::workerIndex();
                          soaRange(lo, hi, localScores + (w * nLocalTrials));
                        },
                        workers)
                    .wait();
                for (std::size_t w = 0; w < workers; ++w) {
                  const T *row = localScores + (w * nLocalTrials);
                  for (std::size_t t = 0; t < nLocalTrials; ++t) {
                    scores[t] += row[t];
                  }
                }
              } else {
                soaRange(0, n, scores.data());
              }
              scoredViaSoa = true;
            }
          }
#endif
          if (!scoredViaSoa) {
            auto scanRange = [&](std::size_t lo, std::size_t hi, T *localScores) noexcept {
              std::array<T, 32> distRowLocal{};
              for (std::size_t i = lo; i < hi; ++i) {
                const T *xi = xData + (i * d);
                const T mi = minSq[i];
                detail::sqEuclideanRowToBatch<T>(xi, candRowsData, nLocalTrials, d,
                                                 distRowLocal.data());
                T *dstRow = candDistSqData + (i * transposedWidth);
                for (std::size_t t = 0; t < nLocalTrials; ++t) {
                  dstRow[t] = distRowLocal[t];
                  localScores[t] += (distRowLocal[t] < mi) ? distRowLocal[t] : mi;
                }
              }
            };

            if (willParallelize) {
              const std::size_t workers = pool.workerCount();
              T *localScores = m_localScores.data();
              for (std::size_t e = 0; e < workers * nLocalTrials; ++e) {
                localScores[e] = T{0};
              }
              pool.pool
                  ->submit_blocks(
                      std::size_t{0}, n,
                      [&](std::size_t lo, std::size_t hi) {
                        const std::size_t w = math::Pool::workerIndex();
                        scanRange(lo, hi, localScores + (w * nLocalTrials));
                      },
                      workers)
                  .wait();
              for (std::size_t w = 0; w < workers; ++w) {
                const T *row = localScores + (w * nLocalTrials);
                for (std::size_t t = 0; t < nLocalTrials; ++t) {
                  scores[t] += row[t];
                }
              }
            } else {
              scanRange(0, n, scores.data());
            }
          }
        }
      }

      std::size_t bestT = 0;
      T bestScore = scores[0];
      for (std::size_t t = 1; t < nLocalTrials; ++t) {
        if (scores[t] < bestScore) {
          bestScore = scores[t];
          bestT = t;
        }
      }
      const std::size_t bestCandidate = candidates[bestT];

      // Commit best candidate: copy its row into outCentroids and refresh @c minSq from the
      // cached candidate-distance plane, skipping a fresh O(n*d) scan against the winner row.
      const T *winnerRow = xData + (bestCandidate * d);
      std::memcpy(centroidsData + (c * d), winnerRow, d * sizeof(T));
      for (std::size_t i = 0; i < n; ++i) {
        const T cand = candDistSqData[(i * transposedWidth) + bestT];
        if (cand < minSq[i]) {
          minSq[i] = cand;
        }
      }
    }
  }

private:
  void ensureShape(std::size_t n, std::size_t d, std::size_t L, std::size_t workers) {
    const std::size_t w = detail::greedyKmppTransposedWidth(L == 0 ? std::size_t{1} : L);
    if (m_candRows.dim(0) != L || m_candRows.dim(1) != d) {
      m_candRows = NDArray<T, 2, Layout::Contig>({L, d});
    }
    if (m_candRowsT.dim(0) != d || m_candRowsT.dim(1) != w) {
      m_candRowsT = NDArray<T, 2, Layout::Contig>({d == 0 ? std::size_t{1} : d, w});
    }
    if (m_candDistSq.dim(0) != n || m_candDistSq.dim(1) != w) {
      m_candDistSq = NDArray<T, 2, Layout::Contig>({n == 0 ? std::size_t{1} : n, w});
    }
    if (m_cumDistSq.dim(0) != n) {
      m_cumDistSq = NDArray<T, 1>({n});
    }
    if (m_minSq.dim(0) != n) {
      m_minSq = NDArray<T, 1>({n});
    }
    // GEMM-scoring-only scratch (distsFlat, xNormsSq, candNormsSq, gemmApArena, gemmBpArena).
    // The GEMM path fires at @c d >= 32 && L >= kKernelNr<float>; outside that envelope we keep
    // unit-sized placeholders so @c .data() stays dereferenceable without paying the @c kKc*kNc
    // envelope tax (@c Bp alone is several MB).
    constexpr std::size_t kNrForGemm = math::detail::kKernelNr<float>;
    const bool gemmScoringUsed = std::is_same_v<T, float> && (d >= 32) && (L >= kNrForGemm);
    const std::size_t nSafe = (n == 0) ? std::size_t{1} : n;
    const std::size_t lSafe = (L == 0) ? std::size_t{1} : L;
    const std::size_t distsFlatRows = gemmScoringUsed ? nSafe : std::size_t{1};
    const std::size_t distsFlatCols = gemmScoringUsed ? lSafe : std::size_t{1};
    if (m_distsFlat.dim(0) != distsFlatRows || m_distsFlat.dim(1) != distsFlatCols) {
      m_distsFlat = NDArray<T, 2, Layout::Contig>({distsFlatRows, distsFlatCols});
    }
    const std::size_t xNormsLen = gemmScoringUsed ? nSafe : std::size_t{1};
    if (m_xNormsSq.dim(0) != xNormsLen) {
      m_xNormsSq = NDArray<T, 1>({xNormsLen});
    }
    const std::size_t candNormsLen = gemmScoringUsed ? lSafe : std::size_t{1};
    if (m_candNormsSq.dim(0) != candNormsLen) {
      m_candNormsSq = NDArray<T, 1>({candNormsLen});
    }
    const std::size_t workersClamped = workers == 0 ? std::size_t{1} : workers;
    // @c gemmRunReference parallelizes the Mc-tile loop, with each worker owning a per-worker
    // slice of the A-pack arena at offset @c (worker * kMc * kKc). Sizing the arena for just
    // one worker was fine while the seeder's envelope kept the GEMM path off (k=16, L=4 fell
    // into the SoA kernel), but the Elkan-eligible shapes push L >= kNrF where the GEMM scoring
    // activates and multiple workers collide into the same slice.
    const std::size_t apSize = gemmScoringUsed
                                   ? (workersClamped * math::detail::kMc<T> * math::detail::kKc<T>)
                                   : std::size_t{1};
    const std::size_t bpSize =
        gemmScoringUsed ? (math::detail::kKc<T> * math::detail::kNc<T>) : std::size_t{1};
    if (m_gemmApArena.dim(0) != apSize) {
      m_gemmApArena = NDArray<T, 1>({apSize});
    }
    if (m_gemmBpArena.dim(0) != bpSize) {
      m_gemmBpArena = NDArray<T, 1>({bpSize});
    }
    const std::size_t lsLen = workersClamped * (L == 0 ? std::size_t{1} : L);
    if (m_localScores.dim(0) != lsLen) {
      m_localScores = NDArray<T, 1>({lsLen});
    }
  }

  /// Packed candidate rows: shape @c (L, d). Reused across all @c k-1 outer picks within one run.
  NDArray<T, 2, Layout::Contig> m_candRows;
  /// Transposed candidate rows: shape @c (d, W) where @c W = greedyKmppTransposedWidth(L).
  /// Padded to a multiple of the 8-wide YMM lane so the transposed kernel iterates over
  /// fixed-width chunks; lanes past @c L are zero-filled and the scoring loop reads only the
  /// first @c L lanes.
  NDArray<T, 2, Layout::Contig> m_candRowsT;
  /// Per-outer-iteration cache of candidate distances: shape @c (n, W) matching the transposed
  /// pack. Lets the commit step pick out the winner column without re-scanning x against the
  /// winner centroid -- saves an O(n*d) pass per outer pick.
  NDArray<T, 2, Layout::Contig> m_candDistSq;
  /// Per-outer-iteration prefix sum of @c minSq, length @c n. Drives inverse-CDF sampling in
  /// @c log(n) time instead of an @c L*n linear scan.
  NDArray<T, 1> m_cumDistSq;
  /// Per-point running min-squared-distance to the selected centroid set. Private to the
  /// seeder; the Lloyd policy owns its own per-point distance scratch.
  NDArray<T, 1> m_minSq;
  /// (n, L) flat scratch holding GEMM-based candidate-distance output for the high-d path.
  NDArray<T, 2, Layout::Contig> m_distsFlat;
  /// Per-point ||x||^2, length n. Computed once per seeder run(); reused across all picks.
  NDArray<T, 1> m_xNormsSq;
  /// Per-candidate ||c||^2, length L. Refreshed each pick.
  NDArray<T, 1> m_candNormsSq;
  /// Persistent GEMM Ap arena (kMc * kKc). Passed by pointer to gemmRunReference per pick.
  NDArray<T, 1> m_gemmApArena;
  /// Persistent GEMM Bp arena (kKc * kNc). Passed by pointer to gemmRunReference per pick.
  NDArray<T, 1> m_gemmBpArena;
  /// Per-worker local-scores scratch (workers * L). Scorers accumulate into their own slab
  /// to avoid atomic contention; the reduce pass at pick-end folds into the outer scores[].
  NDArray<T, 1> m_localScores;
};

} // namespace clustering::kmeans
