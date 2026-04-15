#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <vector>

#include "clustering/always_assert.h"
#include "clustering/math/pairwise.h"
#include "clustering/math/rng.h"
#include "clustering/math/thread.h"
#include "clustering/ndarray.h"

#ifdef CLUSTERING_USE_AVX2
#include <immintrin.h>
#endif

namespace clustering::kmeans::detail {

/**
 * @brief Compute the local-trials count used by greedy k-means++.
 *
 * @return @c 2 + floor(log2(k)); always at least 1 (the @c k=1 path is short-circuited above
 *         this routine, but a non-zero floor is the safer contract).
 */
[[nodiscard]] inline std::size_t greedyKmppLocalTrials(std::size_t k) noexcept {
  if (k <= 1) {
    return 1;
  }
  // floor(log2(k)) for k >= 2.
  std::size_t lg = 0;
  std::size_t v = k;
  while ((v >>= 1U) != 0U) {
    ++lg;
  }
  return 2 + lg;
}

/**
 * @brief Squared Euclidean distance between two contiguous rows of equal length @p d.
 *
 * Hot helper for the seeder's candidate-distance update; routes to the AVX2-vectorized
 * @c math::detail::sqEuclideanRowAvx2 when the build is AVX2-enabled and @p d clears one
 * lane width, otherwise the scalar fallback. The seeder calls this kernel @c O(n*k*L) times
 * at @c (n=1e5, d=32, k=64, L=8); the AVX2 dispatch is load-bearing for the seeding wall-time
 * envelope.
 */
template <class T>
[[nodiscard]] inline T sqEuclideanRowPtr(const T *a, const T *b, std::size_t d) noexcept {
#ifdef CLUSTERING_USE_AVX2
  if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
    if (d >= math::detail::kAvx2Lanes<T>) {
      return math::detail::sqEuclideanRowAvx2(a, b, d);
    }
  }
#endif
  T s = T{0};
  for (std::size_t t = 0; t < d; ++t) {
    const T diff = a[t] - b[t];
    s += diff * diff;
  }
  return s;
}

#ifdef CLUSTERING_USE_AVX2

/**
 * @brief Faster horizontal sum over an AVX2 single-precision register.
 *
 * Replaces the @c hadd_ps pair with a shuffle-and-add ladder. @c hadd_ps is microcoded on
 * Skylake/Zen and adds ~6 cycles per pair; the explicit shuffle path stays on the FP add port
 * and pipelines per-lane reductions inside the candidate-scoring loop where eight reductions
 * fire per x row.
 */
inline float horizontalSumFastAvx2(__m256 v) noexcept {
  const __m128 hi = _mm256_extractf128_ps(v, 1);
  const __m128 lo = _mm256_castps256_ps128(v);
  __m128 sum = _mm_add_ps(lo, hi);
  sum = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
  sum = _mm_add_ss(sum, _mm_shuffle_ps(sum, sum, 1));
  return _mm_cvtss_f32(sum);
}

inline double horizontalSumFastAvx2(__m256d v) noexcept {
  const __m128d hi = _mm256_extractf128_pd(v, 1);
  const __m128d lo = _mm256_castpd256_pd128(v);
  __m128d sum = _mm_add_pd(lo, hi);
  sum = _mm_add_sd(sum, _mm_unpackhi_pd(sum, sum));
  return _mm_cvtsd_f64(sum);
}

/**
 * @brief Compile-time batched scoring kernel: stream @p x once across @c B parallel AVX2
 *        accumulators to compute @c B squared distances against @p candData rows.
 *
 * Templating on @c B unblocks the compiler's full unroll of the inner candidate loop, which
 * is the load-bearing optimisation for the seeder's candidate scoring. The runtime entry
 * @ref sqEuclideanRowToBatchAvx2 dispatches to the @c B=8 specialisation for the common
 * @c L=8 case (k in [16, 31]) and to other compile-time @c B values for adjacent batches.
 */
template <std::size_t B>
inline void sqEuclideanRowToBatchAvx2Fixed(const float *x, const float *candData, std::size_t d,
                                           float *out) noexcept {
  static_assert(B >= 1 && B <= 8, "B must lie in [1, 8] -- 8 ymm regs hold the batch");
  std::array<__m256, B> acc{};
  for (std::size_t t = 0; t < B; ++t) {
    acc[t] = _mm256_setzero_ps();
  }
  std::size_t k = 0;
  for (; k + 8 <= d; k += 8) {
    const __m256 vx = _mm256_loadu_ps(x + k);
    for (std::size_t t = 0; t < B; ++t) {
      const __m256 vc = _mm256_loadu_ps(candData + (t * d) + k);
      const __m256 diff = _mm256_sub_ps(vx, vc);
      acc[t] = _mm256_fmadd_ps(diff, diff, acc[t]);
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
    out[t] = horizontalSumFastAvx2(acc[t]) + tail[t];
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
    out[t] = horizontalSumFastAvx2(acc[t]) + tail[t];
  }
}

/**
 * @brief Compute squared Euclidean distance from one @p x row to @p L candidate rows in
 *        parallel using @p L independent AVX2 accumulators.
 *
 * Streams the @p x row through L parallel @c fmadd accumulators so each x byte is read from
 * memory once for all L distances; the 16 ymm registers comfortably hold up to 8
 * accumulators plus loop scratch on the candidate-scoring inner loop. Dispatches to
 * @ref sqEuclideanRowToBatchAvx2Fixed at compile-time-bound @c B for the common batch sizes
 * (matching @c L = 2 + floor(log2(k)) at @c k in [4, 8192]); larger @p L is handled via
 * full-batch chunks of 8 plus a tail dispatch.
 *
 * @param x         Pointer to the @p d -length x row.
 * @param candData  Pointer to the (L, d) row-major candidate matrix.
 * @param L         Number of candidate rows.
 * @param d         Feature dimension.
 * @param out       Length-@p L output buffer; @c out[t] receives @c ||x - candData[t]||^2.
 */
template <class T>
inline void sqEuclideanRowToBatchAvx2(const T *x, const T *candData, std::size_t L, std::size_t d,
                                      T *out) noexcept {
  std::size_t base = 0;
  while (base + 8 <= L) {
    sqEuclideanRowToBatchAvx2Fixed<8>(x, candData + (base * d), d, out + base);
    base += 8;
  }
  // Switch on the tail size so each branch dispatches to a fully compile-time-unrolled kernel.
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
    // Defensive; the switch covers the full [0, 7] tail range for the loop above.
    break;
  }
}

#endif // CLUSTERING_USE_AVX2

/**
 * @brief Squared Euclidean distance from one @p x row to a batch of @p L candidate rows.
 *
 * Routes to the AVX2 batched kernel when the build is AVX2-enabled and @p d clears one
 * lane width; otherwise dispatches @p L scalar @ref sqEuclideanRowPtr calls. The batched
 * AVX2 path streams the @p x row through @p L parallel accumulators so the inner loop
 * reads each x byte once across the whole batch; on the scoring inner loop this is the
 * difference between L bandwidth-bound passes and one.
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

/**
 * @brief Scratch storage for @ref seedGreedyKMeansPlusPlus.
 *
 * The solver owns one instance across @c run() calls so the candidate-row pack used by the
 * batched scoring kernel is not reallocated at the same shape tuple. Sized lazily by
 * @ref ensureGreedyKmppScratchShape; callers outside the solver should size @c candRows
 * to @c (greedyKmppLocalTrials(k), d) before calling @ref seedGreedyKMeansPlusPlus.
 */
template <class T> struct GreedyKmppScratch {
  /// Packed candidate rows for one outer iteration: shape @c (L, d) where
  /// @c L = greedyKmppLocalTrials(k). Reused across all @c k-1 outer picks within one fit.
  NDArray<T, 2, Layout::Contig> candRows;

  GreedyKmppScratch() : candRows({0, 0}) {}
};

/**
 * @brief Lazily resize @p scratch to hold a @c (L, d) candidate-row pack.
 *
 * No-op when the existing @c candRows already matches the requested shape; otherwise
 * reallocates. Intended for callers that own @ref GreedyKmppScratch across repeated fits.
 */
template <class T>
inline void ensureGreedyKmppScratchShape(GreedyKmppScratch<T> &scratch, std::size_t L,
                                         std::size_t d) {
  if (scratch.candRows.dim(0) != L || scratch.candRows.dim(1) != d) {
    scratch.candRows = NDArray<T, 2, Layout::Contig>({L, d});
  }
}

/**
 * @brief Greedy k-means++ seeding.
 *
 * Picks @c k initial centroid rows from @p X. The first centroid is drawn uniformly; each
 * subsequent centroid is the best of @c n_local_trials candidates sampled with probability
 * proportional to @c D(x)^2 -- the squared distance from each point to its nearest already-
 * chosen centroid. The candidate that yields the smallest resulting sum of squared minimum
 * distances wins.
 *
 * The local-trials count @c L = 2 + floor(log2(k)) matches the conventional sklearn seeder
 * heuristic.
 *
 * Performance: the candidate-scoring inner loop dominates wall time at the @c (n=1e5, d=32,
 * k=64) shape -- @c k*L*n distance computations stream the data matrix once per candidate
 * pick. The implementation packs the @c L candidates into a contiguous @c (L, d) buffer and
 * scores them via @ref sqEuclideanRowToBatch, which streams each x row through @c L parallel
 * AVX2 accumulators so the data matrix is read once per outer pick instead of @c L times.
 * Caller-supplied @p scratch owns the candidate-row pack; the solver allocates it once at
 * shape change and reuses it across runs.
 *
 * @tparam T Element type; @c float only.
 * @param X            Data matrix (n x d), contiguous.
 * @param outCentroids Output centroid matrix (k x d), contiguous; populated in row order.
 * @param outMinDistSq Output per-point min-squared-distance (length n); populated as the
 *                     seeder progresses, leaves the final state at the end of the routine.
 * @param scratch      Solver-owned candidate-row pack, sized @c (L, d).
 * @param seed         RNG seed; identical seed produces identical centroid selections.
 * @param pool         Parallelism injection. Currently unused in the seeder body; reserved
 *                     for a future per-chunk fan-out of the scoring loop.
 */
template <class T>
void seedGreedyKMeansPlusPlus(const NDArray<T, 2, Layout::Contig> &X,
                              NDArray<T, 2, Layout::Contig> &outCentroids,
                              NDArray<T, 1> &outMinDistSq, GreedyKmppScratch<T> &scratch,
                              std::uint64_t seed, math::Pool pool) {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                "seedGreedyKMeansPlusPlus<T> requires T to be float or double");

  const std::size_t n = X.dim(0);
  const std::size_t d = X.dim(1);
  const std::size_t k = outCentroids.dim(0);
  const std::size_t nLocalTrials = greedyKmppLocalTrials(k);

  CLUSTERING_ALWAYS_ASSERT(outCentroids.isMutable());
  CLUSTERING_ALWAYS_ASSERT(outMinDistSq.isMutable());
  CLUSTERING_ALWAYS_ASSERT(outCentroids.dim(1) == d);
  CLUSTERING_ALWAYS_ASSERT(outMinDistSq.dim(0) == n);
  CLUSTERING_ALWAYS_ASSERT(scratch.candRows.isMutable());
  CLUSTERING_ALWAYS_ASSERT(scratch.candRows.dim(0) >= nLocalTrials);
  CLUSTERING_ALWAYS_ASSERT(scratch.candRows.dim(1) == d);
  CLUSTERING_ALWAYS_ASSERT(k >= 1);
  CLUSTERING_ALWAYS_ASSERT(n >= k);

  (void)pool;

  math::pcg64 rng;
  rng.seed(seed);

  const T *xData = X.data();
  T *centroidsData = outCentroids.data();
  T *minSq = outMinDistSq.data();
  T *candRowsData = scratch.candRows.data();

  // Step 1: pick first centroid uniformly. randUniformU64 is the deterministic primitive; we
  // map it onto [0, n) via modulo, which carries a tiny bias for very large n but is the
  // standard sklearn convention.
  const auto first = static_cast<std::size_t>(math::randUniformU64(rng) % n);
  std::memcpy(centroidsData, xData + (first * d), d * sizeof(T));

  // Initialize per-point min-distance against the chosen first centroid.
  for (std::size_t i = 0; i < n; ++i) {
    minSq[i] = sqEuclideanRowPtr(xData + (i * d), centroidsData, d);
  }

  if (k == 1) {
    return;
  }

  // Per-iteration scratch: candidate indices and per-candidate scores. nLocalTrials is small
  // (8 at k=64; 12 at k=4096) so the small-n vector copy at construction is cheap and the
  // allocation only fires once per @c run -- not per outer iteration.
  std::vector<std::size_t> candidates(nLocalTrials, 0);
  std::vector<T> scores(nLocalTrials, T{0});

  for (std::size_t c = 1; c < k; ++c) {
    // Sum of current minSq is the "weight total" -- a deterministic running sum is enough; no
    // need for Kahan because we only use it as a probability normalizer.
    T total = T{0};
    for (std::size_t i = 0; i < n; ++i) {
      total += minSq[i];
    }

    // Degenerate guard: if every chosen centroid coincides with every remaining point (e.g.
    // duplicated data), total can collapse to ~0. Pick the next centroid uniformly so the
    // routine cannot stall in that degenerate corner.
    if (!(total > T{0})) {
      const auto pick = static_cast<std::size_t>(math::randUniformU64(rng) % n);
      std::memcpy(centroidsData + (c * d), xData + (pick * d), d * sizeof(T));
      // Re-seed minSq against the new centroid (other points keep their existing minSq because
      // it was already 0).
      for (std::size_t i = 0; i < n; ++i) {
        const T cand = sqEuclideanRowPtr(xData + (i * d), centroidsData + (c * d), d);
        if (cand < minSq[i]) {
          minSq[i] = cand;
        }
      }
      continue;
    }

    // Draw nLocalTrials candidates by inverse-CDF sampling on the running cumulative-distance
    // array. The fixed-trial loop is a deterministic sequence of randUnit draws so identical
    // seed + identical n produces identical candidate sets.
    for (std::size_t t = 0; t < nLocalTrials; ++t) {
      const T u = math::randUnit<T>(rng) * total;
      T cum = T{0};
      std::size_t pick = n - 1;
      for (std::size_t i = 0; i < n; ++i) {
        cum += minSq[i];
        if (cum > u) {
          pick = i;
          break;
        }
      }
      candidates[t] = pick;
    }

    // Pack the L candidate rows into a contiguous (L, d) buffer so the batched scoring kernel
    // can stream x once across L accumulators. The pack cost is L*d -- negligible against the
    // n-pass scoring it amortizes.
    for (std::size_t t = 0; t < nLocalTrials; ++t) {
      std::memcpy(candRowsData + (t * d), xData + (candidates[t] * d), d * sizeof(T));
    }

    // Fused scoring: for each x row, compute L distances against the candidate pack and update
    // L parallel running sums in one pass. The single-x-stream path is the load-bearing win
    // over L separate per-candidate sweeps because n*d at our envelope (~12.8 MB at the gate
    // shape) far exceeds L2 -- one stream is the difference between bandwidth-bound and
    // bandwidth-bound times L.
    for (std::size_t t = 0; t < nLocalTrials; ++t) {
      scores[t] = T{0};
    }
    // Stack scratch for one row of candidate distances; sized to cover the L envelope at
    // realistic k (L = 2 + log2(k) <= 32 spans k up to 2^30, well past any usable shape).
    constexpr std::size_t kMaxLocalTrials = 32;
    std::array<T, kMaxLocalTrials> distRow{};
    CLUSTERING_ALWAYS_ASSERT(nLocalTrials <= distRow.size());
    for (std::size_t i = 0; i < n; ++i) {
      const T *xi = xData + (i * d);
      const T mi = minSq[i];
      sqEuclideanRowToBatch<T>(xi, candRowsData, nLocalTrials, d, distRow.data());
      for (std::size_t t = 0; t < nLocalTrials; ++t) {
        scores[t] += (distRow[t] < mi) ? distRow[t] : mi;
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

    // Commit best candidate: copy its row into @p outCentroids and refresh @c minSq to the
    // per-point minimum against the newly added centroid. The candidate-row pack already has
    // the winning row at index @c bestT, so the read for the minSq refresh shares the cache
    // line with the data the prior scoring loop just wrote.
    const T *winnerRow = xData + (bestCandidate * d);
    std::memcpy(centroidsData + (c * d), winnerRow, d * sizeof(T));
    for (std::size_t i = 0; i < n; ++i) {
      const T cand = sqEuclideanRowPtr(xData + (i * d), winnerRow, d);
      if (cand < minSq[i]) {
        minSq[i] = cand;
      }
    }
  }
}

} // namespace clustering::kmeans::detail
