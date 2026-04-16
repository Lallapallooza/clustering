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
 * Matches sklearn's convention @c 2 + floor(ln(k)). Gives k=8 -> L=4, k=64 -> L=6, k=256 ->
 * L=7, k=1000 -> L=8, against a @c 2 + floor(log2(k)) variant that would give k=8 -> L=5,
 * k=256 -> L=10. The natural-log form keeps inertia within the per-pick scoring envelope while
 * trimming ~30% off the seeder's candidate work at high @c k.
 *
 * @return Local-trials count; always at least 1 (the @c k=1 path is short-circuited above this
 *         routine, but a non-zero floor is the safer contract).
 */
[[nodiscard]] inline std::size_t greedyKmppLocalTrials(std::size_t k) noexcept {
  if (k <= 1) {
    return 1;
  }
  // std::log is constexpr-callable in C++26 only; this routine fires once per fit so the
  // runtime call is fine, and the cast-to-size_t of a non-negative value is well-defined.
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

using math::detail::sqEuclideanRowPtr;

#ifdef CLUSTERING_USE_AVX2

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
    out[t] = math::detail::horizontalSumAvx2(acc[t]) + tail[t];
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
 * @brief Compute @p L squared Euclidean distances against a transposed @c (d, L) candidate
 *        layout, streaming the @p x row through one SIMD accumulator that holds all @p L lanes.
 *
 * The (d, L) layout puts the k-th feature of every candidate at @c cand[k*L .. k*L + L), so for
 * @c L == 8 a single @c _mm256_load_ps fetches all 8 candidates' k-th component; broadcasting
 * @c x[k] then folds 8 squared-distance contributions in one FMA. At @c d < 8 this collapses
 * the per-row scoring from @c L*d scalar ops to @c d SIMD ops, breaking the d=4 bottleneck on
 * the high-k seeding hot path. Identical numerical result to the @c (L, d) row-batched kernel
 * within float reassociation tolerance.
 *
 * @param x         Pointer to the @p d -length x row.
 * @param candData  Pointer to the (d, 8) transposed candidate matrix; lane t = candidate t.
 * @param d         Feature dimension.
 * @param out       Length-8 output buffer; @c out[t] receives @c ||x - cand_t||^2.
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
 * @brief Compute two 8-way squared distance slabs against a transposed @c (d, 16) candidate
 *        layout in one streaming pass over the @p x row.
 *
 * Unrolls @ref sqEuclideanRowAgainst8Transposed across two adjacent lane groups so each @c x[k]
 * broadcast folds 16 candidate distances per FMA pair. The row stride of @p candData is 16
 * elements; lanes @c [0,8) receive the first 8 candidates and @c [8,16) the next 8. At
 * @c L in @c (8, 16] with @c d <= 8 this shaves half the broadcast + load traffic versus looping
 * the 8-wide kernel twice, which is the load-bearing envelope for @c k >= 256 seeding where
 * @c L = 2 + ln(k) lands at 10 or 11.
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
 * @brief Compute one 8-way squared distance slab against a transposed @c (d, W) candidate
 *        layout with an explicit row stride @p W.
 *
 * Generalizes @ref sqEuclideanRowAgainst8Transposed to the L > 16 regime where the transposed
 * pack keeps @c W = ceil(L/8) * 8 columns so chunked scoring can slide an 8-wide window across
 * it. The row stride differs from the lane count, so the per-k load uses @c k * W rather than
 * @c k * 8.
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
 * The solver owns one instance across @c run() calls so the candidate-row pack, the per-(point,
 * candidate) score plane and the inverse CDF scratch are not reallocated at the same shape
 * tuple. Sized lazily by @ref ensureGreedyKmppScratchShape; callers outside the solver should
 * size @c candRows to @c (greedyKmppLocalTrials(k), d), @c candRowsT to
 * @c (d, greedyKmppTransposedWidth(L)), @c candDistSq to @c (n, greedyKmppTransposedWidth(L)),
 * and @c cumDistSq to length @c n before calling @ref seedGreedyKMeansPlusPlus.
 */
template <class T> struct GreedyKmppScratch {
  /// Packed candidate rows for one outer iteration: shape @c (L, d) where
  /// @c L = greedyKmppLocalTrials(k). Reused across all @c k-1 outer picks within one fit.
  NDArray<T, 2, Layout::Contig> candRows;
  /// Transposed candidate rows: shape @c (d, W) where
  /// @c W = greedyKmppTransposedWidth(L). Padded to a multiple of the 8-wide YMM lane so the
  /// transposed kernel iterates over fixed-width chunks; lanes past @p L are zero-filled and
  /// the scoring loop reads only the first @p L lanes. Used by the transposed scoring kernel
  /// on the @c d <= kAvx2Lanes hot path; ignored when the row-batched kernel is the better
  /// choice.
  NDArray<T, 2, Layout::Contig> candRowsT;
  /// Per-outer-iteration cache of candidate distances: shape @c (n, W) matching the transposed
  /// pack. Lets the commit step pick out the winner column without re-scanning x against the
  /// winner centroid -- saves an O(n*d) pass per outer pick.
  NDArray<T, 2, Layout::Contig> candDistSq;
  /// Per-outer-iteration prefix sum of @c minSq, length @c n. Refreshed once per outer pick;
  /// the L candidate draws then run @c log(n) binary search instead of an @c L*n linear scan.
  NDArray<T, 1> cumDistSq;

  GreedyKmppScratch() : candRows({0, 0}), candRowsT({0, 0}), candDistSq({0, 0}), cumDistSq({0}) {}
};

/**
 * @brief Lazily resize @p scratch to hold the candidate pack and per-(point, candidate)
 *        score-cache buffers.
 *
 * No-op when all buffers already match the requested shape; otherwise reallocates.
 */
template <class T>
inline void ensureGreedyKmppScratchShape(GreedyKmppScratch<T> &scratch, std::size_t L,
                                         std::size_t d, std::size_t n) {
  const std::size_t w = greedyKmppTransposedWidth(L == 0 ? std::size_t{1} : L);
  if (scratch.candRows.dim(0) != L || scratch.candRows.dim(1) != d) {
    scratch.candRows = NDArray<T, 2, Layout::Contig>({L, d});
  }
  if (scratch.candRowsT.dim(0) != d || scratch.candRowsT.dim(1) != w) {
    scratch.candRowsT = NDArray<T, 2, Layout::Contig>({d == 0 ? std::size_t{1} : d, w});
  }
  if (scratch.candDistSq.dim(0) != n || scratch.candDistSq.dim(1) != w) {
    scratch.candDistSq = NDArray<T, 2, Layout::Contig>({n == 0 ? std::size_t{1} : n, w});
  }
  if (scratch.cumDistSq.dim(0) != n) {
    scratch.cumDistSq = NDArray<T, 1>({n});
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
  CLUSTERING_ALWAYS_ASSERT(scratch.candRowsT.isMutable());
  CLUSTERING_ALWAYS_ASSERT(scratch.candRowsT.dim(0) >= d);
  CLUSTERING_ALWAYS_ASSERT(scratch.candRowsT.dim(1) >= greedyKmppTransposedWidth(nLocalTrials));
  CLUSTERING_ALWAYS_ASSERT(scratch.candDistSq.isMutable());
  CLUSTERING_ALWAYS_ASSERT(scratch.candDistSq.dim(0) >= n);
  CLUSTERING_ALWAYS_ASSERT(scratch.candDistSq.dim(1) >= greedyKmppTransposedWidth(nLocalTrials));
  CLUSTERING_ALWAYS_ASSERT(scratch.cumDistSq.isMutable());
  CLUSTERING_ALWAYS_ASSERT(scratch.cumDistSq.dim(0) >= n);
  CLUSTERING_ALWAYS_ASSERT(k >= 1);
  CLUSTERING_ALWAYS_ASSERT(n >= k);

  (void)pool;

  math::pcg64 rng;
  rng.seed(seed);

  const T *xData = X.data();
  T *centroidsData = outCentroids.data();
  T *minSq = outMinDistSq.data();
  T *candRowsData = scratch.candRows.data();
  T *cumDistSq = scratch.cumDistSq.data();
  T *candDistSqData = scratch.candDistSq.data();
#ifdef CLUSTERING_USE_AVX2
  T *candRowsTData = scratch.candRowsT.data();
#endif

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
    // Build the cumulative-distance array in a single pass and pull the total off the tail.
    // No need for Kahan because the cum array is only used as a probability normalizer.
    T runningSum = T{0};
    for (std::size_t i = 0; i < n; ++i) {
      runningSum += minSq[i];
      cumDistSq[i] = runningSum;
    }
    const T total = runningSum;

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

    // Draw nLocalTrials candidates by inverse-CDF sampling on the cumulative array via
    // @c std::upper_bound (binary search), trimming the @c L*n linear scan to @c L*log(n).
    // Determinism is preserved: identical seed + identical n still produces identical candidate
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
    // can stream x once across L accumulators. The pack cost is L*d -- negligible against the
    // n-pass scoring it amortizes.
    for (std::size_t t = 0; t < nLocalTrials; ++t) {
      std::memcpy(candRowsData + (t * d), xData + (candidates[t] * d), d * sizeof(T));
    }

    // Reset per-candidate running scores.
    for (std::size_t t = 0; t < nLocalTrials; ++t) {
      scores[t] = T{0};
    }
    // Stack scratch for one row of candidate distances; sized to cover the L envelope at
    // realistic k (L = 2 + ln(k) <= 32 spans k up to 2^30, well past any usable shape).
    constexpr std::size_t kMaxLocalTrials = 32;
    std::array<T, kMaxLocalTrials> distRow{};
    CLUSTERING_ALWAYS_ASSERT(nLocalTrials <= distRow.size());

    const std::size_t transposedWidth = greedyKmppTransposedWidth(nLocalTrials);
    bool scoredViaTransposed = false;
#ifdef CLUSTERING_USE_AVX2
    // Low-d hot path: at d <= kAvx2Lanes the (L, d) row-batched kernel either falls into the
    // scalar K-tail (d < 8) or pays @c L horizontal-sum reductions for one K-iter of work
    // (d == 8). The transposed @c (d, W) layout with @c W = ceil(L/8)*8 puts the same-feature
    // components of every candidate in consecutive 8-lane YMM registers, so each
    // @c broadcast-of-x[k] + FMA pair folds 8 (or 16, for the 16-lane unroll) distances at
    // once -- @c d SIMD FMAs per row plus one store per 8-lane chunk, replacing the row-batched
    // path's @c L horizontal-sum reductions.
    if constexpr (std::is_same_v<T, float>) {
      if (d > 0 && d <= math::detail::kAvx2Lanes<float>) {
        // Transpose the (L, d) candidate pack into a (d, W) layout. Lanes past nLocalTrials are
        // zeroed; the score loop only reads the first nLocalTrials lanes so the fill value does
        // not matter for correctness, but zeroing keeps the padded distances at whatever an
        // x-to-zero-row squared distance yields -- ignored downstream, still deterministic.
        for (std::size_t kk = 0; kk < d; ++kk) {
          float *dstK = candRowsTData + (kk * transposedWidth);
          for (std::size_t t = 0; t < nLocalTrials; ++t) {
            dstK[t] = candRowsData[(t * d) + kk];
          }
          for (std::size_t t = nLocalTrials; t < transposedWidth; ++t) {
            dstK[t] = 0.0F;
          }
        }
        // L in (8, 16] is the common k >= 256 regime (L = 2 + ln(k) in [8.5, 9.2] for
        // k in [256, 10000]). The 16-lane unroll folds two 8-chunks per x[k] broadcast so the
        // d <= 8 hot loop bottleneck at the k=512/1000 corners drops to d SIMD-pair FMAs per
        // row versus two independent 8-chunk passes. Scores accumulate into per-chunk YMM
        // registers so the per-row @c min(dist, mi) + sum is 2 SIMD ops instead of 16 scalar
        // branches -- the load-bearing tweak that unblocks seeding at @c k >= 512.
        if (transposedWidth == 16) {
          __m256 scoresLoAcc = _mm256_setzero_ps();
          __m256 scoresHiAcc = _mm256_setzero_ps();
          for (std::size_t i = 0; i < n; ++i) {
            const float *xi = xData + (i * d);
            const __m256 miVec = _mm256_set1_ps(minSq[i]);
            float *dstRow = candDistSqData + (i * transposedWidth);
            sqEuclideanRowAgainst16Transposed(xi, candRowsTData, d, dstRow);
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
            sqEuclideanRowAgainst8Transposed(xi, candRowsTData, d, dstRow);
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
          // at a time so each chunk stays on the fully unrolled 8-wide kernel, re-indexing with
          // the true row stride because the contiguous-row form reads @c k * 8.
          for (std::size_t i = 0; i < n; ++i) {
            const float *xi = xData + (i * d);
            const float mi = minSq[i];
            float *dstRow = candDistSqData + (i * transposedWidth);
            for (std::size_t base = 0; base < transposedWidth; base += 8) {
              sqEuclideanRowAgainst8TransposedStrided(xi, candRowsTData + base, d, transposedWidth,
                                                      dstRow + base);
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
      // Fused scoring: for each x row, compute L distances against the candidate pack and update
      // L parallel running sums in one pass. The single-x-stream path is the load-bearing win
      // over L separate per-candidate sweeps because n*d at our envelope (~12.8 MB at the gate
      // shape) far exceeds L2 -- one stream is the difference between bandwidth-bound and
      // bandwidth-bound times L.
      for (std::size_t i = 0; i < n; ++i) {
        const T *xi = xData + (i * d);
        const T mi = minSq[i];
        sqEuclideanRowToBatch<T>(xi, candRowsData, nLocalTrials, d, distRow.data());
        T *dstRow = candDistSqData + (i * transposedWidth);
        for (std::size_t t = 0; t < nLocalTrials; ++t) {
          dstRow[t] = distRow[t];
          scores[t] += (distRow[t] < mi) ? distRow[t] : mi;
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

    // Commit best candidate: copy its row into @p outCentroids and refresh @c minSq from the
    // cached candidate-distance plane. This skips a fresh O(n*d) scan against the winner row
    // -- the scoring loop already computed the per-(point, candidate) distance matrix.
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

} // namespace clustering::kmeans::detail
