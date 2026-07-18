#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>
#include <vector>

#include "clustering/always_assert.h"
#include "clustering/math/detail/avx2_helpers.h"
#include "clustering/math/detail/gemm_outer.h"
#include "clustering/math/detail/inverse_cdf_blocks.h"
#include "clustering/math/detail/matrix_desc.h"
#include "clustering/math/detail/sq_distances_block.h"
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

/**
 * @brief Fan-out width for one of the seeder's per-round `O(n*d)` sweeps.
 *
 * Sweeps that pass the row-count gate keep the full @c stealBlocks width. Below the
 * gate the width is instead capped at one block per fixed kernel-op budget so every
 * block still amortizes its share of the dispatch; a width of one keeps the sweep serial.
 */
[[nodiscard]] inline std::size_t greedyKmppSweepBlocks(math::Pool pool, std::size_t rows,
                                                       std::size_t opsPerRow) noexcept {
  constexpr std::size_t kMinOpsPerBlock = std::size_t{1} << 15;
  if (pool.pool == nullptr || rows == 0) {
    return 1;
  }
  if (pool.shouldParallelize(rows, 1024, 2)) {
    return pool.stealBlocks(rows);
  }
  const std::size_t cap = std::max<std::size_t>(1, (rows * opsPerRow) / kMinOpsPerBlock);
  return std::min(pool.stealBlocks(rows), cap);
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
 * @brief Compute @p L squared Euclidean distances against an `(L, d)` row-batched candidate
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
 * @brief Compute @p L squared distances against an `(d, 8)` transposed candidate layout with
 *        one streaming pass over the @p x row.
 *
 * The `(d, 8)` layout puts the k-th feature of every candidate at `cand[k*8 .. k*8 + 8)`, so a
 * single @c _mm256_load_ps fetches all 8 candidates' k-th component; broadcasting `x[k]` then
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
 * @brief Register-only variant of @ref sqEuclideanRowAgainst8Transposed.
 *
 * Returns the 8-lane squared-distance vector instead of storing it. Used by the seeder's
 * score sweep to skip a per-row materialization into a `(n, L)` distance buffer when only
 * the score reduction is needed; the winner column is recomputed cheaply after pick time.
 */
inline __m256 sqEuclideanRowAgainst8TransposedReg(const float *x, const float *candData,
                                                  std::size_t d) noexcept {
  __m256 acc = _mm256_setzero_ps();
  for (std::size_t k = 0; k < d; ++k) {
    const __m256 cv = _mm256_load_ps(candData + (k * 8));
    const __m256 xv = _mm256_set1_ps(x[k]);
    const __m256 diff = _mm256_sub_ps(xv, cv);
    acc = _mm256_fmadd_ps(diff, diff, acc);
  }
  return acc;
}

/**
 * @brief Register-only 16-wide variant of @ref sqEuclideanRowAgainst16Transposed.
 *
 * Returns the two 8-lane squared-distance vectors as a pair.
 */
inline std::pair<__m256, __m256> sqEuclideanRowAgainst16TransposedReg(const float *x,
                                                                      const float *candData,
                                                                      std::size_t d) noexcept {
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
  return {accLo, accHi};
}

/**
 * @brief Compute two 8-way squared distance slabs against an `(d, 16)` transposed candidate
 *        layout in one streaming pass over the @p x row.
 *
 * Unrolls @ref sqEuclideanRowAgainst8Transposed across two adjacent lane groups so each `x[k]`
 * broadcast folds 16 candidate distances per FMA pair. At `L in (8, 16]` with `d <= 8`
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
 * @brief Compute one 8-way squared distance slab against an `(d, W)` transposed candidate
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
 * width; otherwise dispatches @p L scalar @c sqEuclideanRowPtr calls. The batched AVX2 path
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
 * probability proportional to `D(x)`^2 -- the squared distance from each point to its nearest
 * already-chosen centroid. The candidate that yields the smallest resulting sum of squared
 * minimum distances wins.
 *
 * Scratch is private: the candidate pack, the transposed candidate layout, the per-point
 * per-candidate distance cache, the cumulative-distance array, and the per-point running
 * min-squared-distance all live inside the policy. Repeated @c run calls at a stable
 * `(n, d, k)` shape pay no reallocation.
 *
 * @tparam T Element type; @c float or @c double.
 */
template <class T> class GreedyKmppSeeder {
public:
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                "GreedyKmppSeeder<T> requires T to be float or double");

  GreedyKmppSeeder()
      : m_candRows({0, 0}), m_candRowsT({0, 0}), m_candDistSq({0, 0}), m_sweepSums({0}),
        m_minSq({0}), m_distsFlat({0, 0}), m_xNormsSq({0}), m_candNormsSq({0}), m_gemmApArena({0}),
        m_gemmBpArena({0}), m_localScores({0}) {}

  /**
   * @brief Seed @c k centroids from @p X into @p outCentroids.
   *
   * @param X            Data matrix (n x d), contiguous.
   * @param k            Number of centroids to seed (`>= 1`).
   * @param seed         RNG seed; identical seed + `(X, k)` produces identical centroids.
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
    // Per-worker score slab padded to a cache line so adjacent workers do not false-share it
    // while accumulating. An unpadded slab packs several workers onto one line and serialises
    // the sweep on cross-die coherence traffic.
    constexpr std::size_t kScoreSlabFloats = 16; // 64 bytes / sizeof(float)
    const std::size_t scoreSlab =
        ((nLocalTrials + kScoreSlabFloats - 1) / kScoreSlabFloats) * kScoreSlabFloats;
    ensureShape(n, d, nLocalTrials, pool.workerCount());

    math::pcg64 rng;
    rng.seed(seed);

    const T *xData = X.data();
    T *centroidsData = outCentroids.data();
    T *minSq = m_minSq.data();
    T *candRowsData = m_candRows.data();
    T *sweepSums = m_sweepSums.data();

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

    // Every sweep that mutates minSq banks its block's weight into sweepSums, so candidate
    // sampling never needs a separate pass over the array. The block partition is the
    // deterministic sliceLo split of parallelForExactBlocksWithSlot.
    const std::size_t sweepBlocks = detail::greedyKmppSweepBlocks(pool, n, d + 8);

#ifdef CLUSTERING_USE_AVX2
    if constexpr (std::is_same_v<T, float>) {
      // The init sweep and every pick round run inside one persistent-worker plex: workers
      // stay spin-resident across the per-round scoring and refresh passes while the serial
      // glue rides the pre-phase hook, dropping the two fork-joins each round pays in the
      // dispatch loop below. GEMM-scoring shapes keep the dispatch loop -- their scoring is
      // one whole-matrix GEMM, not a range-invocable body -- as do runs too small to
      // amortize the per-phase epoch cost.
      constexpr std::size_t kMinPlexElems = std::size_t{1} << 16;
      if (pool.pool != nullptr && pool.workerCount() > 1 && k >= 2 && !useGemmScoring &&
          (n * d >= kMinPlexElems)) {
        runPlexRounds(X, k, nLocalTrials, scoreSlab, sweepBlocks, rng, pool, outCentroids);
        return;
      }
    }
#endif

    {
      const T *firstRow = centroidsData;
      pool.parallelForExactBlocksWithSlot(
          std::size_t{0}, n, sweepBlocks,
          [&](std::size_t lo, std::size_t hi, std::size_t s) noexcept {
            initSweepBlock(firstRow, xData, d, lo, hi, s);
          });
    }

    if (k == 1) {
      return;
    }

    std::vector<std::size_t> candidates(nLocalTrials, 0);
    std::vector<T> scores(nLocalTrials, T{0});

    for (std::size_t c = 1; c < k; ++c) {
      // The fold visits blocks in slot order, so the total is deterministic no matter which
      // worker refreshed which block.
      T total = T{0};
      for (std::size_t s = 0; s < sweepBlocks; ++s) {
        total += sweepSums[s];
      }

      // Degenerate guard: when every chosen centroid coincides with every remaining point the
      // total collapses to ~0; pick the next centroid uniformly so the routine cannot stall.
      if (!(total > T{0})) {
        const auto pick = static_cast<std::size_t>(math::randUniformU64(rng) % n);
        std::memcpy(centroidsData + (c * d), xData + (pick * d), d * sizeof(T));
        const T *cRow = centroidsData + (c * d);
        pool.parallelForExactBlocksWithSlot(
            std::size_t{0}, n, sweepBlocks,
            [&](std::size_t lo, std::size_t hi, std::size_t s) noexcept {
              refreshSweepBlock(cRow, xData, d, lo, hi, s);
            });
        continue;
      }

      // Draw nLocalTrials candidates by inverse-CDF sampling. An empty or zero-weight block
      // can never straddle a draw in `[0, total)` because the block fold above accumulates in
      // the same order. Identical seed + identical n produces identical candidate sets.
      drawCandidates(rng, total, n, sweepBlocks, nLocalTrials, candidates.data());

      // Pack the L candidate rows into a contiguous (L, d) buffer so the batched scoring kernel
      // can stream x once across L accumulators. The L*d pack is negligible against the n-pass
      // scoring it amortizes.
      packCandidates(xData, d, nLocalTrials, candidates.data());

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
      // (d == 8). The transposed `(d, W)` layout puts the same-feature components of every
      // candidate in consecutive 8-lane YMM registers, so each broadcast-of-x[k] + FMA pair
      // folds 8 (or 16, for the 16-lane unroll) distances at once.
      if constexpr (std::is_same_v<T, float>) {
        if (d > 0 && d <= math::detail::kAvx2Lanes<float>) {
          packCandidatesTransposed(d, nLocalTrials, transposedWidth);
          // Per-worker score accumulators are reused from @ref m_localScores; the candDistSq
          // writes are row-local so partitioning by `i` is aliasing-free. The fan-out width
          // is capped by the sweep's kernel work; a width of one keeps the sweep on the
          // calling thread.
          const std::size_t blocksT = detail::greedyKmppSweepBlocks(pool, n, d * transposedWidth);
          const bool willParallelizeT = blocksT > 1;
          const std::size_t workersT = willParallelizeT ? pool.workerCount() : std::size_t{1};
          T *localScoresT = m_localScores.data();
          zeroScoreSlabs(workersT, scoreSlab);

          if (transposedWidth == 16) {
            if (willParallelizeT) {
              pool.parallelForBlocks(std::size_t{0}, n, blocksT,
                                     [&](std::size_t lo, std::size_t hi) {
                                       const std::size_t w = math::Pool::workerIndex();
                                       scoreTransposed16Range(xData, d, nLocalTrials, lo, hi,
                                                              localScoresT + (w * scoreSlab));
                                     });
            } else {
              scoreTransposed16Range(xData, d, nLocalTrials, std::size_t{0}, n, localScoresT);
            }
          } else if (transposedWidth == 8) {
            if (willParallelizeT) {
              pool.parallelForBlocks(std::size_t{0}, n, blocksT,
                                     [&](std::size_t lo, std::size_t hi) {
                                       const std::size_t w = math::Pool::workerIndex();
                                       scoreTransposed8Range(xData, d, nLocalTrials, lo, hi,
                                                             localScoresT + (w * scoreSlab));
                                     });
            } else {
              scoreTransposed8Range(xData, d, nLocalTrials, std::size_t{0}, n, localScoresT);
            }
          } else {
            // Generic chunked path for L > 16 (very high k). Walk the transposed layout 8 lanes
            // at a time so each chunk stays on the fully unrolled 8-wide kernel.
            (void)candDistRows(n, transposedWidth);
            if (willParallelizeT) {
              pool.parallelForBlocks(
                  std::size_t{0}, n, blocksT, [&](std::size_t lo, std::size_t hi) {
                    const std::size_t w = math::Pool::workerIndex();
                    scoreTransposedChunkedRange(xData, d, nLocalTrials, transposedWidth, lo, hi,
                                                localScoresT + (w * scoreSlab));
                  });
            } else {
              scoreTransposedChunkedRange(xData, d, nLocalTrials, transposedWidth, std::size_t{0},
                                          n, localScoresT);
            }
          }
          foldScoreSlabs(workersT, scoreSlab, nLocalTrials, scores.data());
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
          T *candDistSqData = candDistRows(n, transposedWidth);
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
          const bool willParallelize = pool.shouldParallelize(n, 1024, 2);
          bool scoredViaSoa = false;
#ifdef CLUSTERING_USE_AVX2
          if constexpr (std::is_same_v<T, float>) {
            // SoA 8-row M-tile kernel: streams X AoS through an in-register 8x8 transpose so 8
            // rows' features land in feature-major YMM accumulators, folds L distances per row
            // without per-row horizontal reductions, writes the per-(row, cand) distances to
            // @c outDist, and accumulates min-capped scores. The kernel handles arbitrary row
            // counts, so per-worker row ranges slot in under the same parallel fan-out that
            // feeds the fallback path.
            // Score-only path: skip the (n, L) cand-dist materialization. The winner row
            // is recomputed by the commit-step refresh below, trading L stores per row for
            // one sqEuclideanRowPtr per row at pick time.
            const bool soaEligible = (d >= 8) && (nLocalTrials >= 1) && (nLocalTrials <= 6);
            if (soaEligible) {
              if (willParallelize) {
                const std::size_t workers = pool.workerCount();
                T *localScores = m_localScores.data();
                zeroScoreSlabs(workers, scoreSlab);
                pool.parallelForBlocks(std::size_t{0}, n, pool.stealBlocks(n),
                                       [&](std::size_t lo, std::size_t hi) {
                                         const std::size_t w = math::Pool::workerIndex();
                                         scoreSoaRange(xData, d, nLocalTrials, transposedWidth, lo,
                                                       hi, localScores + (w * scoreSlab));
                                       });
                foldScoreSlabs(workers, scoreSlab, nLocalTrials, scores.data());
              } else {
                scoreSoaRange(xData, d, nLocalTrials, transposedWidth, std::size_t{0}, n,
                              scores.data());
              }
              scoredViaSoa = true;
            }
          }
#endif
          if (!scoredViaSoa) {
            (void)candDistRows(n, transposedWidth);
            if (willParallelize) {
              const std::size_t workers = pool.workerCount();
              T *localScores = m_localScores.data();
              zeroScoreSlabs(workers, scoreSlab);
              pool.parallelForBlocks(std::size_t{0}, n, pool.stealBlocks(n),
                                     [&](std::size_t lo, std::size_t hi) {
                                       const std::size_t w = math::Pool::workerIndex();
                                       scoreScalarRange(xData, d, nLocalTrials, transposedWidth, lo,
                                                        hi, localScores + (w * scoreSlab));
                                     });
              foldScoreSlabs(workers, scoreSlab, nLocalTrials, scores.data());
            } else {
              scoreScalarRange(xData, d, nLocalTrials, transposedWidth, std::size_t{0}, n,
                               scores.data());
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

      // Commit best candidate: copy its row into outCentroids, then refresh @c minSq with a
      // fresh O(n*d) scan against the winner row. We deliberately DO NOT materialize the
      // full (n, L) candidate-distance plane during the score sweep -- at the d=2 envelope
      // its per-row plane write traffic dominated the seeder's runtime. Recomputing one
      // column for the winner trades 1 row-distance call per row against L row-distance
      // writes per row in the score sweep.
      const T *winnerRow = xData + (bestCandidate * d);
      std::memcpy(centroidsData + (c * d), winnerRow, d * sizeof(T));
      pool.parallelForExactBlocksWithSlot(
          std::size_t{0}, n, sweepBlocks,
          [&](std::size_t lo, std::size_t hi, std::size_t s) noexcept {
            refreshSweepBlock(winnerRow, xData, d, lo, hi, s);
          });
    }
  }

private:
  /**
   * @brief Grow the `(n, W)` candidate-distance plane on first use and return its base.
   *
   * Only the generic transposed walk, the scalar scoring fallback, and the GEMM fold write
   * this plane; the register-resident and SoA scoring paths never touch it, so eagerly
   * sizing it in @ref ensureShape would put an untouched @c n-row slab on every fit.
   */
  T *candDistRows(std::size_t n, std::size_t w) {
    if (m_candDistSq.dim(0) != n || m_candDistSq.dim(1) != w) {
      m_candDistSq = NDArray<T, 2, Layout::Contig>({n == 0 ? std::size_t{1} : n, w});
    }
    return m_candDistSq.data();
  }

  /// Init-sweep body for one sweep block: reset the block's @c minSq lanes to `+inf`,
  /// refresh them against @p firstRow, and bank the block's weight into @c m_sweepSums.
  void initSweepBlock(const T *firstRow, const T *xData, std::size_t d, std::size_t lo,
                      std::size_t hi, std::size_t s) noexcept {
    T *minSq = m_minSq.data();
    for (std::size_t i = lo; i < hi; ++i) {
      minSq[i] = std::numeric_limits<T>::infinity();
    }
    m_sweepSums.data()[s] =
        math::detail::refreshMinSqAgainstRow(firstRow, xData + (lo * d), hi - lo, d, minSq + lo);
  }

  /// Refresh-sweep body for one sweep block: fold @p row into the block's @c minSq lanes and
  /// bank the refreshed weight into @c m_sweepSums.
  void refreshSweepBlock(const T *row, const T *xData, std::size_t d, std::size_t lo,
                         std::size_t hi, std::size_t s) noexcept {
    m_sweepSums.data()[s] = math::detail::refreshMinSqAgainstRow(row, xData + (lo * d), hi - lo, d,
                                                                 m_minSq.data() + lo);
  }

  /// Draw candidate indices by inverse-CDF sampling over the banked block sums: locate the
  /// block whose weight straddles each draw, then walk only that block's @c minSq lanes.
  void drawCandidates(math::pcg64 &rng, T total, std::size_t n, std::size_t sweepBlocks,
                      std::size_t nLocalTrials, std::size_t *candidates) noexcept {
    const T *sweepSums = m_sweepSums.data();
    const T *minSq = m_minSq.data();
    for (std::size_t t = 0; t < nLocalTrials; ++t) {
      const T u = math::randUnit<T>(rng) * total;
      std::size_t s = 0;
      T acc = T{0};
      while (s + 1 < sweepBlocks && acc + sweepSums[s] <= u) {
        acc += sweepSums[s];
        ++s;
      }
      candidates[t] = math::detail::inverseCdfPickInRange(minSq, (n * s) / sweepBlocks,
                                                          (n * (s + 1)) / sweepBlocks, u - acc);
    }
  }

  /// Pack the drawn candidate rows into the contiguous `(L, d)` scoring buffer.
  void packCandidates(const T *xData, std::size_t d, std::size_t nLocalTrials,
                      const std::size_t *candidates) noexcept {
    T *candRowsData = m_candRows.data();
    for (std::size_t t = 0; t < nLocalTrials; ++t) {
      std::memcpy(candRowsData + (t * d), xData + (candidates[t] * d), d * sizeof(T));
    }
  }

  /// Transpose the candidate pack into the zero-padded `(d, W)` feature-major layout the
  /// low-d transposed kernels stream.
  void packCandidatesTransposed(std::size_t d, std::size_t nLocalTrials,
                                std::size_t transposedWidth) noexcept {
    const T *candRowsData = m_candRows.data();
    T *candRowsTData = m_candRowsT.data();
    for (std::size_t kk = 0; kk < d; ++kk) {
      T *dstK = candRowsTData + (kk * transposedWidth);
      for (std::size_t t = 0; t < nLocalTrials; ++t) {
        dstK[t] = candRowsData[(t * d) + kk];
      }
      for (std::size_t t = nLocalTrials; t < transposedWidth; ++t) {
        dstK[t] = T{0};
      }
    }
  }

  void zeroScoreSlabs(std::size_t slabs, std::size_t scoreSlab) noexcept {
    T *localScores = m_localScores.data();
    for (std::size_t e = 0; e < slabs * scoreSlab; ++e) {
      localScores[e] = T{0};
    }
  }

  /// Fold the per-slot score slabs into @p scores in ascending-slot order so the reduced
  /// totals do not depend on which worker ran which range.
  void foldScoreSlabs(std::size_t slabs, std::size_t scoreSlab, std::size_t nLocalTrials,
                      T *scores) const noexcept {
    const T *localScores = m_localScores.data();
    for (std::size_t w = 0; w < slabs; ++w) {
      const T *row = localScores + (w * scoreSlab);
      for (std::size_t t = 0; t < nLocalTrials; ++t) {
        scores[t] += row[t];
      }
    }
  }

  /// Batched-kernel scoring for rows `[lo, hi)`: @c L distances per row against the candidate
  /// pack, min-capped sums accumulated into @p dst, distances materialized into the pre-grown
  /// candidate-distance plane.
  void scoreScalarRange(const T *xData, std::size_t d, std::size_t nLocalTrials,
                        std::size_t transposedWidth, std::size_t lo, std::size_t hi,
                        T *dst) noexcept {
    const T *candRowsData = m_candRows.data();
    const T *minSq = m_minSq.data();
    T *candDistSqData = m_candDistSq.data();
    std::array<T, 32> distRowLocal{};
    for (std::size_t i = lo; i < hi; ++i) {
      const T *xi = xData + (i * d);
      const T mi = minSq[i];
      detail::sqEuclideanRowToBatch<T>(xi, candRowsData, nLocalTrials, d, distRowLocal.data());
      T *dstRow = candDistSqData + (i * transposedWidth);
      for (std::size_t t = 0; t < nLocalTrials; ++t) {
        dstRow[t] = distRowLocal[t];
        dst[t] += (distRowLocal[t] < mi) ? distRowLocal[t] : mi;
      }
    }
  }

#ifdef CLUSTERING_USE_AVX2
  /// 16-lane transposed scoring for rows `[lo, hi)`: two register-resident 8-lane
  /// accumulators, one spill into @p dst per range instead of one per row.
  void scoreTransposed16Range(const T *xData, std::size_t d, std::size_t nLocalTrials,
                              std::size_t lo, std::size_t hi, T *dst) noexcept {
    const T *minSq = m_minSq.data();
    const T *candRowsTData = m_candRowsT.data();
    __m256 scoresLoAcc = _mm256_setzero_ps();
    __m256 scoresHiAcc = _mm256_setzero_ps();
    for (std::size_t i = lo; i < hi; ++i) {
      const float *xi = xData + (i * d);
      const __m256 miVec = _mm256_set1_ps(minSq[i]);
      const auto [dLo, dHi] = detail::sqEuclideanRowAgainst16TransposedReg(xi, candRowsTData, d);
      scoresLoAcc = _mm256_add_ps(scoresLoAcc, _mm256_min_ps(dLo, miVec));
      scoresHiAcc = _mm256_add_ps(scoresHiAcc, _mm256_min_ps(dHi, miVec));
    }
    std::array<float, 16> tmp{};
    _mm256_storeu_ps(tmp.data(), scoresLoAcc);
    _mm256_storeu_ps(tmp.data() + 8, scoresHiAcc);
    for (std::size_t t = 0; t < nLocalTrials; ++t) {
      dst[t] += tmp[t];
    }
  }

  /// 8-lane sibling of @ref scoreTransposed16Range.
  void scoreTransposed8Range(const T *xData, std::size_t d, std::size_t nLocalTrials,
                             std::size_t lo, std::size_t hi, T *dst) noexcept {
    const T *minSq = m_minSq.data();
    const T *candRowsTData = m_candRowsT.data();
    __m256 scoresAcc = _mm256_setzero_ps();
    for (std::size_t i = lo; i < hi; ++i) {
      const float *xi = xData + (i * d);
      const __m256 miVec = _mm256_set1_ps(minSq[i]);
      const __m256 dv = detail::sqEuclideanRowAgainst8TransposedReg(xi, candRowsTData, d);
      scoresAcc = _mm256_add_ps(scoresAcc, _mm256_min_ps(dv, miVec));
    }
    std::array<float, 8> tmp{};
    _mm256_storeu_ps(tmp.data(), scoresAcc);
    for (std::size_t t = 0; t < nLocalTrials; ++t) {
      dst[t] += tmp[t];
    }
  }

  /// Chunked transposed scoring for `L > 16`: slide an 8-wide window across the `(d, W)`
  /// pack, materializing each row's distances into the pre-grown candidate-distance plane.
  void scoreTransposedChunkedRange(const T *xData, std::size_t d, std::size_t nLocalTrials,
                                   std::size_t transposedWidth, std::size_t lo, std::size_t hi,
                                   T *dst) noexcept {
    const T *minSq = m_minSq.data();
    const T *candRowsTData = m_candRowsT.data();
    T *candDistSqData = m_candDistSq.data();
    for (std::size_t i = lo; i < hi; ++i) {
      const float *xi = xData + (i * d);
      const float mi = minSq[i];
      float *dstRow = candDistSqData + (i * transposedWidth);
      for (std::size_t base = 0; base < transposedWidth; base += 8) {
        detail::sqEuclideanRowAgainst8TransposedStrided(xi, candRowsTData + base, d,
                                                        transposedWidth, dstRow + base);
      }
      for (std::size_t t = 0; t < nLocalTrials; ++t) {
        dst[t] += (dstRow[t] < mi) ? dstRow[t] : mi;
      }
    }
  }

  /// SoA M-tile scoring for rows `[lo, hi)`; score-only, the winner column is recomputed at
  /// pick time.
  void scoreSoaRange(const T *xData, std::size_t d, std::size_t nLocalTrials,
                     std::size_t transposedWidth, std::size_t lo, std::size_t hi, T *dst) noexcept {
    const std::size_t rangeN = hi - lo;
    const float *xSlice = xData + (lo * d);
    const float *minSlice = m_minSq.data() + lo;
    const float *candRowsData = m_candRows.data();
    switch (nLocalTrials) {
    case 1:
      math::detail::kmppScoreSoaRowsAvx2F32<1, /*WriteOutDist=*/false>(
          xSlice, rangeN, d, candRowsData, minSlice, nullptr, transposedWidth, dst);
      break;
    case 2:
      math::detail::kmppScoreSoaRowsAvx2F32<2, /*WriteOutDist=*/false>(
          xSlice, rangeN, d, candRowsData, minSlice, nullptr, transposedWidth, dst);
      break;
    case 3:
      math::detail::kmppScoreSoaRowsAvx2F32<3, /*WriteOutDist=*/false>(
          xSlice, rangeN, d, candRowsData, minSlice, nullptr, transposedWidth, dst);
      break;
    case 4:
      math::detail::kmppScoreSoaRowsAvx2F32<4, /*WriteOutDist=*/false>(
          xSlice, rangeN, d, candRowsData, minSlice, nullptr, transposedWidth, dst);
      break;
    case 5:
      math::detail::kmppScoreSoaRowsAvx2F32<5, /*WriteOutDist=*/false>(
          xSlice, rangeN, d, candRowsData, minSlice, nullptr, transposedWidth, dst);
      break;
    case 6:
      math::detail::kmppScoreSoaRowsAvx2F32<6, /*WriteOutDist=*/false>(
          xSlice, rangeN, d, candRowsData, minSlice, nullptr, transposedWidth, dst);
      break;
    default:
      break;
    }
  }

  /// Scoring kernel families whose bodies take arbitrary row ranges; selected once per run,
  /// dispatched per plex scoring phase.
  enum class ScoreKernel : std::uint8_t {
    kTransposed16,
    kTransposed8,
    kTransposedChunked,
    kSoa,
    kScalar
  };

  /// Mirror of the dispatch loop's per-round kernel selection, minus the GEMM family whose
  /// scoring runs one whole-matrix GEMM instead of a range-invocable body.
  [[nodiscard]] static ScoreKernel pickScoreKernel(std::size_t d,
                                                   std::size_t nLocalTrials) noexcept {
    if (d > 0 && d <= math::detail::kAvx2Lanes<float>) {
      const std::size_t w = detail::greedyKmppTransposedWidth(nLocalTrials);
      if (w == 8) {
        return ScoreKernel::kTransposed8;
      }
      if (w == 16) {
        return ScoreKernel::kTransposed16;
      }
      return ScoreKernel::kTransposedChunked;
    }
    if (d >= 8 && nLocalTrials >= 1 && nLocalTrials <= 6) {
      return ScoreKernel::kSoa;
    }
    return ScoreKernel::kScalar;
  }

  /// Route rows `[lo, hi)` to the selected scoring kernel, accumulating into @p dst.
  void scoreRange(ScoreKernel kernel, const T *xData, std::size_t d, std::size_t nLocalTrials,
                  std::size_t transposedWidth, std::size_t lo, std::size_t hi, T *dst) noexcept {
    switch (kernel) {
    case ScoreKernel::kTransposed16:
      scoreTransposed16Range(xData, d, nLocalTrials, lo, hi, dst);
      break;
    case ScoreKernel::kTransposed8:
      scoreTransposed8Range(xData, d, nLocalTrials, lo, hi, dst);
      break;
    case ScoreKernel::kTransposedChunked:
      scoreTransposedChunkedRange(xData, d, nLocalTrials, transposedWidth, lo, hi, dst);
      break;
    case ScoreKernel::kSoa:
      scoreSoaRange(xData, d, nLocalTrials, transposedWidth, lo, hi, dst);
      break;
    case ScoreKernel::kScalar:
      scoreScalarRange(xData, d, nLocalTrials, transposedWidth, lo, hi, dst);
      break;
    }
  }

  /**
   * @brief Plex-driven round loop for the range-invocable scoring families.
   *
   * One @c runPlex dispatch drives the init sweep plus all `k - 1` picks as `1 + 2*(k-1)`
   * phases over the sweep-block partition, so workers stay spin-resident across rounds
   * instead of paying two fork-joins per pick. Odd phases score the round's candidates into
   * per-slot slabs over the slot's sweep-block row range; even phases refresh @c m_minSq
   * against the committed row and bank the block sums exactly as the dispatch loop does. The
   * serial glue (block-sum fold, degenerate check, candidate draws, pack, winner pick) rides
   * the pre-phase hook on the producer with happens-before to every worker's phase body.
   * RNG consumption matches the dispatch loop draw for draw, including degenerate rounds,
   * which consume one uniform pick and no-op their scoring phase.
   */
  void runPlexRounds(const NDArray<T, 2, Layout::Contig> &X, std::size_t k,
                     std::size_t nLocalTrials, std::size_t scoreSlab, std::size_t sweepBlocks,
                     math::pcg64 &rng, math::Pool pool,
                     NDArray<T, 2, Layout::Contig> &outCentroids) {
    const std::size_t n = X.dim(0);
    const std::size_t d = X.dim(1);
    const T *xData = X.data();
    T *centroidsData = outCentroids.data();
    const T *sweepSums = m_sweepSums.data();
    const std::size_t workers = pool.workerCount();
    const std::size_t transposedWidth = detail::greedyKmppTransposedWidth(nLocalTrials);
    const ScoreKernel kernel = pickScoreKernel(d, nLocalTrials);
    auto sweepLo = [n, sweepBlocks](std::size_t s) noexcept { return (n * s) / sweepBlocks; };

    constexpr std::size_t kMaxLocalTrials = 32;
    CLUSTERING_ALWAYS_ASSERT(nLocalTrials <= kMaxLocalTrials);

    // Plane-writing kernels grow the (n, W) plane before workers go plex-resident.
    if (kernel == ScoreKernel::kTransposedChunked || kernel == ScoreKernel::kScalar) {
      (void)candDistRows(n, transposedWidth);
    }

    std::vector<std::size_t> candidates(nLocalTrials, 0);
    std::vector<T> scores(nLocalTrials, T{0});
    const T *refreshRow = centroidsData; // Phase 0 sweeps against the first centroid.
    bool roundDegenerate = false;

    auto prePhase = [&](std::size_t phaseIdx) noexcept {
      if (phaseIdx == 0) {
        return;
      }
      const std::size_t c = (phaseIdx + 1) / 2;
      if ((phaseIdx & 1U) != 0) {
        // Scoring glue: fold the banked block sums, then either record the degenerate
        // uniform pick or draw and pack this round's candidates.
        T total = T{0};
        for (std::size_t s = 0; s < sweepBlocks; ++s) {
          total += sweepSums[s];
        }
        roundDegenerate = !(total > T{0});
        if (roundDegenerate) {
          const auto pick = static_cast<std::size_t>(math::randUniformU64(rng) % n);
          std::memcpy(centroidsData + (c * d), xData + (pick * d), d * sizeof(T));
          refreshRow = centroidsData + (c * d);
          return;
        }
        drawCandidates(rng, total, n, sweepBlocks, nLocalTrials, candidates.data());
        packCandidates(xData, d, nLocalTrials, candidates.data());
        if (kernel != ScoreKernel::kSoa && kernel != ScoreKernel::kScalar) {
          packCandidatesTransposed(d, nLocalTrials, transposedWidth);
        }
        zeroScoreSlabs(workers, scoreSlab);
        return;
      }
      // Refresh glue: fold the slot slabs, pick the winner, and stage its row for the sweep.
      if (roundDegenerate) {
        return;
      }
      for (std::size_t t = 0; t < nLocalTrials; ++t) {
        scores[t] = T{0};
      }
      foldScoreSlabs(workers, scoreSlab, nLocalTrials, scores.data());
      std::size_t bestT = 0;
      T bestScore = scores[0];
      for (std::size_t t = 1; t < nLocalTrials; ++t) {
        if (scores[t] < bestScore) {
          bestScore = scores[t];
          bestT = t;
        }
      }
      const T *winnerRow = xData + (candidates[bestT] * d);
      std::memcpy(centroidsData + (c * d), winnerRow, d * sizeof(T));
      refreshRow = winnerRow;
    };

    auto phase = [&](std::size_t phaseIdx, std::uint32_t slot, std::size_t lo, std::size_t hi,
                     void * /*tlsArena*/ = nullptr) noexcept {
      if (lo >= hi) {
        return;
      }
      if (phaseIdx == 0) {
        for (std::size_t s = lo; s < hi; ++s) {
          initSweepBlock(refreshRow, xData, d, sweepLo(s), sweepLo(s + 1), s);
        }
        return;
      }
      if ((phaseIdx & 1U) != 0) {
        if (roundDegenerate) {
          return;
        }
        scoreRange(kernel, xData, d, nLocalTrials, transposedWidth, sweepLo(lo), sweepLo(hi),
                   m_localScores.data() + (static_cast<std::size_t>(slot) * scoreSlab));
        return;
      }
      for (std::size_t s = lo; s < hi; ++s) {
        refreshSweepBlock(refreshRow, xData, d, sweepLo(s), sweepLo(s + 1), s);
      }
    };

    pool.parallelRunPlex<citor::HintsDefaults>(1 + (2 * (k - 1)), sweepBlocks, std::move(phase),
                                               std::move(prePhase));
  }
#endif // CLUSTERING_USE_AVX2

  void ensureShape(std::size_t n, std::size_t d, std::size_t L, std::size_t workers) {
    const std::size_t w = detail::greedyKmppTransposedWidth(L == 0 ? std::size_t{1} : L);
    if (m_candRows.dim(0) != L || m_candRows.dim(1) != d) {
      m_candRows = NDArray<T, 2, Layout::Contig>({L, d});
    }
    if (m_candRowsT.dim(0) != d || m_candRowsT.dim(1) != w) {
      m_candRowsT = NDArray<T, 2, Layout::Contig>({d == 0 ? std::size_t{1} : d, w});
    }
    const std::size_t sweepSlots = std::max<std::size_t>(workers, std::size_t{1}) * 8;
    if (m_sweepSums.dim(0) != sweepSlots) {
      m_sweepSums = NDArray<T, 1>({sweepSlots});
    }
    if (m_minSq.dim(0) != n) {
      m_minSq = NDArray<T, 1>({n});
    }
    // GEMM-scoring-only scratch (distsFlat, xNormsSq, candNormsSq, gemmApArena, gemmBpArena).
    // The GEMM path fires at `d >= 32` && L >= kKernelNr<float>; outside that envelope we keep
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
    // slice of the A-pack arena at offset `(worker * kMc * kKc)`. Sizing the arena for just
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
    const std::size_t scoreSlab = ((lSafe + 15U) / 16U) * 16U; // cache-line-padded per-worker slab
    const std::size_t lsLen = workersClamped * scoreSlab;
    if (m_localScores.dim(0) != lsLen) {
      m_localScores = NDArray<T, 1>({lsLen});
    }
  }

  /// Packed candidate rows: shape `(L, d)`. Reused across all @c k-1 outer picks within one run.
  NDArray<T, 2, Layout::Contig> m_candRows;
  /// Transposed candidate rows: shape `(d, W)` where @c W = greedyKmppTransposedWidth(L).
  /// Padded to a multiple of the 8-wide YMM lane so the transposed kernel iterates over
  /// fixed-width chunks; lanes past @c L are zero-filled and the scoring loop reads only the
  /// first @c L lanes.
  NDArray<T, 2, Layout::Contig> m_candRowsT;
  /// Per-outer-iteration cache of candidate distances: shape `(n, W)` matching the transposed
  /// pack. Grown lazily through @ref candDistRows by the scoring paths that materialize it.
  NDArray<T, 2, Layout::Contig> m_candDistSq;
  /// Per-sweep-block sums of the refreshed @c minSq, banked by every sweep that mutates the
  /// array. Sampling folds these to the total and walks only the straddling block.
  NDArray<T, 1> m_sweepSums;
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
