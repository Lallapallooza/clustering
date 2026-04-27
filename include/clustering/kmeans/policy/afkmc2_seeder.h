#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>

#ifdef CLUSTERING_USE_AVX2
#include <immintrin.h>
#endif

#include "clustering/always_assert.h"
#include "clustering/math/detail/avx2_helpers.h"
#include "clustering/math/detail/avx2_reductions.h"
#include "clustering/math/detail/sq_distances_block.h"
#include "clustering/math/detail/sq_distances_tile.h"
#include "clustering/math/pairwise.h"
#include "clustering/math/rng.h"
#include "clustering/math/thread.h"
#include "clustering/ndarray.h"

namespace clustering::kmeans {

using math::detail::affineInPlaceAvx2;
using math::detail::fillAvx2;
using math::detail::minDistBatchedAvx2F32;
using math::detail::scaleAvx2;
using math::detail::sqDistancesAosBlock;
using math::detail::sqEuclideanRowPtr;
using math::detail::sumReduceAvx2;

/**
 * @brief AFK-MC2 seeder (Bachem, Lucic, Hassani, Krause, NeurIPS 2016).
 *
 * Sublinear-in-n MCMC approximation to k-means++: draws the first centroid uniformly, builds a
 * length-@c n proposal distribution `q(i)` = 0.5 * D(x_i, c_1)^2 / sum_D2 + 0.5 * 1/n, and then
 * for each remaining centroid runs a Markov chain of length @c m that accepts a proposal with
 * probability `min(1, proposed_weight / current_weight)` where the weight is the squared
 * distance to the current centroid set divided by the proposal density.
 *
 * Implementation specifics. (1) The proposal distribution @c q is sampled in `O(1)` per draw
 * via a Walker alias table built once per `(n, k)` shape on the post-transform @c q. (2) The
 * `m+1` chain proposals at each centroid level are pre-sampled into a single batch and their
 * distances to the chosen-centroid block are computed by a 4-query x 2-centroid AVX2 tile
 * kernel (`minDistBatchedAvx2F32`); the chain then walks the batch with `O(1)` accept/reject
 * arithmetic per step. (3) The q preprocessing (squared-distance scan plus the affine
 * transform plus the alias-bucket partition) fans out across `pool` when the workload exceeds
 * the per-worker amortisation gate; the chain itself remains strictly serial.
 *
 * Same-seed determinism. Successive runs at identical `(seed, n, d, k, m)` produce
 * bit-identical centroids regardless of @p pool worker count: the alias table is built
 * deterministically from the post-transform @c q, the chain pre-samples PRNG draws in a
 * fixed order, and the tile kernel uses a deterministic FMA reduction tree.
 *
 * Degenerate guard: when all points coincide with the first centroid (`sum_D2 == 0`) the
 * proposal collapses to uniform `q(i)` = 1/n so the chain remains ergodic.
 *
 * The chain's log-k approximation bound degrades at small @c k: below @c k = @ref
 * AfkMc2Seeder::kFloor the bound is too loose to beat greedy k-means++, and callers at that
 * regime should pin @ref GreedyKmppSeeder (directly or via @ref AutoSeeder, which picks it by
 * shape).
 *
 * @tparam T Element type; @c float or @c double.
 */
template <class T> class AfkMc2Seeder {
public:
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                "AfkMc2Seeder<T> requires T to be float or double");

#ifdef CLUSTERING_KMEANS_AFKMC2_K_FLOOR
  /**
   * @brief Minimum @c k below which the AFK-MC2 chain's log-k approximation bound is too loose
   *        to beat @ref GreedyKmppSeeder. Exposed as a shape threshold for @ref AutoSeeder's
   *        dispatcher; not checked inside @ref run.
   *
   * Override with @c -DCLUSTERING_KMEANS_AFKMC2_K_FLOOR=<value>.
   */
  static constexpr std::size_t kFloor = CLUSTERING_KMEANS_AFKMC2_K_FLOOR;
#else
  /// Minimum @c k below which the AFK-MC2 chain's log-k bound is too loose to win.
  static constexpr std::size_t kFloor = 100;
#endif

#ifdef CLUSTERING_KMEANS_AFKMC2_CHAIN_LENGTH
  /**
   * @brief Markov chain length per centroid pick. Bachem 2016 reports @c m=200 as the sweet
   *        spot for the log-k approximation guarantee.
   *
   * Override with @c -DCLUSTERING_KMEANS_AFKMC2_CHAIN_LENGTH=<value>; values below a few dozen
   * trade the provable bound for faster seeding, values above 200 amortize into larger @c n
   * regimes where the chain's sublinear-in-n behavior is the dominant cost.
   */
  static constexpr std::size_t chainLengthDefault = CLUSTERING_KMEANS_AFKMC2_CHAIN_LENGTH;
#else
  /// Default Markov-chain length per centroid pick.
  static constexpr std::size_t chainLengthDefault = 200;
#endif

  AfkMc2Seeder()
      : m_q({0}), m_aliasProb({0}), m_aliasIdx({0}), m_aliasSmall({0}), m_aliasLarge({0}),
        m_yIdxBatch({0}), m_uBatch({0}), m_yQBatch({0}), m_yDistBatch({0}) {}

  /**
   * @brief Seed @c k centroids from @p X into @p outCentroids.
   *
   * @param X            Data matrix (n x d), contiguous.
   * @param k            Number of centroids to seed.
   * @param seed         RNG seed; identical seed + `(X, k)` produces identical centroids.
   * @param pool         Parallelism injection (preprocessing sweep only).
   * @param outCentroids Output centroid matrix (k x d), contiguous; populated in row order.
   */
  void run(const NDArray<T, 2, Layout::Contig> &X, std::size_t k, std::uint64_t seed,
           math::Pool pool, NDArray<T, 2, Layout::Contig> &outCentroids) {
    runChain(X, k, chainLengthDefault, seed, pool, outCentroids);
  }

private:
  void runChain(const NDArray<T, 2, Layout::Contig> &X, std::size_t k, std::size_t m,
                std::uint64_t seed, math::Pool pool, NDArray<T, 2, Layout::Contig> &outCentroids) {
    const std::size_t n = X.dim(0);
    const std::size_t d = X.dim(1);

    CLUSTERING_ALWAYS_ASSERT(outCentroids.isMutable());
    CLUSTERING_ALWAYS_ASSERT(outCentroids.dim(0) == k);
    CLUSTERING_ALWAYS_ASSERT(outCentroids.dim(1) == d);
    CLUSTERING_ALWAYS_ASSERT(k >= 1);
    CLUSTERING_ALWAYS_ASSERT(n >= k);
    CLUSTERING_ALWAYS_ASSERT(m >= 1);

    ensureShape(n, k, m);

    math::pcg64 rng;
    rng.seed(seed);

    const T *xData = X.data();
    T *centroidsData = outCentroids.data();
    T *qData = m_q.data();

    // Step 1: first centroid uniformly.
    const auto first = static_cast<std::size_t>(math::randUniformU64(rng) % n);
    std::memcpy(centroidsData, xData + (first * d), d * sizeof(T));

    if (k == 1) {
      return;
    }

    // Step 2: q-precompute. Squared distance from each point to the first centroid drives the
    // data-proximal half of the proposal density; the 1/n floor guarantees ergodicity even at
    // sumD2==0. Distance scan + sumReduce optionally fan out across pool when the per-worker
    // op budget amortises the spawn cost.
    const T *firstRow = centroidsData;
    const std::size_t qOps = n * d;
    if (pool.shouldParallelizeWork(qOps, /*minOpsPerWorker=*/std::size_t{1} << 17)) {
      qPrecomputeParallel(xData, firstRow, n, d, qData, pool);
    } else {
      sqDistancesAosBlock<T>(firstRow, xData, n, d, qData);
    }

    T sumD2;
    if (pool.shouldParallelizeWork(n, /*minOpsPerWorker=*/std::size_t{1} << 17)) {
      sumD2 = sumReduceParallel(qData, n, pool);
    } else {
      sumD2 = sumReduceAvx2(qData, n);
    }

    const T invN = T{1} / static_cast<T>(n);
    if (sumD2 > T{0}) {
      const T invSum = T{1} / sumD2;
      affineInPlaceAvx2(qData, n, T{0.5} * invSum, T{0.5} * invN);
    } else {
      // Degenerate: every point coincides with c_1. Fall back to uniform so the chain stays
      // ergodic over the point set.
      fillAvx2(qData, n, invN);
    }

    // Walker alias table samples in `O(1)` but its build pass has random writes into
    // `prob[l]` whose cost grows with `n` once `prob` overflows L2; the prefix-sum builder is
    // a single sequential pass. They break even when the chain's per-call sample budget
    // amortises the alias build's `O(n)` random-access overhead. The shape gate below routes
    // small-`n` workloads to alias and large-`n` workloads to prefix-sum + binary search.
    const std::size_t chainSamples = (k - 1) * (m + 1);
    const bool useAlias = chainSamples * 5 > n;
    if (useAlias) {
      buildAliasTable(qData, n);
    } else {
      buildPrefixSum(qData, m_aliasProb.data(), n);
    }

    // Step 3: for each remaining centroid, pre-sample the chain's `m+1` proposals plus their
    // accept-uniforms in one PRNG-deterministic order, then dispatch the proposal-vs-chosen
    // distance scan to the 4q x 2c tile kernel. The chain walk consumes the precomputed
    // distances and uniforms with `O(1)` arithmetic per step.
    std::size_t *yIdxBatch = m_yIdxBatch.data();
    T *uBatch = m_uBatch.data();
    T *yQBatch = m_yQBatch.data();
    T *yDistBatch = m_yDistBatch.data();

    for (std::size_t c = 1; c < k; ++c) {
      // Pre-sample m+1 proposals (yIdxBatch[0] is the chain's initial xIdx).
      if (useAlias) {
        for (std::size_t t = 0; t <= m; ++t) {
          yIdxBatch[t] = sampleFromAlias(rng, n);
        }
      } else {
        for (std::size_t t = 0; t <= m; ++t) {
          yIdxBatch[t] = sampleFromPrefix(rng, m_aliasProb.data(), n);
        }
      }
      for (std::size_t t = 0; t < m; ++t) {
        uBatch[t] = math::randUnit<T>(rng);
      }

      // Compute the per-proposal min distance to the chosen-centroid block via the 4q x 2c
      // tile kernel. Centroid rows are loaded once per query block of 4 instead of once per
      // chain step, cutting centroid-side load traffic by `>= 4x`.
      minDistBatchedFromIdx(xData, d, yIdxBatch, m + 1, centroidsData, c, yDistBatch);

      // Gather the per-proposal q values via index lookup.
      for (std::size_t t = 0; t <= m; ++t) {
        yQBatch[t] = qData[yIdxBatch[t]];
      }

      // Walk the chain serially. Acceptance ratio `(yDist / yQ) / (xDist / xQ)` reordered as
      // `yDist * xQ` vs `xDist * yQ` to skip the division. Draw u every step from the
      // precomputed batch so the PRNG sequence depends only on `(seed, n, k, m)` and never on
      // branch outcomes inside the chain.
      std::size_t xIdx = yIdxBatch[0];
      T xDist = yDistBatch[0];
      T xQ = yQBatch[0];
      for (std::size_t step = 0; step < m; ++step) {
        const T yDist = yDistBatch[step + 1];
        const T yQ = yQBatch[step + 1];
        const T u = uBatch[step];

        const T numer = yDist * xQ;
        const T denom = xDist * yQ;
        const bool accept = (denom <= T{0}) || ((u * denom) < numer);

        if (accept) {
          xIdx = yIdxBatch[step + 1];
          xDist = yDist;
          xQ = yQ;
        }
      }

      std::memcpy(centroidsData + (c * d), xData + (xIdx * d), d * sizeof(T));
    }
  }

  void ensureShape(std::size_t n, std::size_t k, std::size_t m) {
    if (m_q.dim(0) != n) {
      m_q = NDArray<T, 1>({n});
    }
    if (m_aliasProb.dim(0) != n) {
      m_aliasProb = NDArray<T, 1>({n});
    }
    if (m_aliasIdx.dim(0) != n) {
      m_aliasIdx = NDArray<std::size_t, 1>({n});
    }
    if (m_aliasSmall.dim(0) != n) {
      m_aliasSmall = NDArray<std::size_t, 1>({n});
    }
    if (m_aliasLarge.dim(0) != n) {
      m_aliasLarge = NDArray<std::size_t, 1>({n});
    }
    if (m_yIdxBatch.dim(0) != m + 1) {
      m_yIdxBatch = NDArray<std::size_t, 1>({m + 1});
    }
    if (m_uBatch.dim(0) != m) {
      m_uBatch = NDArray<T, 1>({m});
    }
    if (m_yQBatch.dim(0) != m + 1) {
      m_yQBatch = NDArray<T, 1>({m + 1});
    }
    if (m_yDistBatch.dim(0) != m + 1) {
      m_yDistBatch = NDArray<T, 1>({m + 1});
    }
    (void)k;
  }

  /// Walker alias table builder. After this call: `m_aliasProb[i]` is the probability of
  /// accepting bucket @c i, and `m_aliasIdx[i]` is the alternate bucket if accept fails.
  /// Sampling: pick @c i uniformly from `[0, n)`; draw @c u in `[0, 1)`; return @c i if
  /// `u < m_aliasProb[i]` else `m_aliasIdx[i]`.
  void buildAliasTable(const T *qSrc, std::size_t n) noexcept {
    T *prob = m_aliasProb.data();
    std::size_t *alias = m_aliasIdx.data();
    std::size_t *smallStack = m_aliasSmall.data();
    std::size_t *largeStack = m_aliasLarge.data();

    // Scale q so each bucket's "expected mass" is n. After scaling, prob[i] in [0, n] maps
    // directly to acceptance probability after the partition step rescales by 1.
    const T total = sumReduceAvx2(qSrc, n);
    CLUSTERING_ALWAYS_ASSERT(total > T{0});
    const T scale = static_cast<T>(n) / total;
    scaleAvx2(qSrc, n, scale, prob);

    std::size_t numSmall = 0;
    std::size_t numLarge = 0;
    for (std::size_t i = 0; i < n; ++i) {
      if (prob[i] < T{1}) {
        smallStack[numSmall++] = i;
      } else {
        largeStack[numLarge++] = i;
      }
    }

    while (numSmall > 0 && numLarge > 0) {
      const std::size_t s = smallStack[--numSmall];
      const std::size_t l = largeStack[--numLarge];
      // `prob[s]` already in `[0, 1)`; it is the acceptance probability for bucket s. The
      // alias bucket is l, which absorbs the residual mass `1 - prob[s]`.
      alias[s] = l;
      const T residual = prob[l] - (T{1} - prob[s]);
      prob[l] = residual;
      if (residual < T{1}) {
        smallStack[numSmall++] = l;
      } else {
        largeStack[numLarge++] = l;
      }
    }
    // Drain any remaining buckets due to FP rounding; their acceptance probability is 1.
    while (numLarge > 0) {
      const std::size_t l = largeStack[--numLarge];
      prob[l] = T{1};
      alias[l] = l;
    }
    while (numSmall > 0) {
      const std::size_t s = smallStack[--numSmall];
      prob[s] = T{1};
      alias[s] = s;
    }
  }

  /// O(1) draw from the alias table.
  [[gnu::always_inline]] std::size_t sampleFromAlias(math::pcg64 &rng, std::size_t n) noexcept {
    const std::uint64_t r = math::randUniformU64(rng);
    const auto i = static_cast<std::size_t>(r % static_cast<std::uint64_t>(n));
    const T u = math::randUnit<T>(rng);
    return (u < m_aliasProb.data()[i]) ? i : m_aliasIdx.data()[i];
  }

  /// Inclusive prefix sum into @p qCumOut so inverse-CDF binary search can sample.
  void buildPrefixSum(const T *qSrc, T *qCumOut, std::size_t n) noexcept {
    T running = T{0};
    for (std::size_t i = 0; i < n; ++i) {
      running += qSrc[i];
      qCumOut[i] = running;
    }
  }

  /// Inverse-CDF binary search over @p qCum (length @p n, monotonically non-decreasing).
  /// Total mass is `qCum[n-1]`.
  [[gnu::always_inline]] std::size_t sampleFromPrefix(math::pcg64 &rng, const T *qCum,
                                                      std::size_t n) noexcept {
    const T total = qCum[n - 1];
    const T u = math::randUnit<T>(rng) * total;
    std::size_t lo = 0;
    std::size_t hi = n;
    while (lo < hi) {
      const std::size_t mid = lo + ((hi - lo) / 2);
      if (qCum[mid] > u) {
        hi = mid;
      } else {
        lo = mid + 1;
      }
    }
    return lo < n ? lo : n - 1;
  }

  /// Routes the batched min-distance scan to the AVX2 tile kernel for `T == float`; falls
  /// back to a per-query @ref sqDistancesAosBlock + scalar min for `T == double`.
  [[gnu::always_inline]] void minDistBatchedFromIdx(const T *xData, std::size_t d,
                                                    const std::size_t *yIdx, std::size_t qCount,
                                                    const T *centroids, std::size_t cCount,
                                                    T *out) noexcept {
#ifdef CLUSTERING_USE_AVX2
    if constexpr (std::is_same_v<T, float>) {
      minDistBatchedAvx2F32(xData, d, yIdx, qCount, centroids, cCount, out);
      return;
    }
#endif
    for (std::size_t t = 0; t < qCount; ++t) {
      const T *qrow = xData + (yIdx[t] * d);
      T best = std::numeric_limits<T>::infinity();
      // Block of 4 to amortise the hsum.
      alignas(16) std::array<T, 4> blockOut{};
      std::size_t j = 0;
      for (; j + 4 <= cCount; j += 4) {
        sqDistancesAosBlock<T>(qrow, centroids + (j * d), 4, d, blockOut.data());
        for (std::size_t r = 0; r < 4; ++r) {
          if (blockOut[r] < best) {
            best = blockOut[r];
          }
        }
      }
      for (; j < cCount; ++j) {
        const T dsq = sqEuclideanRowPtr(qrow, centroids + (j * d), d);
        if (dsq < best) {
          best = dsq;
        }
      }
      out[t] = best;
    }
  }

  /// Pool fan-out for the q-precompute distance scan. Each worker processes a contiguous slice
  /// of the input rows so the centroid load (single firstRow) remains broadcast-friendly per
  /// worker and the only synchronisation is the single barrier after fan-out.
  void qPrecomputeParallel(const T *xData, const T *firstRow, std::size_t n, std::size_t d,
                           T *qData, math::Pool pool) noexcept {
    pool.parallelForExactBlocks(std::size_t{0}, n, pool.workerCount(),
                                [&](std::size_t startIdx, std::size_t endIdx) noexcept {
                                  const std::size_t cnt = endIdx - startIdx;
                                  sqDistancesAosBlock<T>(firstRow, xData + (startIdx * d), cnt, d,
                                                         qData + startIdx);
                                });
  }

  /// Pool fan-out for the sumD2 reduction.
  T sumReduceParallel(const T *p, std::size_t n, math::Pool pool) noexcept {
    const std::size_t workers = pool.workerCount();
    std::array<T, 64> partials{};
    CLUSTERING_ALWAYS_ASSERT(workers <= 64);
    pool.parallelForExactBlocksWithSlot<citor::ScatterFoldHints>(
        std::size_t{0}, n, workers,
        [&, p](std::size_t startIdx, std::size_t endIdx, std::size_t slot) noexcept {
          partials[slot] = sumReduceAvx2(p + startIdx, endIdx - startIdx);
        });
    T s = T{0};
    for (std::size_t w = 0; w < workers; ++w) {
      s += partials[w];
    }
    return s;
  }

  NDArray<T, 1> m_q;
  NDArray<T, 1> m_aliasProb;
  NDArray<std::size_t, 1> m_aliasIdx;
  NDArray<std::size_t, 1> m_aliasSmall;
  NDArray<std::size_t, 1> m_aliasLarge;
  NDArray<std::size_t, 1> m_yIdxBatch;
  NDArray<T, 1> m_uBatch;
  NDArray<T, 1> m_yQBatch;
  NDArray<T, 1> m_yDistBatch;
};

} // namespace clustering::kmeans
