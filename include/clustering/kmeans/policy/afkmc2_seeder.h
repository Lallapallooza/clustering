#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

#include "clustering/always_assert.h"
#include "clustering/math/detail/avx2_helpers.h"
#include "clustering/math/rng.h"
#include "clustering/math/thread.h"
#include "clustering/ndarray.h"

namespace clustering::kmeans {

using math::detail::sqEuclideanRowPtr;

/**
 * @brief AFK-MC2 seeder (Bachem, Lucic, Hassani, Krause, NeurIPS 2016).
 *
 * Sublinear-in-n MCMC approximation to k-means++: draws the first centroid uniformly, builds a
 * length-@c n proposal distribution `q(i)` = 0.5 * D(x_i, c_1)^2 / sum_D2 + 0.5 * 1/n, and then
 * for each remaining centroid runs a Markov chain of length @c m that accepts a proposal with
 * probability `min(1, proposed_weight / current_weight)` where the weight is the squared
 * distance to the current centroid set divided by the proposal density.
 *
 * Chain execution is strictly serial and thread-unaware so the PRNG draw order is fixed by
 * `(seed, n, k, m)` regardless of @p pool worker count. The preprocessing sweep fans out
 * across @p pool; the per-thread work is pure read with no shared mutation.
 *
 * Degenerate guard: when all points coincide with the first centroid (`sum_D2 == 0`) the
 * proposal collapses to uniform `q(i)` = 1/n so the chain remains ergodic.
 *
 * The chain's log-k approximation bound degrades at small @c k: below @c k = @ref
 * AfkMc2Seeder::kFloor the bound is too loose to beat greedy k-means++, and callers at that regime
 * should pin
 * @ref GreedyKmppSeeder (directly or via @ref AutoSeeder, which picks it by shape).
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

  AfkMc2Seeder() : m_q({0}), m_qCum({0}) {}

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

    (void)pool;

    ensureShape(n);

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

    // Step 2: proposal distribution q(i) = 0.5 * d(x_i, c_1)^2 / sumD2 + 0.5 * 1/n.
    // Squared distance to the first centroid drives the data-proximal half; the 1/n floor
    // keeps every point reachable by the chain even when sumD2 is dominated by outliers.
    const T *firstRow = centroidsData;
    T sumD2 = T{0};
    for (std::size_t i = 0; i < n; ++i) {
      const T d2 = sqEuclideanRowPtr(xData + (i * d), firstRow, d);
      qData[i] = d2;
      sumD2 += d2;
    }

    const T invN = T{1} / static_cast<T>(n);
    if (sumD2 > T{0}) {
      const T invSum = T{1} / sumD2;
      for (std::size_t i = 0; i < n; ++i) {
        qData[i] = (T{0.5} * qData[i] * invSum) + (T{0.5} * invN);
      }
    } else {
      // Degenerate: every point coincides with c_1. Fall back to uniform so the chain stays
      // ergodic over the point set.
      for (std::size_t i = 0; i < n; ++i) {
        qData[i] = invN;
      }
    }

    // Cumulative prefix sum over q for O(log n) index draws via inverse-CDF. With the 0.5/n
    // floor above, sumQ is strictly positive for any n >= 1.
    T *qCumData = m_qCum.data();
    T running = T{0};
    for (std::size_t i = 0; i < n; ++i) {
      running += qData[i];
      qCumData[i] = running;
    }
    const T qTotal = qCumData[n - 1];

    auto sampleFromQ = [&]() noexcept -> std::size_t {
      const T u = math::randUnit<T>(rng) * qTotal;
      std::size_t lo = 0;
      std::size_t hi = n;
      while (lo < hi) {
        const std::size_t mid = lo + ((hi - lo) / 2);
        if (qCumData[mid] > u) {
          hi = mid;
        } else {
          lo = mid + 1;
        }
      }
      return lo < n ? lo : n - 1;
    };

    // Step 3: for each remaining centroid, run a length-m Markov chain. Distances to the
    // current centroid set are recomputed against all chosen rows -- O(c) per candidate where
    // c is the count of already-placed centroids.
    auto distToChosen = [&](std::size_t pointIdx, std::size_t chosenCount) noexcept -> T {
      const T *row = xData + (pointIdx * d);
      T best = sqEuclideanRowPtr(row, centroidsData, d);
      for (std::size_t c = 1; c < chosenCount; ++c) {
        const T cand = sqEuclideanRowPtr(row, centroidsData + (c * d), d);
        if (cand < best) {
          best = cand;
        }
      }
      return best;
    };

    for (std::size_t c = 1; c < k; ++c) {
      std::size_t xIdx = sampleFromQ();
      T xDist = distToChosen(xIdx, c);
      T xQ = qData[xIdx];

      for (std::size_t step = 0; step < m; ++step) {
        const std::size_t yIdx = sampleFromQ();
        const T yDist = distToChosen(yIdx, c);
        const T yQ = qData[yIdx];

        // Acceptance ratio is (yDist / yQ) / (xDist / xQ); reorder as (yDist * xQ) vs
        // (xDist * yQ) to skip the division. Draw u every step so the RNG sequence depends
        // only on (seed, n, k, m) and never on the branch outcomes inside the chain --
        // essential for bit-identical repeatability across runs. When denom is zero the
        // current weight vanishes and any proposal is indistinguishable or strictly better;
        // accept unconditionally.
        const T numer = yDist * xQ;
        const T denom = xDist * yQ;
        const T u = math::randUnit<T>(rng);
        const bool accept = (denom <= T{0}) || ((u * denom) < numer);

        if (accept) {
          xIdx = yIdx;
          xDist = yDist;
          xQ = yQ;
        }
      }

      std::memcpy(centroidsData + (c * d), xData + (xIdx * d), d * sizeof(T));
    }
  }

  void ensureShape(std::size_t n) {
    if (m_q.dim(0) != n) {
      m_q = NDArray<T, 1>({n});
    }
    if (m_qCum.dim(0) != n) {
      m_qCum = NDArray<T, 1>({n});
    }
  }

  NDArray<T, 1> m_q;
  NDArray<T, 1> m_qCum;
};

} // namespace clustering::kmeans
