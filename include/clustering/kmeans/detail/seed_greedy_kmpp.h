#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include "clustering/always_assert.h"
#include "clustering/math/rng.h"
#include "clustering/math/thread.h"
#include "clustering/ndarray.h"

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
 * Hot helper for the seeder's candidate-distance update; small enough to inline. Operates on
 * raw pointers so callers can apply it to arbitrary buffers without forcing every caller to
 * wrap them as @c NDArray rows.
 */
template <class T>
[[nodiscard]] inline T sqEuclideanRowPtr(const T *a, const T *b, std::size_t d) noexcept {
  T s = T{0};
  for (std::size_t t = 0; t < d; ++t) {
    const T diff = a[t] - b[t];
    s += diff * diff;
  }
  return s;
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
 * Allocates two scratch arrays sized @c n: @p outMinDistSq (an output, supplied by the caller)
 * receives the per-point squared distance to its chosen centroid; @p tmpDistSq is a per-call
 * scratch buffer. Both are owned by the solver across @c run() invocations.
 *
 * @tparam T Element type; @c float only.
 * @param X            Data matrix (n x d), contiguous.
 * @param outCentroids Output centroid matrix (k x d), contiguous; populated in row order.
 * @param outMinDistSq Output per-point min-squared-distance (length n); populated as the
 *                     seeder progresses, leaves the final state at the end of the routine.
 * @param tmpDistSq    Per-call scratch (length n) for the candidate-distance updates.
 * @param seed         RNG seed; identical seed produces identical centroid selections.
 * @param pool         Parallelism injection. Currently unused in the seeder body but reserved
 *                     for the v2 candidate-scoring fan-out.
 */
template <class T>
void seedGreedyKMeansPlusPlus(const NDArray<T, 2, Layout::Contig> &X,
                              NDArray<T, 2, Layout::Contig> &outCentroids,
                              NDArray<T, 1> &outMinDistSq, NDArray<T, 1> &tmpDistSq,
                              std::uint64_t seed, math::Pool pool) {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                "seedGreedyKMeansPlusPlus<T> requires T to be float or double");

  const std::size_t n = X.dim(0);
  const std::size_t d = X.dim(1);
  const std::size_t k = outCentroids.dim(0);

  CLUSTERING_ALWAYS_ASSERT(outCentroids.isMutable());
  CLUSTERING_ALWAYS_ASSERT(outMinDistSq.isMutable());
  CLUSTERING_ALWAYS_ASSERT(tmpDistSq.isMutable());
  CLUSTERING_ALWAYS_ASSERT(outCentroids.dim(1) == d);
  CLUSTERING_ALWAYS_ASSERT(outMinDistSq.dim(0) == n);
  CLUSTERING_ALWAYS_ASSERT(tmpDistSq.dim(0) == n);
  CLUSTERING_ALWAYS_ASSERT(k >= 1);
  CLUSTERING_ALWAYS_ASSERT(n >= k);

  (void)pool;

  math::pcg64 rng;
  rng.seed(seed);

  const T *xData = X.data();
  T *centroidsData = outCentroids.data();
  T *minSq = outMinDistSq.data();

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

  const std::size_t nLocalTrials = greedyKmppLocalTrials(k);
  std::vector<std::size_t> candidates(nLocalTrials, 0);
  std::vector<T> candidateScores(nLocalTrials, T{0});
  (void)tmpDistSq; // reserved for a future parallel-candidate scoring fan-out

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

    // Score each candidate by computing the resulting sum of min-distances if it were chosen.
    // The current @c minSq array must stay constant during scoring so every candidate competes
    // on the same baseline; an in-loop commit would bias later candidates toward appearing less
    // useful because the baseline would already account for an earlier winner.
    std::size_t bestCandidate = candidates[0];
    T bestScore = std::numeric_limits<T>::infinity();
    for (std::size_t t = 0; t < nLocalTrials; ++t) {
      const std::size_t pick = candidates[t];
      const T *candData = xData + (pick * d);
      T score = T{0};
      for (std::size_t i = 0; i < n; ++i) {
        const T cand = sqEuclideanRowPtr(xData + (i * d), candData, d);
        const T mn = (cand < minSq[i]) ? cand : minSq[i];
        score += mn;
      }
      if (score < bestScore) {
        bestScore = score;
        bestCandidate = pick;
      }
      candidateScores[t] = score;
    }

    // Commit best candidate: copy its row into @p outCentroids and refresh @c minSq to the
    // per-point minimum against the newly added centroid. One pass -- @c tmp is unused in
    // this path.
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
