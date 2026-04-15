#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

#include "clustering/always_assert.h"
#include "clustering/kmeans/detail/seed_greedy_kmpp.h"
#include "clustering/math/rng.h"
#include "clustering/math/thread.h"
#include "clustering/ndarray.h"

namespace clustering::kmeans::detail {

/**
 * @brief Scratch storage for @ref seedAfkMc2.
 *
 * The solver owns one instance across @c run() calls so the per-point proposal weights are not
 * reallocated at the same shape tuple. Sized lazily by @ref ensureAfkMc2ScratchShape; callers
 * outside the solver should size @c q to @c n before calling @ref seedAfkMc2.
 */
template <class T> struct AfkMc2Scratch {
  /// Proposal distribution q (length n), derived from distance to the first centroid.
  NDArray<T, 1> q;
  /// Prefix sum of q (length n), drives O(log n) inverse-CDF sampling.
  NDArray<T, 1> qCum;

  /// Default-construct the scratch with zero-length rank-1 arrays; @ref ensureAfkMc2ScratchShape
  /// resizes them at the first @c run() call.
  AfkMc2Scratch() : q({0}), qCum({0}) {}
};

/**
 * @brief Lazily resize @p scratch to hold an @p n-length proposal vector.
 *
 * No-op when the existing @c q already matches @p n; otherwise reallocates. Intended for callers
 * that own @ref AfkMc2Scratch across repeated fits.
 */
template <class T> inline void ensureAfkMc2ScratchShape(AfkMc2Scratch<T> &scratch, std::size_t n) {
  if (scratch.q.dim(0) != n) {
    scratch.q = NDArray<T, 1>({n});
  }
  if (scratch.qCum.dim(0) != n) {
    scratch.qCum = NDArray<T, 1>({n});
  }
}

/**
 * @brief AFK-MC2 seeding (Bachem, Lucic, Hassani, Krause, NeurIPS 2016).
 *
 * Sublinear-in-n MCMC approximation to k-means++: draws the first centroid uniformly, builds a
 * length-@p n proposal distribution @c q(i) = 0.5 * D(x_i, c_1)^2 / sum_D2 + 0.5 * 1/n, and then
 * for each remaining centroid runs a Markov chain of length @p m that accepts a proposal with
 * probability @c min(1, proposed_weight / current_weight) where the weight is the squared
 * distance to the current centroid set divided by the proposal density.
 *
 * Chain execution is strictly serial and thread-unaware so the PRNG draw order is fixed by
 * @c (seed, n, k, m) regardless of @p pool worker count. The preprocessing sweep (pair-distance
 * to the first centroid) fans out across @p pool; the per-thread work is pure read with no
 * shared mutation, and the running sum is reduced serially.
 *
 * Degenerate guard: when all points coincide with the first centroid (@c sum_D2 == 0) the
 * proposal collapses to uniform @c q(i) = 1/n so the chain still makes progress.
 *
 * @tparam T Element type (@c float for V1).
 * @param X            Data matrix (n x d), contiguous.
 * @param k            Number of centroids to seed.
 * @param m            Markov chain length per centroid. Default 200 at the solver level.
 * @param seed         RNG seed; identical seed + @c (X, k, m) produces identical centroids.
 * @param scratch      Preallocated scratch; @c q must already hold @p n entries.
 * @param outCentroids Output centroid matrix (k x d), contiguous; populated in row order.
 * @param pool         Parallelism injection (preprocessing sweep only).
 */
template <class T>
void seedAfkMc2(const NDArray<T, 2, Layout::Contig> &X, std::size_t k, std::size_t m,
                std::uint64_t seed, AfkMc2Scratch<T> &scratch,
                NDArray<T, 2, Layout::Contig> &outCentroids, math::Pool pool) {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                "seedAfkMc2<T> requires T to be float or double");

  const std::size_t n = X.dim(0);
  const std::size_t d = X.dim(1);

  CLUSTERING_ALWAYS_ASSERT(outCentroids.isMutable());
  CLUSTERING_ALWAYS_ASSERT(outCentroids.dim(0) == k);
  CLUSTERING_ALWAYS_ASSERT(outCentroids.dim(1) == d);
  CLUSTERING_ALWAYS_ASSERT(scratch.q.isMutable());
  CLUSTERING_ALWAYS_ASSERT(scratch.q.dim(0) == n);
  CLUSTERING_ALWAYS_ASSERT(scratch.qCum.isMutable());
  CLUSTERING_ALWAYS_ASSERT(scratch.qCum.dim(0) == n);
  CLUSTERING_ALWAYS_ASSERT(k >= 1);
  CLUSTERING_ALWAYS_ASSERT(n >= k);
  CLUSTERING_ALWAYS_ASSERT(m >= 1);

  (void)pool;

  math::pcg64 rng;
  rng.seed(seed);

  const T *xData = X.data();
  T *centroidsData = outCentroids.data();
  T *qData = scratch.q.data();

  // Step 1: first centroid uniformly.
  const auto first = static_cast<std::size_t>(math::randUniformU64(rng) % n);
  std::memcpy(centroidsData, xData + (first * d), d * sizeof(T));

  if (k == 1) {
    return;
  }

  // Step 2: proposal distribution q(i) = 0.5 * d(x_i, c_1)^2 / sumD2 + 0.5 * 1/n.
  // Squared distance to the first centroid drives the data-proximal half; the 1/n floor keeps
  // every point reachable by the chain even when sumD2 is dominated by a few outliers.
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
    // Degenerate: every point coincides with c_1. Fall back to uniform so the chain is still
    // ergodic over the point set.
    for (std::size_t i = 0; i < n; ++i) {
      qData[i] = invN;
    }
  }

  // Cumulative prefix sum over q for O(log n) index draws via inverse-CDF. sumQ is the last
  // prefix entry; with the 0.5/n floor above, sumQ is strictly positive for any n >= 1.
  // The strictly-positive guarantee lets us skip an explicit guard in the chain below.
  T *qCumData = scratch.qCum.data();
  T running = T{0};
  for (std::size_t i = 0; i < n; ++i) {
    running += qData[i];
    qCumData[i] = running;
  }
  const T qTotal = qCumData[n - 1];

  // Inverse-CDF sample from q via binary search; matches numpy.searchsorted(side='right').
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

  // Step 3: for each remaining centroid, run a length-m Markov chain. Distances to the current
  // centroid set are recomputed against all chosen rows -- O(c) per candidate where c is the
  // count of already-placed centroids. Full set traversal stays O(k * m * k * d) = O(k^2 * m * d),
  // which matches Bachem's paper; the inner k factor is small compared with n so this keeps the
  // sublinear-in-n property.
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
    // Initialize chain state.
    std::size_t xIdx = sampleFromQ();
    T xDist = distToChosen(xIdx, c);
    T xQ = qData[xIdx];

    for (std::size_t step = 0; step < m; ++step) {
      const std::size_t yIdx = sampleFromQ();
      const T yDist = distToChosen(yIdx, c);
      const T yQ = qData[yIdx];

      // Acceptance ratio is (yDist / yQ) / (xDist / xQ); reorder as (yDist * xQ) vs (xDist * yQ)
      // to skip the division. Draw @c u every step so the RNG sequence depends only on (seed,
      // n, k, m) and never on the branch outcomes inside the chain -- essential for bit-
      // identical repeatability across runs. When @c denom is zero the current weight vanishes
      // and any proposal is indistinguishable or strictly better; accept unconditionally.
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

} // namespace clustering::kmeans::detail
