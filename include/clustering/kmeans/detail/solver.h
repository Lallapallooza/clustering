#pragma once

#include <BS_thread_pool.hpp>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <type_traits>
#include <utility>

#include "clustering/always_assert.h"
#include "clustering/kmeans/detail/dispatch.h"
#include "clustering/kmeans/detail/lloyd_fused.h"
#include "clustering/kmeans/detail/seed_afkmc2.h"
#include "clustering/kmeans/detail/seed_greedy_kmpp.h"
#include "clustering/math/detail/pairwise_argmin_outer.h"
#include "clustering/math/reduce.h"
#include "clustering/math/thread.h"
#include "clustering/ndarray.h"

namespace clustering::kmeans::detail {

/**
 * @brief Owner of every scratch buffer consumed by @c KMeans<T>::run.
 *
 * First @c fit call sizes the buffers against the @c (n, d, k, workerCount) shape tuple;
 * subsequent calls at the identical tuple reuse the existing storage. Any component changing
 * triggers a lazy resize at the top of @c fit, before the Lloyd loop. Within a single
 * @c fit call the shape is constant so no allocation fires inside the assignment / label-fold /
 * convergence-check window.
 *
 * @warning The solver borrows @p X through the driving @c fit call. Callers retain ownership
 *          of @p X and must keep it alive across every @c fit invocation that shares scratch.
 */
template <class T> class Solver {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                "Solver<T> requires T to be float or double");

public:
  Solver()
      : m_centroids({0, 0}), m_centroidsOld({0, 0}), m_cSqNorms({0}), m_sums({0, 0}), m_counts({0}),
        m_labels({0}), m_minDistSq({0}), m_tmpDistSq({0}), m_shiftSq({0}), m_partialSums({0}),
        m_partialComps({0}), m_partialCounts({0}), m_foldComp({0}), m_packedB({0}),
        m_packedCSqNorms({0}) {}

  /**
   * @brief Fit one run of k-means with the supplied algorithm/seeder overrides.
   *
   * @param X            Borrowed data matrix (n x d).
   * @param k            Number of clusters.
   * @param maxIter      Iteration cap on the Lloyd loop.
   * @param tol          Convergence tolerance on the L2 centroid shift (linear, not squared).
   * @param seed         PRNG seed.
   * @param pool         Parallelism injection.
   * @param algoOverride Optional forced algorithm; when set the chooser is bypassed.
   * @param seederOverride Optional forced seeder; when set the chooser is bypassed.
   */
  void fit(const NDArray<T, 2, Layout::Contig> &X, std::size_t k, std::size_t maxIter, T tol,
           std::uint64_t seed, math::Pool pool, std::optional<Algorithm> algoOverride,
           std::optional<Seeder> seederOverride) {
    const std::size_t n = X.dim(0);
    const std::size_t d = X.dim(1);

    CLUSTERING_ALWAYS_ASSERT(k >= 1);
    CLUSTERING_ALWAYS_ASSERT(n >= k);

    if (n == 0 || d == 0) {
      m_nIter = 0;
      m_converged = true;
      m_inertia = 0.0;
      return;
    }

    const std::size_t workerCount = pool.workerCount();
    ensureShape(n, d, k, workerCount);

    // Auto-dispatch gates: compute the selection up front so the diagnostic accessors show
    // what ran even when the override forced a different path.
    const Algorithm autoAlgo = chooseAlgorithm(n, d, k);
    const Seeder autoSeeder = chooseSeeder(n, k);
    const Algorithm algo = algoOverride.value_or(autoAlgo);
    const Seeder seeder = seederOverride.value_or(autoSeeder);

    Seeder seeder;
    if (seederOverride.has_value()) {
      seeder = *seederOverride;
      const bool implemented = seeder == Seeder::kGreedyKMeansPlusPlus || seeder == Seeder::kAfkMc2;
      CLUSTERING_ALWAYS_ASSERT(implemented);
    } else {
      seeder = chooseSeeder(n, k);
    }

    // AFK-MC2's MCMC approximation guarantee degrades at small k; below @c afkmc2KFloor the
    // greedy k-means++ variant is cheaper and higher quality. The fallback is observable via
    // @c lastSeeder(), which reports the seeder that actually ran.
    if (seeder == Seeder::kAfkMc2 && k < afkmc2KFloor) {
      seeder = Seeder::kGreedyKMeansPlusPlus;
    }

    m_lastAlgorithm = algo;
    m_lastSeeder = seeder;

    // Seed. Both seeders populate @c m_centroids; the downstream Lloyd / Yinyang drivers run
    // their own assignment sweep before reading @c m_minDistSq, so AFK-MC2 deliberately skips
    // the per-point distance prime that greedy-kmpp produces as a side-effect.
    if (seeder == Seeder::kAfkMc2) {
      ensureAfkMc2ScratchShape<T>(m_afkmc2Scratch, n);
      seedAfkMc2<T>(X, k, afkmc2ChainLengthDefault, seed, m_afkmc2Scratch, m_centroids, pool);
    } else {
      seedGreedyKMeansPlusPlus<T>(X, m_centroids, m_minDistSq, m_tmpDistSq, seed, pool);
    }

    // Drive Lloyd.
    const LloydScratch<T> scratch{&m_centroids,   &m_centroidsOld,  &m_cSqNorms,      &m_sums,
                                  &m_counts,      &m_labels,        &m_minDistSq,     &m_shiftSq,
                                  &m_partialSums, &m_partialComps,  &m_partialCounts, &m_foldComp,
                                  &m_packedB,     &m_packedCSqNorms};
    const auto [iter, converged] = runLloydFused<T>(X, scratch, k, maxIter, tol, pool);
    m_nIter = iter;
    m_converged = converged;

    // Inertia: Kahan-summed in f64 to pin the 1% gate at large (n, k) envelopes where the
    // naive single-pass f32 add would drift. minDistSq still holds the assignment payload
    // because runLloydFused re-ran the final assignment before returning.
    double sum = 0.0;
    double c = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      const auto addend = static_cast<double>(m_minDistSq(i));
      const double y = addend - c;
      const double t = sum + y;
      c = (t - sum) - y;
      sum = t;
    }
    m_inertia = sum;
  }

  /// Drop every buffer so the next @c fit reallocates from a cold state.
  void reset() {
    m_centroids = NDArray<T, 2, Layout::Contig>({0, 0});
    m_centroidsOld = NDArray<T, 2, Layout::Contig>({0, 0});
    m_cSqNorms = NDArray<T, 1>({0});
    m_sums = NDArray<T, 2, Layout::Contig>({0, 0});
    m_counts = NDArray<std::int32_t, 1>({0});
    m_labels = NDArray<std::int32_t, 1>({0});
    m_minDistSq = NDArray<T, 1>({0});
    m_tmpDistSq = NDArray<T, 1>({0});
    m_shiftSq = NDArray<T, 1>({0});
    m_partialSums = NDArray<T, 1>({0});
    m_partialComps = NDArray<T, 1>({0});
    m_partialCounts = NDArray<std::int32_t, 1>({0});
    m_foldComp = NDArray<T, 1>({0});
    m_packedB = NDArray<T, 1>({0});
    m_packedCSqNorms = NDArray<T, 1>({0});
    m_yinyangBounds = YinyangBounds<T>{};
    m_yinyangPlan = YinyangPlan<T>{};
    m_lastYinyangStats = YinyangRunStats{};
    m_afkmc2Scratch = AfkMc2Scratch<T>{};
    m_n = 0;
    m_d = 0;
    m_k = 0;
    m_workerCount = 0;
    m_nIter = 0;
    m_converged = false;
    m_inertia = 0.0;
    m_lastAlgorithm = Algorithm::kLloydFusedGemm;
    m_lastSeeder = Seeder::kGreedyKMeansPlusPlus;
  }

  [[nodiscard]] const NDArray<std::int32_t, 1> &labels() const noexcept { return m_labels; }
  [[nodiscard]] const NDArray<T, 2, Layout::Contig> &centroids() const noexcept {
    return m_centroids;
  }
  [[nodiscard]] double inertia() const noexcept { return m_inertia; }
  [[nodiscard]] std::size_t nIter() const noexcept { return m_nIter; }
  [[nodiscard]] bool converged() const noexcept { return m_converged; }
  [[nodiscard]] Algorithm lastAlgorithm() const noexcept { return m_lastAlgorithm; }
  [[nodiscard]] Seeder lastSeeder() const noexcept { return m_lastSeeder; }

private:
  void ensureShape(std::size_t n, std::size_t d, std::size_t k, std::size_t workerCount) {
    const bool shapeChanged = (n != m_n) || (d != m_d) || (k != m_k);
    const bool workerChanged = (workerCount != m_workerCount);
    if (!shapeChanged && !workerChanged) {
      return;
    }

    if (shapeChanged) {
      m_centroids = NDArray<T, 2, Layout::Contig>({k, d});
      m_centroidsOld = NDArray<T, 2, Layout::Contig>({k, d});
      m_cSqNorms = NDArray<T, 1>({k});
      m_sums = NDArray<T, 2, Layout::Contig>({k, d});
      m_counts = NDArray<std::int32_t, 1>({k});
      m_labels = NDArray<std::int32_t, 1>({n});
      m_minDistSq = NDArray<T, 1>({n});
      m_tmpDistSq = NDArray<T, 1>({n});
      m_shiftSq = NDArray<T, 1>({k});
      m_foldComp = NDArray<T, 1>({k * d});
      const std::size_t packedBSize = math::detail::packedBScratchSizeFloats(k, d);
      const std::size_t packedNormsSize = math::detail::packedCSqNormsScratchSizeFloats(k);
      m_packedB = NDArray<T, 1>({packedBSize == 0 ? std::size_t{1} : packedBSize});
      m_packedCSqNorms = NDArray<T, 1>({packedNormsSize == 0 ? std::size_t{1} : packedNormsSize});
    }

    // Per-block scratch sizing. Block count caps at workerCount (see BlockPartition in
    // accumulate_by_label.h); we size to the upper bound so both serial and parallel dispatch
    // fit without reallocation inside the loop.
    const std::size_t blocks = workerCount == 0 ? std::size_t{1} : workerCount;
    m_partialSums = NDArray<T, 1>({blocks * k * d});
    m_partialComps = NDArray<T, 1>({blocks * k * d});
    m_partialCounts = NDArray<std::int32_t, 1>({blocks * k});

    m_n = n;
    m_d = d;
    m_k = k;
    m_workerCount = workerCount;
  }

  NDArray<T, 2, Layout::Contig> m_centroids;
  NDArray<T, 2, Layout::Contig> m_centroidsOld;
  NDArray<T, 1> m_cSqNorms;
  NDArray<T, 2, Layout::Contig> m_sums;
  NDArray<std::int32_t, 1> m_counts;
  NDArray<std::int32_t, 1> m_labels;
  NDArray<T, 1> m_minDistSq;
  NDArray<T, 1> m_tmpDistSq;
  NDArray<T, 1> m_shiftSq;
  NDArray<T, 1> m_partialSums;
  NDArray<T, 1> m_partialComps;
  NDArray<std::int32_t, 1> m_partialCounts;
  NDArray<T, 1> m_foldComp;
  NDArray<T, 1> m_packedB;
  NDArray<T, 1> m_packedCSqNorms;

  YinyangBounds<T> m_yinyangBounds;
  YinyangPlan<T> m_yinyangPlan;
  YinyangRunStats m_lastYinyangStats{};
  AfkMc2Scratch<T> m_afkmc2Scratch{};

  std::size_t m_n = 0;
  std::size_t m_d = 0;
  std::size_t m_k = 0;
  std::size_t m_workerCount = 0;

  std::size_t m_nIter = 0;
  bool m_converged = false;
  double m_inertia = 0.0;
  Algorithm m_lastAlgorithm = Algorithm::kLloydFusedGemm;
  Seeder m_lastSeeder = Seeder::kGreedyKMeansPlusPlus;
};

} // namespace clustering::kmeans::detail
