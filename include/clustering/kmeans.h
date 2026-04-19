#pragma once

#include <BS_thread_pool.hpp>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <thread>
#include <type_traits>

#include "clustering/always_assert.h"
#include "clustering/kmeans/policy/auto_seeder.h"
#include "clustering/kmeans/policy/lloyd.h"
#include "clustering/kmeans/policy/lloyd_fused_gemm.h"
#include "clustering/kmeans/policy/seeder.h"
#include "clustering/math/thread.h"
#include "clustering/ndarray.h"

namespace clustering {

/**
 * @brief Lloyd-family k-means.
 *
 * The algorithm and seeder are template parameters with concept constraints. The default
 * instantiation carries @c LloydFusedGemm<T> and @c AutoSeeder<T>, the latter picking between
 * greedy k-means++ and AFK-MC2 against workload shape at @c run time. Callers who want to pin
 * a specific combination spell it out, e.g.
 * @c KMeans<float, LloydFusedGemm<float>, AfkMc2Seeder<float>>.
 *
 * @note @c KMeans does NOT own @p X. The caller must keep the @c NDArray alive for the lifetime
 *       of every @ref run call on this instance. An @c n_init > 1 harness constructs @c KMeans
 *       once and calls @c run repeatedly against the same @p X so policy scratch amortizes
 *       across runs at a fixed @c (n, d, k, nJobs) tuple.
 *
 * @tparam T      Element type. Only @c float is supported; add a @c double specialization to
 *                extend.
 * @tparam Algo   Lloyd driver satisfying @ref kmeans::LloydStrategy<Algo, T>.
 * @tparam Seeder Seeder satisfying @ref kmeans::SeederStrategy<Seeder, T>.
 */
template <class T, class Algo = kmeans::LloydFusedGemm<T>, class Seeder = kmeans::AutoSeeder<T>>
  requires kmeans::LloydStrategy<Algo, T> && kmeans::SeederStrategy<Seeder, T>
class KMeans {
  static_assert(std::is_same_v<T, float>,
                "KMeans<T> supports only float; add a double specialization to extend.");

public:
  /**
   * @brief Construct a reusable k-means fitter.
   *
   * @param k     Number of clusters (>= 1).
   * @param nJobs Worker count for the internal thread pool. A value of @c 0 is clamped upward
   *              to @c std::thread::hardware_concurrency() so the pool is always usable by the
   *              @ref math::Pool helpers.
   */
  explicit KMeans(std::size_t k, std::size_t nJobs = std::thread::hardware_concurrency())
      : m_k(k), m_centroids({0, 0}), m_labels({0}) {
    CLUSTERING_ALWAYS_ASSERT(k >= 1);
    // Skip pool construction when the caller asks for serial execution. The pool ctor spawns
    // nJobs std::thread workers plus a detach/join pair per instance; at the small-n corner
    // those threads sit idle yet still cost ~20 us per KMeans instance. Leaving m_pool empty
    // lets run() pass Pool{nullptr} down without the creation/destruction detour.
    const std::size_t clamped = clampedJobCount(nJobs);
    if (clamped > 1) {
      m_pool.emplace(clamped);
    }
  }

  KMeans(const KMeans &) = delete;
  KMeans &operator=(const KMeans &) = delete;
  KMeans(KMeans &&) = delete;
  KMeans &operator=(KMeans &&) = delete;
  ~KMeans() = default;

  /**
   * @brief Fit to @p X.
   *
   * @param X       Contiguous n x d dataset. The caller retains ownership; @p X must outlive
   *                this @c run call and every subsequent call that intends to reuse scratch.
   * @param maxIter Iteration cap on the inner Lloyd loop.
   * @param tol     Convergence tolerance relative to the mean column variance of @p X
   *                (sklearn convention). The effective sum-of-shift-squared threshold is
   *                @c tol * mean(var(X, axis=0)); iteration stops when the Kahan-summed per-
   *                centroid shift-squared falls at or below that threshold.
   * @param seed    PRNG seed. Identical @c (seed, nJobs, X, maxIter, tol) produces bit-identical
   *                labels, centroids, and inertia at @c nJobs=1.
   *
   * @warning @p X must remain alive and unchanged for the full duration of this call.
   */
  void run(const NDArray<T, 2> &X, std::size_t maxIter = 300, T tol = T{1e-4},
           std::uint64_t seed = 0) {
    const std::size_t n = X.dim(0);
    const std::size_t d = X.dim(1);

    CLUSTERING_ALWAYS_ASSERT(m_k >= 1);
    CLUSTERING_ALWAYS_ASSERT(n >= m_k);

    ensureOutputShape(n, d);

    if (n == 0 || d == 0) {
      m_nIter = 0;
      m_converged = true;
      m_inertia = 0.0;
      return;
    }

    const math::Pool pool{m_pool.has_value() ? &*m_pool : nullptr};
    m_seeder.run(X, m_k, seed, pool, m_centroids);
    m_lloyd.run(X, m_centroids, m_k, maxIter, tol, pool, m_labels, m_inertia, m_nIter, m_converged);
  }

  /// Length-n assignment; each entry is in @c [0, k).
  [[nodiscard]] const NDArray<std::int32_t, 1> &labels() const noexcept { return m_labels; }
  /// k x d fitted centroids.
  [[nodiscard]] const NDArray<T, 2, Layout::Contig> &centroids() const noexcept {
    return m_centroids;
  }
  /// Final inertia: Kahan-summed @c f64 total of per-point squared distance to assignment.
  [[nodiscard]] double inertia() const noexcept { return m_inertia; }
  /// Iterations executed before @c tol or @c maxIter fired.
  [[nodiscard]] std::size_t nIter() const noexcept { return m_nIter; }
  /// True iff the last run stopped because centroid shift fell at or below @c tol.
  [[nodiscard]] bool converged() const noexcept { return m_converged; }

  /// Release every scratch buffer. The next @ref run call reallocates against its shape.
  void reset() {
    m_centroids = NDArray<T, 2, Layout::Contig>({0, 0});
    m_labels = NDArray<std::int32_t, 1>({0});
    m_inertia = 0.0;
    m_nIter = 0;
    m_converged = false;
    m_lloyd = Algo{};
    m_seeder = Seeder{};
  }

private:
  static std::size_t clampedJobCount(std::size_t nJobs) noexcept {
    if (nJobs == 0) {
      const std::size_t hw = std::thread::hardware_concurrency();
      return hw == 0 ? std::size_t{1} : hw;
    }
    return nJobs;
  }

  void ensureOutputShape(std::size_t n, std::size_t d) {
    if (m_centroids.dim(0) != m_k || m_centroids.dim(1) != d) {
      m_centroids = NDArray<T, 2, Layout::Contig>({m_k, d});
    }
    if (m_labels.dim(0) != n) {
      m_labels = NDArray<std::int32_t, 1>({n});
    }
  }

  std::size_t m_k;
  std::optional<BS::light_thread_pool> m_pool;
  NDArray<T, 2, Layout::Contig> m_centroids;
  NDArray<std::int32_t, 1> m_labels;
  double m_inertia = 0.0;
  std::size_t m_nIter = 0;
  bool m_converged = false;

  Algo m_lloyd{};
  Seeder m_seeder{};
};

} // namespace clustering
