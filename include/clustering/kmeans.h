#pragma once

#include <BS_thread_pool.hpp>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <thread>
#include <type_traits>

#include "clustering/always_assert.h"
#include "clustering/kmeans/detail/dispatch.h"
#include "clustering/kmeans/detail/solver.h"
#include "clustering/math/thread.h"
#include "clustering/ndarray.h"

namespace clustering {

/**
 * @brief Lloyd-family k-means with fused-argmin-GEMM Lloyd seeded by greedy k-means++.
 *
 * Default @c run dispatches between Lloyd and Yinyang based on @c k; both produce final
 * labels in @c [0, k) and matching inertia within float tolerance. Seeder dispatch picks
 * greedy k-means++ by default and escalates to AFK-MC2 at
 * @c n >= @c kmeans::detail::afkmc2NThreshold AND @c k >= @c kmeans::detail::afkmc2KFloor;
 * forcing AFK-MC2 below either threshold falls through to greedy k-means++ and @ref lastSeeder
 * reports the seeder that actually ran.
 *
 * @note @c KMeans does NOT own @p X. The caller must keep the @c NDArray alive for the
 *       lifetime of every @ref run call on this instance. An @c n_init > 1 harness
 *       constructs @c KMeans once and calls @c run repeatedly against the same @p X so the
 *       solver's scratch amortizes across runs at a fixed @c (n, d, k, nJobs) tuple.
 *       Mirrors @c DBSCAN and @c KDTree consumption semantics.
 *
 * @tparam T Element type. Only @c float is supported; add a @c double specialization to
 *           extend.
 */
template <class T> class KMeans {
  static_assert(std::is_same_v<T, float>,
                "KMeans<T> supports only float; add a double specialization to extend.");

public:
  /**
   * @brief Construct a reusable k-means fitter.
   *
   * @param k     Number of clusters (>= 1).
   * @param nJobs Worker count for the internal thread pool. A value of @c 0 is clamped
   *              upward to @c std::thread::hardware_concurrency() so the pool is always
   *              usable by the @ref math::Pool helpers.
   */
  explicit KMeans(std::size_t k, std::size_t nJobs = std::thread::hardware_concurrency()) : m_k(k) {
    CLUSTERING_ALWAYS_ASSERT(k >= 1);
    // Skip pool construction when the caller asks for serial execution. The pool ctor spawns
    // @c nJobs std::thread workers plus a detach/join pair per instance; at the small-n corner
    // those threads sit idle yet still cost ~20 us per KMeans instance. Leaving @c m_pool empty
    // lets @ref run pass @c Pool{nullptr} down without the creation/destruction detour.
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
   * @brief Fit to @p X. No allocation fires inside the iteration loop; solver scratch is
   *        sized at the first @c run call and lazily resized when the @c (n, d, k, nJobs)
   *        shape tuple changes between calls.
   *
   * @param X       Contiguous n x d dataset. The caller retains ownership; @p X must outlive
   *                this @c run call and every subsequent call that intends to reuse scratch.
   * @param maxIter Iteration cap on the inner Lloyd loop.
   * @param tol     Convergence tolerance on the L2 centroid shift (linear). Internally
   *                compared as @c tol*tol against the Kahan-summed per-centroid shift-squared.
   * @param seed    PRNG seed. Identical @c (seed, nJobs, X, maxIter, tol) produces
   *                bit-identical labels, centroids, and inertia.
   *
   * @warning @p X must remain alive and unchanged for the full duration of this call.
   */
  void run(const NDArray<T, 2> &X, std::size_t maxIter = 300, T tol = T{1e-4},
           std::uint64_t seed = 0) {
    // Absent @c m_pool the run is strictly serial; the kernels short-circuit every
    // @c shouldParallelize check via the null @c Pool.pool pointer and skip the per-site
    // @c BS::submit_blocks dispatch round-trip (futures + move_only_function + queue).
    const math::Pool pool{m_pool.has_value() ? &*m_pool : nullptr};
    m_solver.fit(X, m_k, maxIter, tol, seed, pool, m_forcedAlgorithm, m_forcedSeeder);
  }

  /// Length-n assignment; each entry is in @c [0, k).
  [[nodiscard]] const NDArray<std::int32_t, 1> &labels() const noexcept {
    return m_solver.labels();
  }
  /// k x d fitted centroids.
  [[nodiscard]] const NDArray<T, 2> &centroids() const noexcept { return m_solver.centroids(); }
  /// Final inertia: Kahan-summed @c f64 total of per-point squared distance to assignment.
  [[nodiscard]] double inertia() const noexcept { return m_solver.inertia(); }
  /// Iterations executed before @c tol or @c maxIter fired.
  [[nodiscard]] std::size_t nIter() const noexcept { return m_solver.nIter(); }
  /// True iff the last run stopped because centroid shift fell at or below @c tol.
  [[nodiscard]] bool converged() const noexcept { return m_solver.converged(); }

  /**
   * @brief Diagnostic: which inner algorithm fired on the last run. Non-stable API; the
   *        @c detail::Algorithm enumerators are reserved across releases and must not be
   *        branched on in production code.
   */
  [[nodiscard]] kmeans::detail::Algorithm lastAlgorithm() const noexcept {
    return m_solver.lastAlgorithm();
  }
  /**
   * @brief Diagnostic: which seeder fired on the last run. Same non-contractual status as
   *        @ref lastAlgorithm.
   */
  [[nodiscard]] kmeans::detail::Seeder lastSeeder() const noexcept { return m_solver.lastSeeder(); }

  /// Release every scratch buffer. The next @ref run call reallocates against its shape.
  void reset() { m_solver.reset(); }

  /**
   * @brief Pin the dispatched algorithm across subsequent @ref run calls.
   *
   * @warning The enumerator values in @c kmeans::detail::Algorithm are not a stable API;
   *          new variants land between releases. Use @ref clearForcedAlgorithm to return
   *          the instance to auto-dispatch.
   */
  void forceAlgorithm(kmeans::detail::Algorithm alg) noexcept { m_forcedAlgorithm = alg; }
  void clearForcedAlgorithm() noexcept { m_forcedAlgorithm.reset(); }

  /**
   * @brief Pin the dispatched seeder across subsequent @ref run calls. Same non-stable
   *        status as @ref forceAlgorithm.
   */
  void forceSeeder(kmeans::detail::Seeder s) noexcept { m_forcedSeeder = s; }
  void clearForcedSeeder() noexcept { m_forcedSeeder.reset(); }

private:
  static std::size_t clampedJobCount(std::size_t nJobs) noexcept {
    if (nJobs == 0) {
      const std::size_t hw = std::thread::hardware_concurrency();
      return hw == 0 ? std::size_t{1} : hw;
    }
    return nJobs;
  }

  std::size_t m_k;
  std::optional<BS::light_thread_pool> m_pool;
  kmeans::detail::Solver<T> m_solver{};
  std::optional<kmeans::detail::Algorithm> m_forcedAlgorithm;
  std::optional<kmeans::detail::Seeder> m_forcedSeeder;
};

} // namespace clustering
