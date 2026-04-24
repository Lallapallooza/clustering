#pragma once

#include <BS_thread_pool.hpp>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include "clustering/always_assert.h"
#include "clustering/index/auto_range_index.h"
#include "clustering/index/range_query.h"
#include "clustering/math/thread.h"
#include "clustering/ndarray.h"

namespace clustering {

/**
 * @brief Density-based clustering over the eps-neighborhood graph produced by a
 *        @ref clustering::index::RangeIndex backend.
 *
 * DBSCAN groups points whose eps-ball contains at least @c minPts neighbors into density-
 * reachable clusters; points that fall outside any cluster are noise. The backend surfaces the
 * whole adjacency in one call so this class never touches pairwise distances directly.
 *
 * @note @c DBSCAN does NOT own @p X. The caller must keep the @c NDArray alive for the duration
 *       of every @ref run call. Construction is stateless in @p X so a single @c DBSCAN instance
 *       can be reused across fits; repeated runs reallocate the label buffer only when @p n
 *       changes and keep the lazy thread pool spawned after the first parallel-eligible shape.
 *
 * @tparam T Element type of the point cloud.
 * @tparam QueryModel Range-index backend. Defaults to @ref clustering::index::AutoRangeIndex,
 *                    which picks a KD-Tree below @c bruteForceDimFloor and a blocked pairwise
 *                    sweep at or above it.
 */
template <class T, class QueryModel = index::AutoRangeIndex<T>>
  requires index::RangeIndex<QueryModel, T>
class DBSCAN {
public:
  static constexpr std::int32_t UNCLASSIFIED = -2; ///< Sentinel for a point not yet visited.
  static constexpr std::int32_t NOISY = -1; ///< Label assigned to points that no cluster claimed.

  /**
   * @brief Construct a reusable DBSCAN fitter.
   *
   * @param eps    Radius of the density neighbourhood used to test reachability.
   * @param minPts Minimum neighbour count (including self) that marks a core point.
   * @param nJobs  Worker count for the range-index backend. A value of @c 0 is clamped upward
   *               to @c std::thread::hardware_concurrency() so the pool is always usable by the
   *               @ref math::Pool helpers.
   */
  explicit DBSCAN(T eps, std::size_t minPts, std::size_t nJobs = 0)
      : m_eps(eps), m_minPts(minPts), m_nJobs(math::clampedJobCount(nJobs)), m_labels({0}) {
    CLUSTERING_ALWAYS_ASSERT(minPts >= 1);
    // Defer pool construction to @ref run: at small shapes every hot phase gates serial, and
    // spawning nJobs workers in the ctor burns tens of microseconds of thread-create futex
    // traffic that the fit never amortizes. @ref run emplaces on first need and reuses.
  }

  DBSCAN(const DBSCAN &) = delete;
  DBSCAN &operator=(const DBSCAN &) = delete;
  DBSCAN(DBSCAN &&) = delete;
  DBSCAN &operator=(DBSCAN &&) = delete;
  ~DBSCAN() = default;

  /**
   * @brief Fit to @p X.
   *
   * Queries the backend for the full eps-neighbourhood adjacency, derives core-point flags
   * from per-row degrees, then expands clusters sequentially via graph BFS over the adjacency.
   * Remaining unclassified points are marked @ref NOISY.
   *
   * @param X Contiguous n x d dataset. The caller retains ownership; @p X must outlive this
   *          @c run call.
   *
   * @warning @p X must remain alive and unchanged for the full duration of this call.
   */
  void run(const NDArray<T, 2> &X) {
    const std::size_t n = X.dim(0);
    ensureLabelsShape(n);
    m_clusterId = 0;

    if (n == 0) {
      return;
    }

    // One adjacency query per point is the unit of work the range-index backends fan out on;
    // shouldSpawnPool with minOpsPerWorker=16 matches the KDTree adjacency sweep's own
    // `shouldParallelize(n, 4, 2)` gate (@c n / 4 >= 2 * workerCount => `n >= 8` * workerCount)
    // so the pool spawn and the backend fan-out fire at the same shape. @c n * d would
    // under-estimate DBSCAN work -- the backend does per-query tree walks that are much heavier
    // than @c d ops -- and caused @c n_jobs=16 at low @c d to fall back to serial here while
    // every worker-side kernel would have cleared its own gate.
    if (!m_pool.has_value() && math::shouldSpawnPool(n, m_nJobs, /*minOpsPerWorker=*/16)) {
      m_pool.emplace(m_nJobs);
    }
    const math::Pool pool{m_pool.has_value() ? &*m_pool : nullptr};

    QueryModel queryModel(X);
    const auto adj = queryModel.query(m_eps, pool);

    std::vector<std::uint8_t> isCore(n, 0);
    for (std::size_t i = 0; i < n; ++i) {
      isCore[i] = (adj[i].size() >= m_minPts) ? 1U : 0U;
    }

    std::int32_t *labelPtr = m_labels.data();
    std::fill(labelPtr, labelPtr + n, UNCLASSIFIED);

    for (std::size_t i = 0; i < n; ++i) {
      if (labelPtr[i] == UNCLASSIFIED && isCore[i] != 0) {
        expandCluster(i, isCore, adj, labelPtr);
        ++m_clusterId;
      }
    }

    std::replace(labelPtr, labelPtr + n, UNCLASSIFIED, NOISY);
  }

  /// Per-point cluster labels after @ref run; @ref NOISY marks outliers.
  [[nodiscard]] const NDArray<std::int32_t, 1> &labels() const noexcept { return m_labels; }

  /// Total number of clusters discovered by the most recent @ref run.
  [[nodiscard]] std::size_t nClusters() const noexcept { return m_clusterId; }

  /// Release every scratch buffer. The next @ref run call reallocates against its shape.
  void reset() {
    m_labels = NDArray<std::int32_t, 1>({0});
    m_clusterId = 0;
  }

private:
  void ensureLabelsShape(std::size_t n) {
    if (m_labels.dim(0) != n) {
      m_labels = NDArray<std::int32_t, 1>({n});
    }
  }

  /**
   * @brief Expands a cluster from @p seed via BFS over the pre-computed adjacency.
   *
   * Per DBSCAN's density-reachability rule, border points are labelled into the cluster but do
   * not seed further expansion -- only core points contribute their neighbours to the frontier.
   *
   * @param seed    Index of the core point that starts the cluster.
   * @param isCore  Per-point core-point flag derived from the adjacency row sizes.
   * @param adj     Borrowed eps-adjacency list indexed by point row.
   * @param labels  Pointer to the dense label buffer; aliased from `m_labels.data()`.
   */
  void expandCluster(std::size_t seed, const std::vector<std::uint8_t> &isCore,
                     const std::vector<std::vector<std::int32_t>> &adj, std::int32_t *labels) {
    const auto clusterLabel = static_cast<std::int32_t>(m_clusterId);
    labels[seed] = clusterLabel;
    std::vector<std::size_t> frontier;
    frontier.reserve(adj[seed].size());
    for (const std::int32_t n : adj[seed]) {
      frontier.push_back(static_cast<std::size_t>(n));
    }

    while (!frontier.empty()) {
      std::vector<std::size_t> next;
      next.reserve(frontier.size());
      for (const std::size_t point : frontier) {
        if (labels[point] != UNCLASSIFIED) {
          continue;
        }
        labels[point] = clusterLabel;
        if (isCore[point] == 0) {
          continue;
        }
        for (const std::int32_t n : adj[point]) {
          if (labels[static_cast<std::size_t>(n)] == UNCLASSIFIED) {
            next.push_back(static_cast<std::size_t>(n));
          }
        }
      }
      frontier.swap(next);
    }
  }

  T m_eps;              ///< Density-neighbourhood radius.
  std::size_t m_minPts; ///< Minimum neighbour count for a core point.
  std::size_t m_nJobs;  ///< Worker count clamped at construction via @ref math::clampedJobCount.
  std::size_t m_clusterId = 0; ///< Next cluster id to assign; also the final cluster count.
  /// Lazy worker pool. Emplaced inside @ref run on the first call whose shape clears
  /// @ref math::shouldSpawnPool; reused across subsequent runs so repeat fits on the same
  /// @c DBSCAN instance skip the thread-spawn cost.
  std::optional<BS::light_thread_pool> m_pool;
  /// Dense label buffer. Reallocated inside @ref run only when @c n differs from the previous
  /// call's size; the `[0, n)` range is overwritten each run with @ref UNCLASSIFIED, then
  /// filled with cluster ids (or @ref NOISY) by the expansion sweep.
  NDArray<std::int32_t, 1> m_labels;
};

} // namespace clustering
