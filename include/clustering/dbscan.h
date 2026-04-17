#pragma once

#include <BS_thread_pool.hpp>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <thread>
#include <vector>

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
 * @tparam T The data type of the points.
 * @tparam QueryModel Range-index backend. Defaults to @ref clustering::index::AutoRangeIndex,
 *                    which picks a KD-Tree below @c bruteForceDimFloor and a blocked pairwise
 *                    sweep at or above it.
 */
template <class T, class QueryModel = index::AutoRangeIndex<T>>
  requires index::RangeIndex<QueryModel, T>
class DBSCAN {
public:
  static constexpr int UNCLASSIFIED = -2; ///< Sentinel for a point not yet visited.
  static constexpr int NOISY = -1;        ///< Label assigned to points that no cluster claimed.

  /**
   * @brief Constructs a DBSCAN fit context over a borrowed point matrix.
   *
   * @param points Row-major @c n x @c d point matrix. Must outlive the instance.
   * @param eps Radius of the density neighbourhood used to test reachability.
   * @param minPts Minimum neighbour count (including self) that marks a core point.
   * @param n_jobs Worker count for the range-index backend; defaults to hardware concurrency.
   */
  DBSCAN(const NDArray<T, 2> &points, T eps, std::size_t minPts,
         std::size_t n_jobs = std::thread::hardware_concurrency())
      : m_eps(eps), m_minPts(minPts), m_labels(points.dim(0), UNCLASSIFIED), m_thread_pool(n_jobs),
        m_query_model(points) {}

  /**
   * @brief Clusters the point cloud in place, populating @ref labels and @ref nClusters.
   *
   * Queries the backend for the full eps-neighbourhood adjacency, derives core-point flags
   * from per-row degrees, then expands clusters sequentially via graph BFS over the adjacency.
   * Remaining unclassified points are marked @ref NOISY.
   */
  void run() {
    const auto adj = m_query_model.query(m_eps, math::Pool{&m_thread_pool});
    const std::size_t n = adj.size();

    std::vector<std::uint8_t> isCore(n, 0);
    for (std::size_t i = 0; i < n; ++i) {
      isCore[i] = (adj[i].size() >= m_minPts) ? 1U : 0U;
    }

    for (std::size_t i = 0; i < n; ++i) {
      if (m_labels[i] == UNCLASSIFIED && isCore[i] != 0) {
        expandCluster(i, isCore, adj);
        ++m_clusterId;
      }
    }

    std::replace(m_labels.begin(), m_labels.end(), UNCLASSIFIED, NOISY);
  }

  /**
   * @brief Per-point cluster labels after @ref run; @ref NOISY marks outliers.
   *
   * @return Borrowed view valid until the @c DBSCAN instance is destroyed.
   */
  [[nodiscard]] const std::vector<int> &labels() const { return m_labels; }

  /**
   * @brief Total number of clusters discovered by the most recent @ref run.
   */
  [[nodiscard]] std::size_t nClusters() const { return m_clusterId; }

private:
  T m_eps;                     ///< Density-neighbourhood radius.
  std::size_t m_minPts;        ///< Minimum neighbour count for a core point.
  std::size_t m_clusterId = 0; ///< Next cluster id to assign; also the final cluster count.
  /// Per-point labels. Plain @c int is safe: only the backend's adjacency sweep fans work out
  /// to the pool; every subsequent label write happens on the calling thread inside a
  /// sequential cluster loop.
  std::vector<int> m_labels;
  BS::light_thread_pool m_thread_pool; ///< Worker pool handed to the range-index backend.
  QueryModel m_query_model;            ///< Backend that answers the eps-adjacency query.

  /**
   * @brief Expands a cluster from @p seed via BFS over the pre-computed adjacency.
   *
   * Per DBSCAN's density-reachability rule, border points are labelled into the cluster but do
   * not seed further expansion -- only core points contribute their neighbours to the frontier.
   *
   * @param seed   Index of the core point that starts the cluster.
   * @param isCore Per-point core-point flag derived from the adjacency row sizes.
   * @param adj    Borrowed eps-adjacency list indexed by point row.
   */
  void expandCluster(std::size_t seed, const std::vector<std::uint8_t> &isCore,
                     const std::vector<std::vector<std::int32_t>> &adj) {
    m_labels[seed] = static_cast<int>(m_clusterId);
    std::vector<std::size_t> frontier;
    frontier.reserve(adj[seed].size());
    for (const std::int32_t n : adj[seed]) {
      frontier.push_back(static_cast<std::size_t>(n));
    }

    while (!frontier.empty()) {
      std::vector<std::size_t> next;
      next.reserve(frontier.size());
      for (const std::size_t point : frontier) {
        if (m_labels[point] != UNCLASSIFIED) {
          continue;
        }
        m_labels[point] = static_cast<int>(m_clusterId);
        if (isCore[point] == 0) {
          continue;
        }
        for (const std::int32_t n : adj[point]) {
          if (m_labels[static_cast<std::size_t>(n)] == UNCLASSIFIED) {
            next.push_back(static_cast<std::size_t>(n));
          }
        }
      }
      frontier.swap(next);
    }
  }
};

} // namespace clustering
