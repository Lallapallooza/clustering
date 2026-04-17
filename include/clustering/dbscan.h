#pragma once

#include <BS_thread_pool.hpp>
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <vector>

#include "clustering/index/kdtree.h"
#include "clustering/index/range_query.h"

#if defined(__GNUC__) || defined(__clang__)
#define CLUSTERING_UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
#define CLUSTERING_UNLIKELY(x) (x)
#endif

namespace clustering {

/**
 * @brief Implements the DBSCAN clustering algorithm.
 *
 * DBSCAN (Density-Based Spatial Clustering of Applications with Noise) groups points by
 * local density: regions of sufficient density become clusters, and points that fall outside
 * any such region are labeled as noise.
 *
 * @tparam T The data type of the points. It must support basic arithmetic operations.
 * @tparam QueryModel Spatial index satisfying @ref clustering::index::RangeQuery. Defaults to
 *         @c KDTree<T, KDTreeDistanceType::kEucledian>.
 */
template <class T, class QueryModel = KDTree<T, KDTreeDistanceType::kEucledian>>
  requires index::RangeQuery<QueryModel, T>
class DBSCAN {
public:
  static constexpr int UNCLASSIFIED = -2; ///< Constant to represent an unclassified point.
  static constexpr int NOISY = -1;        ///< Constant to represent a noisy point (outlier).

  /**
   * @brief Constructs a DBSCAN object with given parameters.
   *
   * @param points NDArray of data points where each row is a data point and columns are features.
   * @param eps The radius of neighborhood around a point.
   * @param minPts The minimum number of points required to form a dense region (core point).
   */
  DBSCAN(const NDArray<T, 2> &points, T eps, size_t minPts,
         size_t n_jobs = std::thread::hardware_concurrency())
      : m_points(points), m_points_dim0(m_points.dim(0)), m_points_dim1(m_points.dim(1)),
        m_eps(eps), m_minPts(minPts), m_labels(m_points_dim0), m_seen_wave(m_points_dim0),
        m_query_model(m_points), m_thread_pool(n_jobs) {
    std::fill(m_labels.begin(), m_labels.end(), UNCLASSIFIED);
  }

  /**
   * @brief Runs the DBSCAN clustering algorithm.
   *
   * Identifies core points sequentially and then expands clusters in parallel
   * from these core points.
   */
  void run() {
    // Phase 1: Parallel core-point identification
    std::vector<uint8_t> is_core(m_points_dim0, 0);
    m_thread_pool
        .submit_blocks(size_t{0}, m_points_dim0,
                       [this, &is_core](size_t start, size_t end) {
                         for (size_t i = start; i < end; ++i) {
                           if (isCorePoint(i)) {
                             is_core[i] = 1;
                           }
                         }
                       })
        .wait();

    // Phase 2: Sequential cluster expansion from core points only
    for (size_t i = 0; i < m_points_dim0; ++i) {
      if (m_labels[i] == UNCLASSIFIED && is_core[i] != 0) {
        expandCluster(i, is_core);
        ++m_clusterId;
      }
    }

    // Phase 3: Mark remaining UNCLASSIFIED points as NOISY
    for (size_t i = 0; i < m_points_dim0; ++i) {
      if (m_labels[i] == UNCLASSIFIED) {
        m_labels[i] = NOISY;
      }
    }
  }

  /**
   * @brief Returns the labels of each point after clustering.
   *
   * Each point in the dataset is assigned a cluster label or marked as noise.
   * @return Vector of labels where each element corresponds to a data point in the original
   * dataset.
   */
  [[nodiscard]] const std::vector<std::atomic_int> &labels() const { return m_labels; }

  /**
   * @brief Returns the number of clusters identified by DBSCAN.
   *
   * @return Number of clusters.
   */
  [[nodiscard]] size_t nClusters() const { return m_clusterId; }

private:
  const NDArray<T, 2> &m_points;         ///< The dataset of points to be clustered.
  const size_t m_points_dim0;            ///< Number of rows (data points) in m_points.
  const size_t m_points_dim1;            ///< Number of columns (features) in m_points.
  T m_eps;                               ///< Radius for neighborhood.
  const size_t m_minPts;                 ///< Minimum number of points to form a core point.
  size_t m_clusterId = 0;                ///< Current cluster ID being assigned.
  std::vector<std::atomic_int> m_labels; ///< Labels assigned to each point in the dataset.
  std::vector<std::atomic<uint32_t>>
      m_seen_wave;                     ///< Per-point epoch marker for BFS-frontier dedup.
  uint32_t m_current_wave = 0;         ///< Monotonic counter; bumped before each frontier wave.
  QueryModel m_query_model;            ///< Query model built from m_points for efficient querying.
  BS::light_thread_pool m_thread_pool; ///< ThreadPool for parallel execution.

  /**
   * @brief Checks if a point at a given index is a core point.
   *
   * A core point has at least m_minPts points within m_eps radius.
   * @param idx Index of the point in m_points.
   * @return True if the point is a core point, false otherwise.
   */
  [[nodiscard]] bool isCorePoint(size_t idx) const {
    const NDArray<T, 1> queryPoint = extractPoint(idx);
    const auto neighbors =
        m_query_model.query(queryPoint, m_eps, static_cast<std::int64_t>(m_minPts));
    return neighbors.size() == m_minPts;
  }

  /**
   * @brief Expands a cluster from a given core point.
   *
   * This function implements the core part of the DBSCAN clustering algorithm. Starting from a core
   * point, it expands the cluster by iteratively adding all points that are density-reachable from
   * the core point within the epsilon (m_eps) radius and have a sufficient number of points
   * (minPts) in their neighborhoods.
   *
   * @param idx The index of the core point from which to expand the cluster.
   */
  void expandCluster(size_t idx, const std::vector<uint8_t> &is_core) {
    m_labels[idx] = m_clusterId;

    std::vector<size_t> seeds_vec = m_query_model.query(extractPoint(idx), m_eps);
    std::mutex mutex;

    while (!seeds_vec.empty()) {
      std::vector<size_t> seeds_vec_cpy;
      seeds_vec_cpy.reserve(seeds_vec.size());

      const uint32_t wave = ++m_current_wave;

      auto multi_fut = m_thread_pool.submit_blocks(
          size_t{0}, seeds_vec.size(),
          [this, &seeds_vec, &seeds_vec_cpy, &mutex, &is_core, wave](size_t start, size_t end) {
            std::vector<size_t> local_seeds_vec_cpy;

            for (size_t i = start; i < end; ++i) {
              const size_t point = seeds_vec[i];

              if (m_labels[point] != UNCLASSIFIED) {
                continue;
              }

              // Assign cluster label to ALL density-reachable points (core and border)
              m_labels[point] = m_clusterId;

              // Only core points expand further
              if (is_core[point] == 0) {
                continue;
              }

              const NDArray<T, 1> queryPoint = extractPoint(point);
              const auto neighbors = m_query_model.query(queryPoint, m_eps);

              for (const size_t neighbor_id : neighbors) {
                if (m_labels[neighbor_id] != UNCLASSIFIED) {
                  continue;
                }
                // Epoch bitmap dedup. Plain relaxed load/store: avoids the locked
                // xchg that contends across threads. Two threads can race and both
                // push -- the next wave's labels-check absorbs the duplicate.
                if (m_seen_wave[neighbor_id].load(std::memory_order_relaxed) != wave) {
                  m_seen_wave[neighbor_id].store(wave, std::memory_order_relaxed);
                  local_seeds_vec_cpy.push_back(neighbor_id);
                }
              }
            }

            const std::scoped_lock lock{mutex};
            seeds_vec_cpy.insert(seeds_vec_cpy.end(), local_seeds_vec_cpy.begin(),
                                 local_seeds_vec_cpy.end());
          });

      multi_fut.wait();
      seeds_vec.swap(seeds_vec_cpy);
    }
  }

  /**
   * @brief Extracts a single point from m_points.
   *
   * Given an index, extracts the point at that index into a new NDArray.
   * @param idx Index of the point in m_points.
   * @return NDArray representing the extracted point.
   */
  NDArray<T, 1> extractPoint(size_t idx) const {
    NDArray<T, 1> point({m_points_dim1});

    for (size_t i = 0; i < m_points_dim1; ++i) {
      point[i] = m_points[idx][i];
    }
    return point;
  }
};

} // namespace clustering
