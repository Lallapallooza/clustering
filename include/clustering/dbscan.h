#pragma once

#include <vector>
#include <algorithm>

#include "tsl/hopscotch_set.h"
#include "thread_pool.h"

#include "clustering/kdtree.h"

#if defined(__GNUC__) || defined(__clang__)
  #define CLUSTERING_UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
  #define CLUSTERING_UNLIKELY(x) (x)
#endif


/**
 * @brief Implements the DBSCAN clustering algorithm.
 *
 * DBSCAN (Density-Based Spatial Clustering of Applications with Noise) is a popular
 * clustering algorithm which is fundamentally very different from k-means.
 * It focuses on the density of data points in a region and connects regions of
 * sufficient density into clusters. It can also identify outliers (noise) in the data.
 *
 * @tparam T The data type of the points. It must support basic arithmetic operations.
 */
template<class T, class QueryModel=KDTree<T, KDTreeDistanceType::kEucledian>>
class DBSCAN {
 public:
  static constexpr int UNCLASSIFIED = -2; ///< Constant to represent an unclassified point.
  static constexpr int NOISY        = -1; ///< Constant to represent a noisy point (outlier).

  /**
   * @brief Constructs a DBSCAN object with given parameters.
   *
   * @param points NDArray of data points where each row is a data point and columns are features.
   * @param eps The radius of neighborhood around a point.
   * @param minPts The minimum number of points required to form a dense region (core point).
   */
  DBSCAN(const NDArray<T, 2> &points, T eps, size_t minPts, size_t n_jobs = std::thread::hardware_concurrency())
    : m_points(points),
      m_points_dim0(m_points.dim(0)),
      m_points_dim1(m_points.dim(1)),
      m_eps(eps),
      m_minPts(minPts),
      m_clusterId(0),
      m_labels(m_points_dim0),
      m_query_model(m_points),
      m_thread_pool(n_jobs) {
    std::fill(m_labels.begin(), m_labels.end(), UNCLASSIFIED);
  }

  /**
   * @brief Runs the DBSCAN clustering algorithm.
   *
   * Identifies core points in parallel and then sequentially expands clusters
   * from these core points.
   */
  void run() {
    for (size_t i = 0; i < m_points_dim0; ++i) {
      if (m_labels[i] == UNCLASSIFIED) {
        if (isCorePoint(i)) {
          expandCluster(i);
          m_clusterId += static_cast<size_t>(m_labels[i] != NOISY);
        } else {
          m_labels[i] = NOISY;
        }
      }
    }
  }

  /**
   * @brief Returns the labels of each point after clustering.
   *
   * Each point in the dataset is assigned a cluster label or marked as noise.
   * @return Vector of labels where each element corresponds to a data point in the original dataset.
   */
  [[nodiscard]] const std::vector<std::atomic_int> &labels() const {
    return m_labels;
  }

  /**
   * @brief Returns the number of clusters identified by DBSCAN.
   *
   * @return Number of clusters.
   */
  [[nodiscard]] size_t nClusters() const {
    return m_clusterId;
  }

 private:
  const NDArray<T, 2>          &m_points;      ///< The dataset of points to be clustered.
  const size_t                 m_points_dim0; ///< Number of rows (data points) in m_points.
  const size_t                 m_points_dim1; ///< Number of columns (features) in m_points.
  T                            m_eps;         ///< Radius for neighborhood.
  const size_t                 m_minPts;      ///< Minimum number of points to form a core point.
  size_t                       m_clusterId;   ///< Current cluster ID being assigned.
  std::vector<std::atomic_int> m_labels;      ///< Labels assigned to each point in the dataset.
  QueryModel                   m_query_model;  ///< Query model  built from m_points for efficient querying.
  BS::thread_pool              m_thread_pool; ///< ThreadPool for parallel execution.


  /**
   * @brief Checks if a point at a given index is a core point.
   *
   * A core point has at least m_minPts points within m_eps radius.
   * @param idx Index of the point in m_points.
   * @return True if the point is a core point, false otherwise.
   */
  [[nodiscard]] bool isCorePoint(size_t idx) const {
    NDArray<T, 1> queryPoint = extractPoint(idx);
    auto          neighbors  = m_query_model.query(queryPoint, m_eps, m_minPts);
    return neighbors.size() == m_minPts;
  }

  /**
   * @brief Expands a cluster from a given core point.
   *
   * This function implements the core part of the DBSCAN clustering algorithm. Starting from a core point,
   * it expands the cluster by iteratively adding all points that are density-reachable from the core point
   * within the epsilon (m_eps) radius and have a sufficient number of points (minPts) in their neighborhoods.
   *
   * @param idx The index of the core point from which to expand the cluster.
   */
  void expandCluster(size_t idx) {
    // Initial query to find all points within epsilon radius of the core point.
    std::vector<size_t> seeds_vec = m_query_model.query(extractPoint(idx), m_eps);
    std::mutex          mutex;

    // Loop until there are no more points to add to the cluster.
    while (!seeds_vec.empty()) {
      std::vector<size_t> seeds_vec_cpy; // Temporary vector to hold new seeds found in this iteration.
      seeds_vec_cpy.reserve(256);

      auto multi_fut = m_thread_pool.parallelize_loop(0, seeds_vec.size(),
        [this, &seeds_vec, &seeds_vec_cpy, &mutex](size_t start, size_t end) {
          std::vector<size_t> local_seeds_vec_cpy;  // Local vector to accumulate seeds found by this thread.
          tsl::hopscotch_pg_set<size_t> local_uniq; // Set to ensure unique addition of points to the cluster.

          for (size_t i = start; i < end; ++i) {
            size_t point = seeds_vec[i];

            // Skip already classified points.
            if (m_labels[point] != UNCLASSIFIED) {
              continue;
            }

            // Query to find neighbors of the current point within epsilon radius.
            NDArray<T, 1> query     = extractPoint(point);
            auto          neighbors = m_query_model.query(query, m_eps);

            // Mark point as NOISY if it doesn't have enough neighbors to be a core point.
            if (CLUSTERING_UNLIKELY(neighbors.size() < m_minPts)) {
              m_labels[point] = NOISY;
              continue;
            }
            m_labels[point] = m_clusterId;

            // Iterate over neighbors and add them to the list of points to be processed if not already processed.
            for (size_t neighbor_id: neighbors) {
              if (m_labels[neighbor_id] == UNCLASSIFIED && local_uniq.insert(neighbor_id).second) {
                local_seeds_vec_cpy.push_back(neighbor_id);
              }
            }
          }

          std::lock_guard lock{mutex};
          seeds_vec_cpy.insert(seeds_vec_cpy.end(), local_seeds_vec_cpy.begin(),local_seeds_vec_cpy.end());
      });

      multi_fut.wait();

      // Prepare for the next iteration.
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
  inline NDArray<T, 1> extractPoint(size_t idx) const {
    NDArray<T, 1> point({m_points_dim1});

    for (size_t i = 0; i < m_points_dim1; ++i) {
      point[i] = m_points[idx][i];
    }
    return point;
  }
};
