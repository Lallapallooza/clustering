#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

#include "clustering/always_assert.h"
#include "clustering/index/auto_range_index.h"
#include "clustering/index/range_query.h"
#include "clustering/math/dsu.h"
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
 *       changes and borrow a worker pool from the process-wide shared registry on
 *       parallel-eligible shapes.
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
  static constexpr std::int32_t UNCLASSIFIED = -2; ///< Sentinel for a component without a
                                                   ///< cluster id yet; never an output label.
  static constexpr std::int32_t NOISY = -1; ///< Label assigned to points that no cluster claimed.

  /**
   * @brief Construct a reusable DBSCAN fitter.
   *
   * @param eps    Radius of the density neighbourhood used to test reachability.
   * @param minPts Minimum neighbour count (including self) that marks a core point.
   * @param nJobs  Worker-count upper bound for the range-index backend. A value of @c 0 is
   *               clamped upward to @c std::thread::hardware_concurrency() so the pool is always
   *               usable by the @ref math::Pool helpers.
   */
  explicit DBSCAN(T eps, std::size_t minPts, std::size_t nJobs = 0)
      : m_eps(eps), m_minPts(minPts), m_nJobs(math::clampedJobCount(nJobs)), m_labels({0}) {
    CLUSTERING_ALWAYS_ASSERT(minPts >= 1);
    // Pool acquisition is deferred to @ref run: at small shapes every hot phase gates serial, so
    // a fully-serial fit should not force the shared registry to materialize a worker pool. The
    // shape-gated borrow from @ref math::sharedPool happens inside @ref run.
  }

  DBSCAN(const DBSCAN &) = delete;
  DBSCAN &operator=(const DBSCAN &) = delete;
  DBSCAN(DBSCAN &&) = delete;
  DBSCAN &operator=(DBSCAN &&) = delete;
  ~DBSCAN() = default;

  /**
   * @brief Fit to @p X.
   *
   * Queries the backend for the full eps-neighbourhood adjacency, derives core-point flags from
   * per-row degrees, unions core-core edges into connected components, then labels every point:
   * cores carry their component id, border points take the lowest cluster id among their adjacent
   * cores, and the rest are @ref NOISY.
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
    const std::size_t poolJobs = effectiveWorkerCount(n, X.dim(1), m_nJobs);
    const math::Pool pool{math::shouldSpawnPool(n, poolJobs, /*minOpsPerWorker=*/16)
                              ? &math::sharedPool(poolJobs)
                              : nullptr};

    // Pool-aware query models parallelize their own construction (the KDTree build forks
    // subtrees); models without that constructor keep the plain shape contract.
    QueryModel queryModel = [&] {
      if constexpr (std::is_constructible_v<QueryModel, const NDArray<T, 2> &, math::Pool>) {
        return QueryModel(X, pool);
      } else {
        return QueryModel(X);
      }
    }();
    const auto adj = queryModel.query(m_eps, pool);

    // Core flag per point: degree (adjacency size, counts self) at or above minPts. Each entry is
    // an independent size read with a disjoint write.
    std::vector<std::uint8_t> isCore(n, 0);
    for (std::size_t i = 0; i < n; ++i) {
      isCore[i] = (adj[i].size() >= m_minPts) ? std::uint8_t{1} : std::uint8_t{0};
    }

    // Connected components over core-core edges: density-reachability is the transitive closure of
    // "core within eps of core", which a disjoint-set union builds directly.
    const std::vector<std::uint32_t> componentRoots = buildComponentRoots(adj, isCore, pool);

    // Dense cluster ids in first-core-index order so the lowest-index core of each component names
    // its cluster. Writing each core's id into the label buffer now freezes it so the border pass
    // reads it without another root lookup.
    std::int32_t *labels = m_labels.data();
    std::vector<std::int32_t> rootCluster(n, UNCLASSIFIED);
    for (std::size_t i = 0; i < n; ++i) {
      if (isCore[i] == 0) {
        continue;
      }
      const auto root = componentRoots[i];
      std::int32_t &slot = rootCluster[root];
      if (slot == UNCLASSIFIED) {
        slot = static_cast<std::int32_t>(m_clusterId);
        ++m_clusterId;
      }
      labels[i] = slot;
    }

    assignBorderAndNoise(adj, isCore);
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

  static constexpr std::size_t kSmall2dMaxParallelN = 50'000;
  static constexpr std::size_t kSmall2dWorkerCap = 8;

  [[nodiscard]] static std::size_t effectiveWorkerCount(std::size_t n, std::size_t d,
                                                        std::size_t requested) noexcept {
    if (d <= 2 && n <= kSmall2dMaxParallelN && requested > kSmall2dWorkerCap) {
      return kSmall2dWorkerCap;
    }
    return requested;
  }

  /**
   * @brief Per-point component roots over the core-core eps-graph.
   *
   * Once the range queries fan out, the component build is the dominant serial cost at low
   * dimension, where dense neighbourhoods make the core-core edge set far larger than @p n. The
   * parallel path folds edges into one shared @ref AtomicUnionFind: merges linearize through a
   * compare-and-swap on the losing root's parent slot, so workers share the structure with no
   * per-worker copies and no serial fold afterwards. Connectivity is order independent, and the
   * union-by-min link rule pins each parallel component root to its minimum member, so the
   * flattened roots are reproducible across thread interleavings. Root identity only keys the
   * first-core-order cluster id mapping in @ref run, which depends on the partition alone.
   *
   * @param adj    Eps-adjacency list indexed by point row.
   * @param isCore Per-point core flag.
   * @param pool   Worker pool borrowed by @ref run, or an unset handle for serial execution.
   * @return Length-@c n root array; entries of the same density-reachable component share a root.
   */
  static std::vector<std::uint32_t>
  buildComponentRoots(const std::vector<std::vector<std::int32_t>> &adj,
                      const std::vector<std::uint8_t> &isCore, math::Pool pool) {
    const std::size_t n = adj.size();
    std::vector<std::uint32_t> roots(n);

    const auto uniteRange = [&](auto &dsu, std::size_t lo, std::size_t hi) {
      for (std::size_t i = lo; i < hi; ++i) {
        if (isCore[i] == 0) {
          continue;
        }
        const auto iu = static_cast<std::uint32_t>(i);
        for (const std::int32_t neighbor : adj[i]) {
          const auto j = static_cast<std::size_t>(neighbor);
          if (j > i && isCore[j] != 0) {
            dsu.unite(iu, static_cast<std::uint32_t>(j));
          }
        }
      }
    };

    const std::size_t workers = pool.workerCount();
    if (pool.pool == nullptr || workers <= 1) {
      UnionFind<std::uint32_t> components(n);
      uniteRange(components, 0, n);
      for (std::size_t i = 0; i < n; ++i) {
        roots[i] = components.find(static_cast<std::uint32_t>(i));
      }
      return roots;
    }

    // More blocks than workers lets dynamic stealing balance the skewed per-row degree; the
    // shared structure makes block placement irrelevant to the result.
    AtomicUnionFind<std::uint32_t> components(n);
    pool.parallelForBlocks(std::size_t{0}, n, workers * 4,
                           [&](std::size_t lo, std::size_t hi) { uniteRange(components, lo, hi); });
    pool.parallelForBlocks(std::size_t{0}, n, std::size_t{0}, [&](std::size_t lo, std::size_t hi) {
      for (std::size_t i = lo; i < hi; ++i) {
        roots[i] = components.find(static_cast<std::uint32_t>(i));
      }
    });
    return roots;
  }

  /**
   * @brief Label every non-core point: borders take the lowest cluster id among adjacent cores,
   *        the rest are @ref NOISY.
   *
   * Core labels were frozen into the buffer by the dense-id pass, so this pass reads them and the
   * immutable @p adj / @p isCore. A border within eps of cores from several clusters takes the
   * lowest cluster id, a deterministic tie-break so both adjacency backends agree on the partition.
   *
   * @param adj    Eps-adjacency list indexed by point row.
   * @param isCore Per-point core flag.
   */
  void assignBorderAndNoise(const std::vector<std::vector<std::int32_t>> &adj,
                            const std::vector<std::uint8_t> &isCore) {
    const std::size_t n = adj.size();
    std::int32_t *labels = m_labels.data();
    for (std::size_t p = 0; p < n; ++p) {
      if (isCore[p] != 0) {
        continue;
      }
      std::int32_t best = NOISY;
      for (const std::int32_t neighbor : adj[p]) {
        const auto q = static_cast<std::size_t>(neighbor);
        if (isCore[q] == 0) {
          continue;
        }
        const std::int32_t cluster = labels[q];
        if (best == NOISY || cluster < best) {
          best = cluster;
        }
      }
      labels[p] = best;
    }
  }

  T m_eps;                     ///< Density-neighbourhood radius.
  std::size_t m_minPts;        ///< Minimum neighbour count for a core point.
  std::size_t m_nJobs;         ///< Worker-count upper bound from @ref math::clampedJobCount.
  std::size_t m_clusterId = 0; ///< Next cluster id to assign; also the final cluster count.
  /// Dense label buffer. Reallocated inside @ref run only when @c n differs from the previous
  /// call's size; every entry in `[0, n)` is rewritten each run with a cluster id (cores and
  /// claimed borders) or @ref NOISY, so no stale label survives across calls.
  NDArray<std::int32_t, 1> m_labels;
};

} // namespace clustering
