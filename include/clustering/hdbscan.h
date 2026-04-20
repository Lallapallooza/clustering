#pragma once

#include <BS_thread_pool.hpp>
#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <type_traits>
#include <vector>

#include "clustering/always_assert.h"
#include "clustering/math/thread.h"
#include "clustering/ndarray.h"

namespace clustering::hdbscan {

/**
 * @brief Cluster extraction method on the condensed tree.
 *
 * The two methods differ in traversal behaviour, not in output shape. Excess-of-mass is the
 * default and optimizes per-cluster stability over the full tree; leaf returns every leaf of
 * the condensed tree as its own cluster.
 */
enum class ClusterSelectionMethod : std::uint8_t {
  kEom,  ///< Excess-of-mass selection (the published default).
  kLeaf, ///< Leaf-cluster selection; every condensed-tree leaf becomes a cluster.
};

/**
 * @brief One edge of the minimum spanning tree of mutual-reachability distances.
 *
 * Endpoints are @c std::int32_t because the pipeline contract caps @c N at the signed 32-bit
 * range. The weight is the mutual-reachability distance between the two endpoints under the
 * configured @c minSamples.
 *
 * @tparam T Element type of the point cloud. Only @c float is supported; a @c double
 *         specialization is out of scope.
 */
template <class T> struct MstEdge {
  std::int32_t u = 0;
  std::int32_t v = 0;
  T weight = T{};
};

/**
 * @brief Frozen output contract of every MST backend.
 *
 * The MST boundary is the one axis of variation across backends; everything downstream
 * (single-linkage tree, condensed tree, cluster extraction, outlier scoring) is monomorphic and
 * reads from this shape. Fields default to well-defined empty values so a @c MstOutput produced
 * by the default constructor is already in a valid "no fit yet" state.
 *
 * @tparam T Element type of the point cloud.
 */
template <class T> struct MstOutput {
  /// The @c N - 1 MST edges, in insertion order.
  std::vector<MstEdge<T>> edges;
  /// Per-point core distance (length @c N; self-excluded kNN distance at @c minSamples).
  NDArray<T, 1> coreDistances{std::array<std::size_t, 1>{0}};
};

/**
 * @brief Contract for an MST backend satisfying the frozen @ref MstOutput shape.
 *
 * A backend is default-constructible and exposes a single @c run entry point that consumes the
 * input dataset, the @c minSamples parameter, and a worker-pool handle, and writes its result
 * into a caller-provided @ref MstOutput. Backends own their private scratch and may amortize
 * shape-indexed buffers across calls; per the HDBSCAN class invariant, data-dependent indices
 * (KDTree, kNN graph) are rebuilt per fit.
 *
 * @tparam B Candidate backend type.
 * @tparam T Element type of the point cloud.
 */
template <class B, class T>
concept MstBackendStrategy = std::default_initializable<B> &&
                             requires(B &backend, const NDArray<T, 2> &X, std::size_t minSamples,
                                      math::Pool pool, MstOutput<T> &out) {
                               { backend.run(X, minSamples, pool, out) };
                             };

} // namespace clustering::hdbscan

namespace clustering {

/**
 * @brief Hierarchical density-based clustering over mutual-reachability distances.
 *
 * HDBSCAN* extends DBSCAN with a hierarchical condensation step that auto-selects density
 * thresholds, produces per-cluster stability, and yields GLOSH outlier scores as a byproduct.
 * The MST boundary is the only template axis; everything downstream (condensed tree, cluster
 * extraction, outlier scoring) is monomorphic. Callers pin a specific backend via @p MstBackend
 * to control the time / memory / approximation trade-off.
 *
 * @note @c HDBSCAN does NOT own @p X. The caller must keep the @c NDArray alive for the duration
 *       of every @ref run call. Data-dependent indices (KDTree, kNN graph) are rebuilt on every
 *       fit so in-place buffer mutations through a borrowed view can never produce a silent
 *       cache miss. Shape-indexed scratch (heaps, reusable buffers) may be amortized at fixed
 *       @c (n, d, minSamples); a shape change rebuilds. @ref reset returns the instance to
 *       fresh-constructed state.
 *
 * @note On a freshly-constructed or just- @ref reset instance, all result accessors return empty
 *       values (an empty label array, empty outlier-score array, zero cluster count, and an
 *       empty condensed-tree view).
 *
 * @tparam T          Element type. Only @c float is supported in this class; a @c double
 *                    specialization is out of scope.
 * @tparam MstBackend Backend satisfying @ref hdbscan::MstBackendStrategy. Must be pinned
 *                    explicitly by the caller.
 */
template <class T, class MstBackend>
  requires hdbscan::MstBackendStrategy<MstBackend, T>
class HDBSCAN {
  static_assert(std::is_same_v<T, float>,
                "HDBSCAN<T> supports only float; a double specialization is out of scope.");

public:
  /**
   * @brief Read-only view over the condensed-tree result.
   *
   * The parallel-array layout matches the reference implementation -- parent, child,
   * lambda-at-merge, and child-size -- so excess-of-mass and leaf extraction can walk it without
   * auxiliary structures.
   */
  struct CondensedTreeView {
    std::span<const std::int32_t> parent;
    std::span<const std::int32_t> child;
    std::span<const T> lambda;
    std::span<const std::int32_t> childSize;

    [[nodiscard]] bool empty() const noexcept { return parent.empty(); }
    [[nodiscard]] std::size_t size() const noexcept { return parent.size(); }
  };

  /**
   * @brief Construct a reusable HDBSCAN fitter.
   *
   * @param minClusterSize The smallest allowable cluster; must be at least 2.
   * @param minSamples     Neighbour count used to compute core distances. A value of @c 0 is a
   *                       sentinel meaning "resolve to @c minClusterSize at fit time"; the fit
   *                       entry asserts the resolved value is positive and strictly less than
   *                       @c N.
   * @param method         Cluster selection method; defaults to excess-of-mass.
   * @param nJobs          Worker count for the internal thread pool. A value of @c 0 is clamped
   *                       upward to @c std::thread::hardware_concurrency().
   */
  explicit HDBSCAN(std::size_t minClusterSize, std::size_t minSamples = 0,
                   hdbscan::ClusterSelectionMethod method = hdbscan::ClusterSelectionMethod::kEom,
                   std::size_t nJobs = 0)
      : m_minClusterSize(minClusterSize), m_minSamples(minSamples), m_method(method),
        m_nJobs(math::clampedJobCount(nJobs)), m_labels({0}), m_outlierScores({0}) {
    CLUSTERING_ALWAYS_ASSERT(minClusterSize >= 2);
    // Defer pool construction to @ref run: at small shapes every hot phase gates serial, and
    // spawning nJobs workers in the ctor burns thread-create overhead that the fit never
    // amortizes at small shapes. @ref run emplaces on first need and reuses.
  }

  HDBSCAN(const HDBSCAN &) = delete;
  HDBSCAN &operator=(const HDBSCAN &) = delete;
  HDBSCAN(HDBSCAN &&) = delete;
  HDBSCAN &operator=(HDBSCAN &&) = delete;
  ~HDBSCAN() = default;

  /**
   * @brief Fit to @p X.
   *
   * Every precondition fires a @c CLUSTERING_ALWAYS_ASSERT before any work begins so failures
   * surface at the call site regardless of build configuration.
   *
   * @param X Contiguous n x d dataset. The caller retains ownership; @p X must outlive this
   *          @c run call.
   *
   * @warning @p X must remain alive and unchanged for the full duration of this call.
   */
  void run(const NDArray<T, 2> &X) {
    const std::size_t n = X.dim(0);

    CLUSTERING_ALWAYS_ASSERT(m_minClusterSize >= 2);

    const std::size_t effectiveMinSamples = (m_minSamples == 0) ? m_minClusterSize : m_minSamples;
    CLUSTERING_ALWAYS_ASSERT(effectiveMinSamples >= 1);
    CLUSTERING_ALWAYS_ASSERT(effectiveMinSamples < n);
    CLUSTERING_ALWAYS_ASSERT(n >= m_minClusterSize);
    CLUSTERING_ALWAYS_ASSERT(n <=
                             static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()));

    (void)m_backend;
    (void)m_mstOutput;
    (void)m_pool;
    (void)m_method;
    (void)m_nJobs;
  }

  /// Length-n assignment; @c -1 marks noise. Empty on a freshly-constructed or just- @ref reset
  /// instance.
  [[nodiscard]] const NDArray<std::int32_t, 1> &labels() const noexcept { return m_labels; }

  /// Length-n per-point GLOSH outlier scores in @c [0, 1]. Empty on a freshly-constructed or
  /// just- @ref reset instance.
  [[nodiscard]] const NDArray<T, 1> &outlierScores() const noexcept { return m_outlierScores; }

  /// Total number of clusters discovered by the most recent @ref run, or @c 0 if no fit has
  /// produced a result yet.
  [[nodiscard]] std::size_t nClusters() const noexcept { return m_nClusters; }

  /// Borrowed view over the condensed tree from the most recent @ref run, or an empty view if
  /// no fit has produced a result yet.
  [[nodiscard]] CondensedTreeView condensedTree() const noexcept {
    return CondensedTreeView{
        .parent = std::span<const std::int32_t>(m_ctParent.data(), m_ctParent.size()),
        .child = std::span<const std::int32_t>(m_ctChild.data(), m_ctChild.size()),
        .lambda = std::span<const T>(m_ctLambda.data(), m_ctLambda.size()),
        .childSize = std::span<const std::int32_t>(m_ctChildSize.data(), m_ctChildSize.size()),
    };
  }

  /// Release every scratch buffer. The next @ref run call reallocates against its shape.
  void reset() {
    m_labels = NDArray<std::int32_t, 1>({0});
    m_outlierScores = NDArray<T, 1>({0});
    m_nClusters = 0;
    m_ctParent = std::vector<std::int32_t>{};
    m_ctChild = std::vector<std::int32_t>{};
    m_ctLambda = std::vector<T>{};
    m_ctChildSize = std::vector<std::int32_t>{};
    m_mstOutput = hdbscan::MstOutput<T>{};
    m_backend = MstBackend{};
  }

private:
  std::size_t m_minClusterSize;
  std::size_t m_minSamples;
  hdbscan::ClusterSelectionMethod m_method;
  std::size_t m_nJobs;
  /// Lazy worker pool. Emplaced inside @ref run on the first call whose shape clears
  /// @ref math::shouldSpawnPool; reused across subsequent runs so repeat fits on the same
  /// @c HDBSCAN instance skip the thread-spawn cost.
  std::optional<BS::light_thread_pool> m_pool;
  NDArray<std::int32_t, 1> m_labels;
  NDArray<T, 1> m_outlierScores;
  std::size_t m_nClusters = 0;

  // Condensed-tree parallel arrays, filled by the post-MST pipeline and surfaced through
  // @ref condensedTree as a read-only view.
  std::vector<std::int32_t> m_ctParent;
  std::vector<std::int32_t> m_ctChild;
  std::vector<T> m_ctLambda;
  std::vector<std::int32_t> m_ctChildSize;

  MstBackend m_backend{};
  hdbscan::MstOutput<T> m_mstOutput{};
};

} // namespace clustering
