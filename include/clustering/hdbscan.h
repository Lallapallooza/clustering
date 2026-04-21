#pragma once

#include <BS_thread_pool.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

#include "clustering/always_assert.h"
#include "clustering/hdbscan/detail/condensed_tree.h"
#include "clustering/hdbscan/detail/eom_extract.h"
#include "clustering/hdbscan/detail/glosh.h"
#include "clustering/hdbscan/detail/leaf_extract.h"
#include "clustering/hdbscan/detail/single_linkage.h"
#include "clustering/hdbscan/mst_backend.h"
#include "clustering/hdbscan/mst_output.h"
#include "clustering/hdbscan/policy/auto_mst_backend.h"
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

} // namespace clustering::hdbscan

namespace clustering {

/**
 * @brief Hierarchical density-based clustering over mutual-reachability distances.
 *
 * HDBSCAN* extends DBSCAN with a hierarchical condensation step that auto-selects density
 * thresholds, produces per-cluster stability, and yields GLOSH outlier scores as a byproduct.
 * The MST boundary is the only template axis; everything downstream (condensed tree, cluster
 * extraction, outlier scoring) is monomorphic. The default @p MstBackend is
 * @ref hdbscan::AutoMstBackend, which dispatches between Prim, Boruvka, and NN-Descent on the
 * input shape; callers who want to pin a specific backend may supply it as the second template
 * argument.
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
 * @note Labels and outlier scores follow the Campello 2015 formula over Euclidean
 *       mutual-reachability distances, matching the reference implementation.
 *
 * @par Thread safety
 * A single @c HDBSCAN instance is not safe to drive concurrently; @ref run mutates internal
 * state. Separate instances on distinct inputs are safe when each instance spawns its own
 * internal pool (the default). The internal pool obeys a no-nested-dispatch invariant:
 * worker tasks never re-submit to the pool.
 *
 * @tparam T          Element type. Only @c float is supported in this class; a @c double
 *                    specialization is out of scope.
 * @tparam MstBackend Backend satisfying @ref hdbscan::MstBackendStrategy. Defaults to
 *                    @ref hdbscan::AutoMstBackend which picks Prim, Boruvka, or NN-Descent on
 *                    input shape.
 */
template <class T, class MstBackend = hdbscan::AutoMstBackend<T>>
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

    // Lazy pool spawn: the MST backend is the only phase that can parallelise at all; post-MST is
    // serial by contract. Size the work gate by the backend's dominant shape (N * N) so small
    // inputs never pay the thread-create cost.
    if (!m_pool.has_value() &&
        math::shouldSpawnPool(n * n, m_nJobs, /*minOpsPerWorker=*/1U << 15)) {
      m_pool.emplace(m_nJobs);
    }
    const math::Pool pool{m_pool.has_value() ? &*m_pool : nullptr};

    // Phase 1: MST via the pinned backend. The backend writes edges and core distances into
    // `m_mstOutput`.
    m_backend.run(X, effectiveMinSamples, pool, m_mstOutput);

    // Convert squared distances to linear distances before the post-MST pipeline consumes them.
    // The backends store squared Euclidean internally (avoids an @c sqrt per pair-distance); the
    // MST structure is invariant under @c d -> @c sqrt(d) (monotone) but the condensed-tree
    // stability DP compares absolute lambda values whose outcome is not invariant under that
    // transform. Linearising here aligns the lambda scale with the reference implementation and
    // with outlier-score bounds users expect.
    {
      const std::size_t nCore = m_mstOutput.coreDistances.dim(0);
      T *coreData = m_mstOutput.coreDistances.data();
      for (std::size_t i = 0; i < nCore; ++i) {
        coreData[i] = std::sqrt(coreData[i]);
      }
      for (auto &edge : m_mstOutput.edges) {
        edge.weight = std::sqrt(edge.weight);
      }
    }

    // Phase 2: build the single-linkage dendrogram from the MST edges.
    hdbscan::detail::SingleLinkageTree<T> slt;
    hdbscan::detail::buildSingleLinkageTree(m_mstOutput, n, slt);

    // Phase 3: condense the dendrogram under `minClusterSize`.
    hdbscan::detail::CondensedTree<T> condensed;
    hdbscan::detail::condenseTree(slt, n, m_minClusterSize, condensed);

    // Phase 4: cluster extraction (EOM or leaf).
    std::vector<std::int32_t> labels;
    if (m_method == hdbscan::ClusterSelectionMethod::kEom) {
      hdbscan::detail::extractEom(condensed, n, labels);
    } else {
      hdbscan::detail::extractLeaf(condensed, n, labels);
    }

    // Phase 5: GLOSH outlier scores.
    std::vector<T> scores;
    hdbscan::detail::computeGlosh(condensed, n, labels, scores);

    // Finalise result accessors. The label array lands in the public NDArray buffer; ditto the
    // outlier-score array. The condensed tree is retained in its parallel-array form so the
    // public view can borrow from it without an additional copy.
    m_labels = NDArray<std::int32_t, 1>(std::array<std::size_t, 1>{n});
    std::int32_t maxLabel = -1;
    for (std::size_t i = 0; i < n; ++i) {
      m_labels(i) = labels[i];
      maxLabel = std::max(maxLabel, labels[i]);
    }
    m_outlierScores = NDArray<T, 1>(std::array<std::size_t, 1>{n});
    for (std::size_t i = 0; i < n; ++i) {
      m_outlierScores(i) = scores[i];
    }
    m_nClusters = (maxLabel < 0) ? std::size_t{0} : static_cast<std::size_t>(maxLabel) + 1;

    m_ctParent = std::move(condensed.parent);
    m_ctChild = std::move(condensed.child);
    m_ctLambda = std::move(condensed.lambdaVal);
    m_ctChildSize = std::move(condensed.childSize);
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
