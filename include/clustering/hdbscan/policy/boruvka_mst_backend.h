#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>
#include <vector>

#include "clustering/always_assert.h"
#include "clustering/hdbscan/detail/boruvka_traversal.h"
#include "clustering/hdbscan/mst_output.h"
#include "clustering/index/kdtree.h"
#include "clustering/math/dsu.h"
#include "clustering/math/thread.h"
#include "clustering/ndarray.h"

namespace clustering::hdbscan {

/**
 * @brief KDTree-accelerated Borůvka MST backend over mutual-reachability distances.
 *
 * Given a point matrix @p X and core-distance parameter @p minSamples, the backend:
 *   1. Builds a KDTree on @p X.
 *   2. Computes per-point core distances as the @c minSamples-th nearest-neighbour squared
 *      distance (self-excluded) via @ref KDTree::knnQuery.
 *   3. Runs Borůvka rounds: each round invokes @ref detail::nearestOutComponent to find the
 *      minimum-MRD-weight out-of-component edge per component via per-point KDTree traversals
 *      pruned on each subtree's axis-aligned bounding box, then unions those edges into the
 *      spanning tree under the union-find component structure.
 *   4. Sorts the resulting edges ascending by weight for a canonical output ordering.
 *
 * The backend satisfies @ref MstBackendStrategy and holds no persistent state across calls;
 * data-dependent state (the KDTree) is built fresh each @ref run in line with the HDBSCAN
 * pipeline contract that forbids cross-call caching of pointer-keyed data.
 *
 * @tparam T Element type of the point cloud. Only @c float is supported; a @c double
 *         specialization is out of scope in line with @ref HDBSCAN's top-level restriction.
 */
template <class T> class BoruvkaMstBackend {
  static_assert(
      std::is_same_v<T, float>,
      "BoruvkaMstBackend<T> supports only float; a double specialization is out of scope.");

public:
  BoruvkaMstBackend() = default;

  /**
   * @brief Build the MRD-weighted minimum spanning tree of @p X.
   *
   * @pre @p minSamples is positive and strictly less than @c X.dim(0).
   * @pre @c X.dim(0) fits in @c std::int32_t; the HDBSCAN class shell asserts this before a
   *      call reaches the backend.
   *
   * @param X          Contiguous @c (n x d) dataset; caller retains ownership.
   * @param minSamples Neighbour count driving the core-distance definition.
   * @param pool       Worker pool; forwarded to the kNN query and the per-round KDTree
   *                   traversal. A null pool runs single-threaded.
   * @param out        Destination; @c edges filled with @c n - 1 entries in ascending-weight
   *                   order and @c coreDistances sized to @c n.
   */
  void run(const NDArray<T, 2> &X, std::size_t minSamples, math::Pool pool, MstOutput<T> &out) {
    const std::size_t n = X.dim(0);
    CLUSTERING_ALWAYS_ASSERT(minSamples >= 1);
    CLUSTERING_ALWAYS_ASSERT(minSamples < n);

    out.edges.clear();
    out.edges.reserve(n - 1);
    out.coreDistances = NDArray<T, 1>(std::array<std::size_t, 1>{n});

    // Build the tree once; reused for the kNN core-distance query and for every Borůvka
    // round's per-point nearest-out-of-component traversal.
    const KDTree<T> tree(X);

    // Phase 1: core distances via kNN at k = minSamples. knnQuery returns neighbours sorted
    // ascending by squared distance, so the minSamples-th (1-based) neighbour is at column
    // @c minSamples - 1. Squared distances are the in-memory unit across the pipeline; the
    // downstream MRD weight and Prim oracle use the same scale.
    const auto kSigned = static_cast<std::int32_t>(minSamples);
    auto [knnIdx, knnSqDist] = tree.knnQuery(kSigned, pool);
    (void)knnIdx;
    T *coreDistData = out.coreDistances.data();
    for (std::size_t i = 0; i < n; ++i) {
      coreDistData[i] = knnSqDist(i, minSamples - 1);
    }

    // Phase 2: Borůvka rounds. Each round computes one best out-of-component candidate edge
    // per component via the per-point traversal, then unions those edges in ascending order of
    // weight. Processing in ascending weight keeps the MRD-MST total identical to the classical
    // Kruskal/Prim result, modulo equal-weight ties which admit multiple valid MSTs.
    UnionFind<std::uint32_t> uf(n);
    std::vector<std::int32_t> componentOf(n, std::int32_t{0});
    std::vector<detail::ComponentBestEdge<T>> bestPerComponent(n);
    std::vector<std::int32_t> activeRoots;
    activeRoots.reserve(n);

    while (uf.countComponents() > 1) {
      for (std::size_t i = 0; i < n; ++i) {
        componentOf[i] = static_cast<std::int32_t>(uf.find(static_cast<std::uint32_t>(i)));
      }

      detail::nearestOutComponent<T>(tree, std::span<const std::int32_t>(componentOf),
                                     out.coreDistances, pool, bestPerComponent);

      // Collect unique root candidates that produced a finite best. A component without a
      // finite best on a >1-component graph would be a correctness failure -- the fan-out
      // must have reached every component's points. Assert rather than silently skip so the
      // bug surfaces at the failing round.
      activeRoots.clear();
      for (std::size_t i = 0; i < n; ++i) {
        if (std::cmp_equal(componentOf[i], i)) {
          activeRoots.push_back(static_cast<std::int32_t>(i));
        }
      }

      // Apply in ascending-weight order. Tie-break by (u, v) keeps the output deterministic
      // across equal-weight candidates that may arrive from different components.
      std::sort(activeRoots.begin(), activeRoots.end(),
                [&](std::int32_t a, std::int32_t b) noexcept {
                  const auto &ea = bestPerComponent[static_cast<std::size_t>(a)];
                  const auto &eb = bestPerComponent[static_cast<std::size_t>(b)];
                  if (ea.weight != eb.weight) {
                    return ea.weight < eb.weight;
                  }
                  if (ea.u != eb.u) {
                    return ea.u < eb.u;
                  }
                  return ea.v < eb.v;
                });

      bool unitedAny = false;
      for (const std::int32_t c : activeRoots) {
        const auto &edge = bestPerComponent[static_cast<std::size_t>(c)];
        CLUSTERING_ALWAYS_ASSERT(edge.u >= 0 && edge.v >= 0);
        if (uf.unite(static_cast<std::uint32_t>(edge.u), static_cast<std::uint32_t>(edge.v))) {
          out.edges.push_back(MstEdge<T>{edge.u, edge.v, edge.weight});
          unitedAny = true;
        }
      }

      // Safety gate: a round that fails to unite any component would loop forever. This cannot
      // happen on a connected graph when every component has a finite best, but assert rather
      // than infinite-loop if the invariant ever breaks.
      CLUSTERING_ALWAYS_ASSERT(unitedAny);
    }

    CLUSTERING_ALWAYS_ASSERT(out.edges.size() == n - 1);

    // Canonical output ordering: ascending by weight. The single-linkage tree construction
    // downstream consumes edges in this order, matching the Prim backend's implicit Prim-pop
    // ordering (which is weight-monotone outside of tie groups). Sort explicitly here so the
    // output contract is uniform across backends.
    std::sort(out.edges.begin(), out.edges.end(),
              [](const MstEdge<T> &a, const MstEdge<T> &b) noexcept {
                if (a.weight != b.weight) {
                  return a.weight < b.weight;
                }
                if (a.u != b.u) {
                  return a.u < b.u;
                }
                return a.v < b.v;
              });
  }
};

} // namespace clustering::hdbscan
