#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

#include "clustering/always_assert.h"
#include "clustering/hdbscan/mst_output.h"
#include "clustering/index/kdtree.h"
#include "clustering/math/heap.h"
#include "clustering/math/pairwise.h"
#include "clustering/math/thread.h"
#include "clustering/ndarray.h"

namespace clustering::hdbscan {

/**
 * @brief Memory budget in bytes for the dense mutual-reachability matrix.
 *
 * The Prim backend materialises the full @c (n x n) mutual-reachability matrix in memory. This
 * bound caps the per-fit allocation so a pathological @c n cannot OOM the host. Sized at
 * @c 256 MiB which comfortably accommodates @c n <= 8192 at @c float precision; the auto
 * dispatcher's Prim threshold must satisfy this invariant at its tuned default.
 */
inline constexpr std::size_t kPrimMrdMatrixByteBudget = std::size_t{512} << 20;

/**
 * @brief Dense, exact minimum-spanning-tree backend over the mutual-reachability distance.
 *
 * Given a point matrix @p X and a core-distance parameter @p minSamples, the backend:
 *   1. Computes per-point core distances as the @c minSamples-th nearest-neighbour distance
 *      (self-excluded, squared) via @ref KDTree::knnQuery.
 *   2. Materialises the @c (n x n) mutual-reachability matrix
 *      @c MRD(i, j) = max(sqDist(i, j), coreDist(i), coreDist(j)).
 *   3. Extracts the minimum spanning tree rooted at vertex @c 0 using Prim's algorithm with a
 *      binary min-heap; each popped edge connects an unvisited vertex through the current best
 *      incident weight.
 *
 * The backend satisfies @ref MstBackendStrategy; it carries no persistent state and is cheap to
 * construct.
 *
 * @tparam T Element type of the point cloud. Only @c float is supported.
 */
template <class T> class PrimMstBackend {
  static_assert(std::is_same_v<T, float>,
                "PrimMstBackend<T> supports only float; a double specialization is out of scope.");

public:
  PrimMstBackend() = default;

  /**
   * @brief Build the MRD-weighted minimum spanning tree of @p X.
   *
   * @pre @p minSamples is positive and strictly less than @c X.dim(0).
   * @pre @c X.dim(0) * X.dim(0) * sizeof(T) does not exceed @ref kPrimMrdMatrixByteBudget.
   *      A violation fires @c CLUSTERING_ALWAYS_ASSERT before any allocation.
   *
   * @param X          Contiguous @c (n x d) dataset; caller retains ownership.
   * @param minSamples Neighbour count driving the core-distance definition.
   * @param pool       Worker pool; forwarded to the KDTree build, kNN query, and pairwise kernel.
   * @param out        Destination; @c edges filled with @c n - 1 entries in insertion order and
   *                   @c coreDistances sized to @c n.
   */
  void run(const NDArray<T, 2> &X, std::size_t minSamples, math::Pool pool, MstOutput<T> &out) {
    const std::size_t n = X.dim(0);
    CLUSTERING_ALWAYS_ASSERT(minSamples >= 1);
    CLUSTERING_ALWAYS_ASSERT(minSamples < n);

    // Fire the budget guard BEFORE any allocation so out-of-budget calls die deterministically at
    // the entry rather than after a partial build that leaks progress. The comparison uses the
    // upper bound @c kPrimMrdMatrixByteBudget / sizeof(T) on @c n*n so the multiplication never
    // wraps even at pathological @c n on 64-bit size_t; the @c minSamples precondition above
    // already ruled out @c n == 0, so division is safe.
    constexpr std::size_t kNsqBudget = kPrimMrdMatrixByteBudget / sizeof(T);
    CLUSTERING_ALWAYS_ASSERT(n <= kNsqBudget / n);

    out.edges.clear();
    out.edges.reserve(n - 1);
    out.coreDistances = NDArray<T, 1>(std::array<std::size_t, 1>{n});

    // Phase 1: core distances from the kNN query. The tree is built on @p X directly; the query
    // returns neighbours sorted ascending by squared distance. Core distance is the squared
    // distance to the @c minSamples-th nearest neighbour, consistent with the MRD-matrix
    // population below (squared distances on both sides).
    const KDTree<T> tree(X);
    const auto kSigned = static_cast<std::int32_t>(minSamples);
    auto [knnIdx, knnSqDist] = tree.knnQuery(kSigned, pool);
    (void)knnIdx;
    T *coreDistData = out.coreDistances.data();
    for (std::size_t i = 0; i < n; ++i) {
      coreDistData[i] = knnSqDist(i, minSamples - 1);
    }

    // Phase 2: dense MRD matrix. The public pairwise kernel writes the symmetric @c (n x n)
    // squared-Euclidean matrix; then the elementwise max lifts each cell to the MRD weight.
    NDArray<T, 2> mrd(std::array<std::size_t, 2>{n, n});
    math::pairwiseSqEuclidean(X, X, mrd, pool);

    T *mrdData = mrd.data();
    for (std::size_t i = 0; i < n; ++i) {
      const T coreI = coreDistData[i];
      T *row = mrdData + (i * n);
      for (std::size_t j = 0; j < n; ++j) {
        const T coreJ = coreDistData[j];
        const T sq = row[j];
        T m = sq;
        if (coreI > m) {
          m = coreI;
        }
        if (coreJ > m) {
          m = coreJ;
        }
        row[j] = m;
      }
    }

    // Phase 3: Prim's algorithm starting from vertex 0. The heap holds @c (weight, targetVertex)
    // pairs keyed on the smallest incident edge weight discovered so far. A visited bitmap rules
    // out stale heap entries popped after their target was already admitted through a cheaper
    // edge; we allow those entries into the heap and filter at pop time, which keeps the push
    // path branch-free.
    std::vector<std::uint8_t> visited(n, 0U);
    // Parent of each vertex in the growing spanning tree; only meaningful once the vertex becomes
    // visited. The weight parallel keeps the MRD cost of the edge @c (parent[v], v) so the final
    // emit carries the edge's true MRD weight rather than the heap key (which may be stale when
    // multiple candidate edges pointed at the same vertex before the cheapest was popped).
    std::vector<std::int32_t> parent(n, std::int32_t{0});
    std::vector<T> edgeWeight(n, T{});

    BinaryHeap<T, std::int32_t> heap;

    // Seed the tree at vertex 0 and relax its incident edges. Every other vertex inherits
    // @c parent = 0 and @c edgeWeight = MRD(0, v) as its initial candidate edge. Subsequent Prim
    // iterations only lower these weights.
    visited[0] = 1U;
    {
      const T *row0 = mrdData;
      for (std::size_t j = 1; j < n; ++j) {
        const T w = row0[j];
        edgeWeight[j] = w;
        heap.push(w, static_cast<std::int32_t>(j));
      }
    }

    while (!heap.empty() && out.edges.size() + 1 < n) {
      const auto entry = heap.top();
      const std::int32_t target = entry.second;
      heap.pop();
      const auto tIdx = static_cast<std::size_t>(target);
      if (visited[tIdx] != 0U) {
        continue; // Stale entry superseded by a cheaper relaxation.
      }
      visited[tIdx] = 1U;
      out.edges.push_back(MstEdge<T>{parent[tIdx], target, edgeWeight[tIdx]});

      // Relax every unvisited neighbour through @c target. The MRD matrix is symmetric, so
      // @c row[v] carries @c MRD(target, v).
      const T *row = mrdData + (tIdx * n);
      for (std::size_t v = 0; v < n; ++v) {
        if (visited[v] != 0U) {
          continue;
        }
        const T w = row[v];
        if (w < edgeWeight[v]) {
          parent[v] = target;
          edgeWeight[v] = w;
          heap.push(w, static_cast<std::int32_t>(v));
        }
      }
    }
  }
};

} // namespace clustering::hdbscan
