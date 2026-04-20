#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

#include "clustering/always_assert.h"
#include "clustering/hdbscan/mst_output.h"
#include "clustering/math/dsu.h"

namespace clustering::hdbscan::detail {

/**
 * @brief Single-linkage dendrogram stored as parallel arrays of @c N - 1 merges.
 *
 * Matches the scipy @c linkage layout in spirit: merge row @c i of @c (left, right, distance, size)
 * records that cluster id @c left and cluster id @c right coalesce at @c distance, producing a new
 * cluster of @c size points identified by @c N + i. Leaf ids are @c [0, N); merge ids are
 * @c [N, 2N - 1). Consumers walk rows in the natural order for cluster condensation.
 *
 * @tparam T Element type carrying the merge distance (mutual-reachability).
 */
template <class T> struct SingleLinkageTree {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                "SingleLinkageTree<T> requires float or double");
  std::vector<std::int32_t> left;
  std::vector<std::int32_t> right;
  std::vector<T> distance;
  std::vector<std::int32_t> size;
};

/**
 * @brief Build the single-linkage dendrogram from an MST edge list.
 *
 * Sorts the @c N - 1 MST edges ascending by weight, streams them through a union-find, and emits
 * one merge row per union. The classic Kruskal-style construction: every edge connects two
 * currently distinct components, so each unite becomes a merge and each merge's cluster id is the
 * running merge counter offset by @c N.
 *
 * @pre @p mst.edges.size() == @p n - 1 (the MST contract); the caller supplies @p n explicitly so
 *      the single-linkage builder has no dependence on the backend that produced @p mst.
 *
 * @param mst Frozen MST output; the edge list is read-only, edges may appear in any order.
 * @param n   Number of input points.
 * @param out Destination; each parallel array is resized to @p n - 1 and overwritten.
 */
template <class T>
inline void buildSingleLinkageTree(const MstOutput<T> &mst, std::size_t n,
                                   SingleLinkageTree<T> &out) {
  CLUSTERING_ALWAYS_ASSERT(n >= 2);
  const std::size_t nEdges = n - 1;
  CLUSTERING_ALWAYS_ASSERT(mst.edges.size() == nEdges);

  // Sort edges ascending by weight; a stable sort keeps ties in insertion order for reproducible
  // dendrograms across runs that produce the same MST up to equal-weight permutations.
  std::vector<MstEdge<T>> sorted(mst.edges.begin(), mst.edges.end());
  std::sort(sorted.begin(), sorted.end(),
            [](const MstEdge<T> &a, const MstEdge<T> &b) noexcept { return a.weight < b.weight; });

  out.left.assign(nEdges, std::int32_t{0});
  out.right.assign(nEdges, std::int32_t{0});
  out.distance.assign(nEdges, T{0});
  out.size.assign(nEdges, std::int32_t{0});

  // `clusterOf` maps each point to its current composite cluster id. Initially every point is a
  // singleton whose id is its own row index; after each merge, both member points' entries are
  // rewritten to the new merge id `n + step`.
  std::vector<std::int32_t> clusterOf(n);
  for (std::size_t i = 0; i < n; ++i) {
    clusterOf[i] = static_cast<std::int32_t>(i);
  }

  UnionFind<std::uint32_t> uf(n);

  for (std::size_t step = 0; step < nEdges; ++step) {
    const MstEdge<T> &e = sorted[step];
    const auto u = static_cast<std::uint32_t>(e.u);
    const auto v = static_cast<std::uint32_t>(e.v);
    const std::uint32_t ru = uf.find(u);
    const std::uint32_t rv = uf.find(v);
    CLUSTERING_ALWAYS_ASSERT(ru != rv);
    const std::int32_t leftId = clusterOf[ru];
    const std::int32_t rightId = clusterOf[rv];
    (void)uf.unite(ru, rv);
    const std::uint32_t newRoot = uf.find(ru);
    const auto mergedSize = static_cast<std::int32_t>(uf.componentSize(newRoot));
    const auto mergedId = static_cast<std::int32_t>(n + step);
    // Normalize (left, right) so smaller id appears first; the orientation is irrelevant to the
    // condensed-tree walk but stabilises golden-file comparisons against other implementations.
    if (leftId <= rightId) {
      out.left[step] = leftId;
      out.right[step] = rightId;
    } else {
      out.left[step] = rightId;
      out.right[step] = leftId;
    }
    out.distance[step] = e.weight;
    out.size[step] = mergedSize;
    // Rewrite both former cluster ids' representative slots to the new merged id. `newRoot` is the
    // union-find root of the freshly merged component; `clusterOf[newRoot]` is the slot the next
    // Kruskal step reads when this component participates in another merge.
    clusterOf[newRoot] = mergedId;
  }
}

} // namespace clustering::hdbscan::detail
