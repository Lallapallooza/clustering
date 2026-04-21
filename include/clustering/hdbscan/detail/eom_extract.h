#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

#include "clustering/hdbscan/detail/condensed_tree.h"

namespace clustering::hdbscan::detail {

/**
 * @brief Excess-of-mass cluster selection on a condensed tree.
 *
 * Implements the Campello 2015 EOM dynamic program:
 *   1. Compute per-cluster stability @c S(c) = sum over member points of
 *      @c (lambda_drop - lambda_birth), where @c lambda_birth is the lambda at which the cluster
 *      split off from its parent and @c lambda_drop is the per-point lambda at which the point
 *      fell out (as a leaf-row in the condensed tree).
 *   2. Sweep condensed-cluster ids in reverse (children first, then parents), and for each node
 *      choose between (a) keeping the node as a selected cluster, claiming stability @c S(c), or
 *      (b) replacing it with the sum of the best selections across its children. The parent
 *      selects @c (b) only when it strictly improves on @c S(c); ties go to the parent (shallower)
 *      so the realised clusters are as shallow as possible without losing stability.
 *   3. Propagate the selection: any node reached while walking the parent chain of a "chosen"
 *      node is marked as ancestor-of-chosen and is itself never chosen.
 *
 * Emits a dense per-point label array: points reach the chosen ancestor of their containing
 * condensed cluster, or @c -1 if no ancestor was chosen (the root-only case). The label array is
 * remapped to the canonical @c [0, numChosen) order at the end, with the first chosen cluster
 * visited in condensed-id order becoming label @c 0.
 *
 * @param tree Condensed tree produced by @ref condenseTree.
 * @param n    Number of input points; leaf rows in @p tree have @c child in @c [0, n).
 * @param out  Destination vector of length @p n; overwritten with the final per-point labels.
 */
template <class T>
inline void extractEom(const CondensedTree<T> &tree, std::size_t n,
                       std::vector<std::int32_t> &out) {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                "extractEom<T> requires float or double");
  out.assign(n, std::int32_t{-1});

  const std::size_t nRows = tree.parent.size();
  if (nRows == 0) {
    return;
  }
  const auto numClusters = tree.numClusters;
  if (numClusters <= 0) {
    return;
  }
  const auto kSignedN = static_cast<std::int32_t>(n);

  // Per-cluster birth lambda (the lambda at which the cluster split from its parent). The root
  // cluster was never split off; conventionally its birth lambda is 0 so its stability integrates
  // from lambda = 0.
  std::vector<T> birth(static_cast<std::size_t>(numClusters), T{0});
  // Per-cluster children (cluster ids only, ignoring leaf-point rows).
  std::vector<std::vector<std::int32_t>> children(static_cast<std::size_t>(numClusters));
  // Per-cluster stability accumulator.
  std::vector<T> stability(static_cast<std::size_t>(numClusters), T{0});

  auto clusterIdx = [kSignedN](std::int32_t clusterId) noexcept -> std::size_t {
    return static_cast<std::size_t>(clusterId - kSignedN);
  };

  // Scan rows to populate `birth` (whenever a child is itself a cluster id) and `children`.
  for (std::size_t i = 0; i < nRows; ++i) {
    const std::int32_t child = tree.child[i];
    if (child >= kSignedN) {
      const std::size_t cIdx = clusterIdx(child);
      birth[cIdx] = tree.lambdaVal[i];
      children[clusterIdx(tree.parent[i])].push_back(child);
    }
  }

  // Accumulate stability row-by-row. For leaf rows (child is a point), the contribution is the
  // fall-out lambda minus the containing cluster's birth lambda. For internal-cluster rows, the
  // contribution is the size of that sub-cluster times (sub-cluster birth lambda - parent birth
  // lambda), which captures the points that passed through the parent up to the split.
  for (std::size_t i = 0; i < nRows; ++i) {
    const std::int32_t parent = tree.parent[i];
    const std::int32_t child = tree.child[i];
    const T lambda = tree.lambdaVal[i];
    const std::size_t pIdx = clusterIdx(parent);
    if (child < kSignedN) {
      stability[pIdx] += (lambda - birth[pIdx]);
    } else {
      const auto sz = tree.childSize[i];
      stability[pIdx] += static_cast<T>(sz) * (lambda - birth[pIdx]);
    }
  }

  // DP: a cluster is "chosen" if its own stability is at least the sum of its children's chosen
  // stabilities. Processing in reverse cluster-id order guarantees children finish before parents.
  // The root cluster (cIdx == 0) is never chosen: choosing the root collapses every point into a
  // single cluster labelled 0, which the Campello 2015 FORC algorithm explicitly forbids. Its
  // subtree stability is still propagated so the walk below routes around it.
  std::vector<T> subtreeStab(static_cast<std::size_t>(numClusters), T{0});
  std::vector<std::uint8_t> chosen(static_cast<std::size_t>(numClusters), 0U);
  for (std::int32_t cIdx = numClusters - 1; cIdx >= 0; --cIdx) {
    T sumChildren = T{0};
    for (const std::int32_t childCluster : children[static_cast<std::size_t>(cIdx)]) {
      sumChildren += subtreeStab[clusterIdx(childCluster)];
    }
    const T own = stability[static_cast<std::size_t>(cIdx)];
    if (cIdx != 0 && own >= sumChildren) {
      chosen[static_cast<std::size_t>(cIdx)] = 1U;
      subtreeStab[static_cast<std::size_t>(cIdx)] = own;
    } else {
      subtreeStab[static_cast<std::size_t>(cIdx)] = sumChildren;
    }
  }

  // Propagate "descendant of chosen" down the tree: if any ancestor is chosen, no descendant can
  // also be chosen (only one chosen cluster along any root-to-leaf path is kept). Walk top-down.
  // `ancestorChosen[c]` becomes 1 once any ancestor is chosen; then `c` itself is un-chosen.
  std::vector<std::uint8_t> ancestorChosen(static_cast<std::size_t>(numClusters), 0U);
  // Roots of the condensed tree: cluster ids whose parent in `children` is nobody. Since there is
  // exactly one root (the condensed id kSignedN, i.e. index 0) we start the walk there.
  std::vector<std::int32_t> dfs;
  dfs.push_back(std::int32_t{0});
  while (!dfs.empty()) {
    const std::int32_t cIdx = dfs.back();
    dfs.pop_back();
    const auto c = static_cast<std::size_t>(cIdx);
    if (ancestorChosen[c] != 0U) {
      chosen[c] = 0U;
    }
    const std::uint8_t markChildren =
        (chosen[c] != 0U || ancestorChosen[c] != 0U) ? std::uint8_t{1} : std::uint8_t{0};
    for (const std::int32_t childCluster : children[c]) {
      const std::size_t childIdx = clusterIdx(childCluster);
      if (markChildren != 0U) {
        ancestorChosen[childIdx] = 1U;
      }
      dfs.push_back(static_cast<std::int32_t>(childIdx));
    }
  }

  // Remap chosen cluster indices to canonical [0, numChosen) labels in condensed-id order.
  std::vector<std::int32_t> denseLabelOf(static_cast<std::size_t>(numClusters), std::int32_t{-1});
  std::int32_t nextLabel = 0;
  for (std::size_t c = 0; c < static_cast<std::size_t>(numClusters); ++c) {
    if (chosen[c] != 0U) {
      denseLabelOf[c] = nextLabel++;
    }
  }

  // For each point, find the condensed cluster containing it (the cluster whose leaf row points at
  // the point), then walk up the parent chain until hitting a chosen ancestor. No chosen ancestor
  // means noise. The walk happens in amortised constant time per point because the condensed tree
  // is shallow by construction (N pre-condensation levels collapse to O(log N) post-condensation).
  std::vector<std::int32_t> parentOfCluster(static_cast<std::size_t>(numClusters),
                                            std::int32_t{-1});
  for (std::size_t i = 0; i < nRows; ++i) {
    const std::int32_t child = tree.child[i];
    if (child >= kSignedN) {
      parentOfCluster[clusterIdx(child)] = tree.parent[i];
    }
  }

  for (std::size_t i = 0; i < nRows; ++i) {
    const std::int32_t child = tree.child[i];
    if (child >= kSignedN) {
      continue;
    }
    const auto pointIdx = static_cast<std::size_t>(child);
    // The enclosing cluster is `tree.parent[i]`. Climb until a chosen ancestor is found.
    std::int32_t walker = tree.parent[i];
    auto label = std::int32_t{-1};
    while (walker >= kSignedN) {
      const auto wIdx = clusterIdx(walker);
      if (chosen[wIdx] != 0U) {
        label = denseLabelOf[wIdx];
        break;
      }
      walker = parentOfCluster[wIdx];
    }
    out[pointIdx] = label;
  }
}

} // namespace clustering::hdbscan::detail
