#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

#include "clustering/hdbscan/detail/condensed_tree.h"

namespace clustering::hdbscan::detail {

/**
 * @brief Leaf-cluster selection on a condensed tree.
 *
 * Every condensed-tree leaf (a cluster node with no cluster-child, only point-children) becomes
 * its own selected cluster. Internal cluster nodes are passed through; points reach the nearest
 * leaf ancestor, or @c -1 when no leaf contains them. Leaves are assigned dense ids in
 * condensed-id order (lowest cluster id first).
 *
 * @param tree Condensed tree produced by @ref condenseTree.
 * @param n    Number of input points; leaf rows in @p tree have @c child in @c [0, n).
 * @param out  Destination vector of length @p n; overwritten with the final per-point labels.
 */
template <class T>
inline void extractLeaf(const CondensedTree<T> &tree, std::size_t n,
                        std::vector<std::int32_t> &out) {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                "extractLeaf<T> requires float or double");
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

  auto clusterIdx = [kSignedN](std::int32_t clusterId) noexcept -> std::size_t {
    return static_cast<std::size_t>(clusterId - kSignedN);
  };

  // A cluster is a "leaf cluster" iff no other cluster has it as a parent. Scan the condensed
  // tree: every row with a cluster-child marks the parent as "not a leaf."
  std::vector<std::uint8_t> hasClusterChild(static_cast<std::size_t>(numClusters), std::uint8_t{0});
  std::vector<std::int32_t> parentOfCluster(static_cast<std::size_t>(numClusters),
                                            std::int32_t{-1});
  for (std::size_t i = 0; i < nRows; ++i) {
    const std::int32_t child = tree.child[i];
    if (child >= kSignedN) {
      const std::int32_t parent = tree.parent[i];
      hasClusterChild[clusterIdx(parent)] = std::uint8_t{1};
      parentOfCluster[clusterIdx(child)] = parent;
    }
  }

  // Assign dense labels to every leaf cluster in condensed-id order.
  std::vector<std::int32_t> denseLabelOf(static_cast<std::size_t>(numClusters), std::int32_t{-1});
  std::int32_t nextLabel = 0;
  for (std::size_t c = 0; c < static_cast<std::size_t>(numClusters); ++c) {
    if (hasClusterChild[c] == 0U) {
      denseLabelOf[c] = nextLabel++;
    }
  }

  // For each leaf-row (point-child), walk up to the nearest leaf-cluster ancestor and copy its
  // dense label. `parentOfCluster` lets us climb in amortised constant time per point.
  for (std::size_t i = 0; i < nRows; ++i) {
    const std::int32_t child = tree.child[i];
    if (child >= kSignedN) {
      continue;
    }
    const auto pointIdx = static_cast<std::size_t>(child);
    std::int32_t walker = tree.parent[i];
    auto label = std::int32_t{-1};
    while (walker >= kSignedN) {
      const auto wIdx = clusterIdx(walker);
      if (hasClusterChild[wIdx] == 0U) {
        label = denseLabelOf[wIdx];
        break;
      }
      walker = parentOfCluster[wIdx];
    }
    out[pointIdx] = label;
  }
}

} // namespace clustering::hdbscan::detail
