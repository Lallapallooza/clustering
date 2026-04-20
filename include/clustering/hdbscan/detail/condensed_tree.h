#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

#include "clustering/always_assert.h"
#include "clustering/hdbscan/detail/single_linkage.h"

namespace clustering::hdbscan::detail {

/**
 * @brief Condensed cluster tree stored as parallel arrays, one entry per non-root edge.
 *
 * Each row @c i records a parent-child edge of the condensed hierarchy with the lambda value at
 * which the child separated from the parent. @c childSize encodes whether the child is a leaf
 * point (size 1, the "fall-out-as-noise" exit) or an internal condensed cluster node (size > 1).
 *
 * Internal cluster ids live in the range @c [n, n + numClusters); leaf ids (points) live in
 * @c [0, n). The root condensed cluster gets id @c n and has no incoming edge (so no row).
 *
 * @tparam T Lambda value type (the reciprocal of the MRD weight).
 */
template <class T> struct CondensedTree {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                "CondensedTree<T> requires float or double");
  std::vector<std::int32_t> parent;
  std::vector<std::int32_t> child;
  std::vector<T> lambdaVal;
  std::vector<std::int32_t> childSize;
  /// The total count of condensed cluster nodes (root + internal), equal to @c nClusters in the
  /// Campello walk. Useful for EOM / leaf extraction which reason over the cluster-id space.
  std::int32_t numClusters = 0;
};

/**
 * @brief Compute the number of descendant leaves (points) at a single-linkage-tree node.
 *
 * Leaf nodes are @c [0, n) and have size @c 1 implicitly; internal merge nodes have their size
 * stored in @p slt.size. The helper reads the correct slot either way.
 */
template <class T>
[[nodiscard]] inline std::int32_t sltNodeSize(const SingleLinkageTree<T> &slt, std::size_t n,
                                              std::int32_t nodeId) noexcept {
  if (std::cmp_less(nodeId, n)) {
    return std::int32_t{1};
  }
  const auto row = static_cast<std::size_t>(nodeId) - n;
  return slt.size[row];
}

/**
 * @brief Condense a single-linkage dendrogram by pruning splits that drop below @p minClusterSize.
 *
 * Implements the Campello 2015 condensation: walk the single-linkage tree top-down from the root,
 * and at each merge decide whether either side of the split contributes a "real" cluster. A side
 * is real only if its descendant count is at least @p minClusterSize. The three cases:
 *   - Both sides real: emit a new parent-child edge for each real side, assigning fresh
 *     condensed-cluster ids; each subtree roots its own condensed sub-cluster and recurses.
 *   - Exactly one side real: the real side inherits the current condensed-cluster id (no split);
 *     the small side "falls out as noise" and every one of its descendant points gets a leaf row
 *     at the lambda-at-split, with @c childSize == 1 and a single shared lambda.
 *   - Neither side real: impossible if the parent sub-cluster is real; the algorithm only recurses
 *     into real sub-clusters.
 *
 * Lambda is the inverse of the MRD-distance at which the merge occurred; points and sub-clusters
 * that drop out at larger distances have smaller lambdas. Zero distance is mapped to the
 * floating-point infinity so core-zero merges produce well-defined ordering relative to any
 * positive-distance merge.
 *
 * @param slt             Single-linkage dendrogram built by @ref buildSingleLinkageTree.
 * @param n               Number of input points (also the leaf id upper bound).
 * @param minClusterSize  Minimum descendant count for a sub-cluster to survive condensation.
 * @param out             Destination; parallel arrays are cleared then filled in BFS order.
 */
template <class T>
inline void condenseTree(const SingleLinkageTree<T> &slt, std::size_t n, std::size_t minClusterSize,
                         CondensedTree<T> &out) {
  CLUSTERING_ALWAYS_ASSERT(minClusterSize >= 2);
  out.parent.clear();
  out.child.clear();
  out.lambdaVal.clear();
  out.childSize.clear();

  const std::size_t nEdges = slt.distance.size();
  if (nEdges == 0) {
    out.numClusters = 0;
    return;
  }
  CLUSTERING_ALWAYS_ASSERT(slt.left.size() == nEdges);
  CLUSTERING_ALWAYS_ASSERT(slt.right.size() == nEdges);
  CLUSTERING_ALWAYS_ASSERT(slt.size.size() == nEdges);

  // The root of the single-linkage tree is the final merge: slt node id n + nEdges - 1.
  const auto rootSltId = static_cast<std::int32_t>(n + nEdges - 1);
  const auto kSignedN = static_cast<std::int32_t>(n);
  const std::int32_t rootCondensedId = kSignedN; // first condensed id after leaves.

  // Assign every slt internal node a condensed-cluster id; leaves are left at -1 to mean "not an
  // internal condensed cluster." Sizing is 2n-1 total slt nodes (n leaves + n-1 merges); only the
  // merge range is written.
  std::vector<std::int32_t> condensedIdOf(n + nEdges, std::int32_t{-1});
  condensedIdOf[static_cast<std::size_t>(rootSltId)] = rootCondensedId;
  std::int32_t nextCondensedId = rootCondensedId + 1;

  // BFS over "real" sub-clusters. The frontier stores (sltNodeId, inheritedCondensedId).
  struct Frame {
    std::int32_t sltNode;
    std::int32_t condensedId;
  };
  std::vector<Frame> frontier;
  frontier.push_back(Frame{rootSltId, rootCondensedId});

  // Helper: emit `childSize` leaf rows under `parentCondensedId` for every descendant point of
  // `sltNode`, all at the same `lambda`. This is the "drop the whole subtree as noise" branch of
  // the Campello algorithm. Implemented iteratively through a dfs stack to keep recursion depth
  // bounded by slt height only implicitly; the explicit stack avoids stack overflow at n ~ 1M.
  auto dropSubtreeAsLeaves = [&](std::int32_t sltNode, std::int32_t parentCondensedId, T lambda) {
    std::vector<std::int32_t> stack;
    stack.push_back(sltNode);
    while (!stack.empty()) {
      const std::int32_t cur = stack.back();
      stack.pop_back();
      if (std::cmp_less(cur, n)) {
        out.parent.push_back(parentCondensedId);
        out.child.push_back(cur);
        out.lambdaVal.push_back(lambda);
        out.childSize.push_back(std::int32_t{1});
      } else {
        const auto row = static_cast<std::size_t>(cur) - n;
        stack.push_back(slt.left[row]);
        stack.push_back(slt.right[row]);
      }
    }
  };

  // Map dist -> lambda with a divide-by-zero guard: a zero-distance merge maps to infinity, so
  // those points never drop out (they persist at the tightest possible lambda).
  auto lambdaOf = [](T dist) noexcept -> T {
    if (dist <= T{0}) {
      return std::numeric_limits<T>::infinity();
    }
    return T{1} / dist;
  };

  while (!frontier.empty()) {
    const Frame frame = frontier.back();
    frontier.pop_back();
    const std::int32_t sltNode = frame.sltNode;
    const std::int32_t condensedId = frame.condensedId;

    // Leaf frontier entries (points) are not processed; the only way we push a leaf is via the
    // drop-as-noise path above, which does its own row emission. Guard just in case.
    if (std::cmp_less(sltNode, n)) {
      continue;
    }

    const auto row = static_cast<std::size_t>(sltNode) - n;
    const std::int32_t leftChild = slt.left[row];
    const std::int32_t rightChild = slt.right[row];
    const T splitLambda = lambdaOf(slt.distance[row]);

    const std::int32_t leftSize = sltNodeSize(slt, n, leftChild);
    const std::int32_t rightSize = sltNodeSize(slt, n, rightChild);
    const auto kMin = static_cast<std::int32_t>(minClusterSize);
    const bool leftReal = leftSize >= kMin;
    const bool rightReal = rightSize >= kMin;

    if (leftReal && rightReal) {
      // Split: each side becomes its own new condensed cluster node. Emit two rows and recurse.
      const std::int32_t leftId = nextCondensedId++;
      const std::int32_t rightId = nextCondensedId++;
      out.parent.push_back(condensedId);
      out.child.push_back(leftId);
      out.lambdaVal.push_back(splitLambda);
      out.childSize.push_back(leftSize);
      out.parent.push_back(condensedId);
      out.child.push_back(rightId);
      out.lambdaVal.push_back(splitLambda);
      out.childSize.push_back(rightSize);
      // Record ids for downstream code (e.g. leaf extraction) that walks node identities.
      condensedIdOf[static_cast<std::size_t>(leftChild)] = leftId;
      condensedIdOf[static_cast<std::size_t>(rightChild)] = rightId;
      frontier.push_back(Frame{leftChild, leftId});
      frontier.push_back(Frame{rightChild, rightId});
    } else if (leftReal) {
      // Right side falls out; left inherits the current condensed cluster id.
      dropSubtreeAsLeaves(rightChild, condensedId, splitLambda);
      condensedIdOf[static_cast<std::size_t>(leftChild)] = condensedId;
      frontier.push_back(Frame{leftChild, condensedId});
    } else if (rightReal) {
      dropSubtreeAsLeaves(leftChild, condensedId, splitLambda);
      condensedIdOf[static_cast<std::size_t>(rightChild)] = condensedId;
      frontier.push_back(Frame{rightChild, condensedId});
    } else {
      // Both sides fall out at this split. Each descendant point drops at `splitLambda`.
      dropSubtreeAsLeaves(leftChild, condensedId, splitLambda);
      dropSubtreeAsLeaves(rightChild, condensedId, splitLambda);
    }
  }

  out.numClusters = nextCondensedId - kSignedN;
}

} // namespace clustering::hdbscan::detail
