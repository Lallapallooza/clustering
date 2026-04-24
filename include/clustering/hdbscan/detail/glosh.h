#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

#include "clustering/hdbscan/detail/condensed_tree.h"

namespace clustering::hdbscan::detail {

/**
 * @brief Compute per-point GLOSH outlier scores using the Campello 2015 formula.
 *
 * For each point @c x, let @c C_parent be the condensed-tree cluster containing @c x. Let
 * `lambda_max(C_parent)` be the maximum lambda reached anywhere in @c C_parent's subtree (either
 * the largest point-fall-out lambda or the deepest sub-cluster's birth lambda, whichever is
 * larger). The score is
 *
 * @verbatim
 *   outlier(x) = (lambda_max(C_parent) - lambda(x)) / lambda_max(C_parent)
 * @endverbatim
 *
 * where `lambda(x)` is the lambda at which @c x fell out of its containing cluster. The ratio is
 * bounded in `[0, 1]` by construction. A point that persists all the way to @c lambda_max scores
 * @c 0; a point that dropped out at the boundary where @c lambda_max was reached scores nearly
 * @c 1; a point that falls out at @c lambda = 0 scores exactly @c 1 (modulo the divide-by-zero
 * guard below).
 *
 * The Campello formula uses the **containing cluster's** lambda_max as the normalizer, not the
 * sub-cluster's own death lambda. This is the distinction that keeps scores in `[0, 1]`.
 *
 * @param tree   Condensed tree produced by @ref condenseTree.
 * @param n      Number of input points; leaf rows in @p tree have @c child in `[0, n)`.
 * @param labels Per-point labels (result of EOM or leaf extraction). Noise points (`-1`) receive
 *               an outlier score of @c 0 -- scores are only meaningful for selected clusters.
 * @param out    Destination vector of length @p n; overwritten with the per-point scores in
 *               `[0, 1]`.
 */
template <class T>
inline void computeGlosh(const CondensedTree<T> &tree, std::size_t n,
                         const std::vector<std::int32_t> & /*labels*/, std::vector<T> &out) {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                "computeGlosh<T> requires float or double");
  out.assign(n, T{0});

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

  // Step 1: compute per-cluster `lambdaMaxInSubtree[c]` = the maximum lambda reached anywhere in
  // c's subtree, inclusive of point fall-outs and child cluster births. We walk rows once to seed
  // each cluster's own local max (the largest lambda value on any direct edge out of c), then
  // propagate maxes up from children via a reverse sweep over cluster ids.
  std::vector<T> lambdaMaxInSubtree(static_cast<std::size_t>(numClusters), T{0});
  std::vector<std::int32_t> parentOfCluster(static_cast<std::size_t>(numClusters),
                                            std::int32_t{-1});
  for (std::size_t i = 0; i < nRows; ++i) {
    const std::int32_t parent = tree.parent[i];
    const std::int32_t child = tree.child[i];
    const T lambda = tree.lambdaVal[i];
    const std::size_t pIdx = clusterIdx(parent);
    if (lambda > lambdaMaxInSubtree[pIdx]) {
      lambdaMaxInSubtree[pIdx] = lambda;
    }
    if (child >= kSignedN) {
      parentOfCluster[clusterIdx(child)] = parent;
    }
  }
  // Propagate: each cluster's subtree max is at least each child cluster's subtree max. Reverse
  // cluster-id order ensures a child is finalised before its parent reads from it.
  for (std::int32_t cIdx = numClusters - 1; cIdx > 0; --cIdx) {
    const std::int32_t parent = parentOfCluster[static_cast<std::size_t>(cIdx)];
    if (parent < kSignedN) {
      continue;
    }
    const std::size_t pIdx = clusterIdx(parent);
    const T childSubtreeMax = lambdaMaxInSubtree[static_cast<std::size_t>(cIdx)];
    if (childSubtreeMax > lambdaMaxInSubtree[pIdx]) {
      lambdaMaxInSubtree[pIdx] = childSubtreeMax;
    }
  }

  // Step 2: per-point score. For each leaf-row (the point fell out of its containing cluster at
  // `lambdaVal`), the parent of the row is the containing cluster `C_parent`. The score uses
  // `lambdaMaxInSubtree[C_parent]` as the normaliser.
  for (std::size_t i = 0; i < nRows; ++i) {
    const std::int32_t child = tree.child[i];
    if (child >= kSignedN) {
      continue;
    }
    const auto pointIdx = static_cast<std::size_t>(child);
    const std::int32_t parent = tree.parent[i];
    const std::size_t pIdx = clusterIdx(parent);
    const T lambdaMax = lambdaMaxInSubtree[pIdx];
    const T lambdaX = tree.lambdaVal[i];
    T score;
    if (lambdaMax == std::numeric_limits<T>::infinity()) {
      // Degenerate: a zero-distance merge drove lambdaMax to infinity. Every finite point-lambda
      // produces a score of 1; a point also at infinity produces 0.
      score = (lambdaX == std::numeric_limits<T>::infinity()) ? T{0} : T{1};
    } else if (lambdaMax <= T{0}) {
      // Degenerate: no positive lambda observed in this cluster. Scoring would divide by zero; the
      // well-defined behaviour is 0 (no evidence of outlierness).
      score = T{0};
    } else {
      score = (lambdaMax - lambdaX) / lambdaMax;
      // Clamp defensively against floating-point rounding that lands just outside [0, 1].
      if (score < T{0}) {
        score = T{0};
      } else if (score > T{1}) {
        score = T{1};
      }
    }
    out[pointIdx] = score;
  }
}

} // namespace clustering::hdbscan::detail
