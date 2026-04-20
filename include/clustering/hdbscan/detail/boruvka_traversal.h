#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <type_traits>
#include <vector>

#include "clustering/always_assert.h"
#include "clustering/index/kdtree.h"
#include "clustering/math/detail/avx2_helpers.h"
#include "clustering/math/thread.h"
#include "clustering/ndarray.h"

namespace clustering::hdbscan::detail {

/**
 * @brief Squared gap distance between a point and an axis-aligned box.
 *
 * Per dimension, either the point is inside the extent (zero gap) or the gap is the signed
 * distance to the nearer face. Strict lower bound on the squared Euclidean distance from the
 * point to any point within the box.
 *
 * @tparam T Scalar element type.
 *
 * @param point  Pointer to the @c d-coordinate query point.
 * @param boxMin Min-coords of the box; length @c d.
 * @param boxMax Max-coords of the box; length @c d.
 * @return Squared point-to-AABB gap distance.
 */
template <class T>
[[nodiscard]] inline T pointAabbGapSq(const T *point, std::span<const T> boxMin,
                                      std::span<const T> boxMax) noexcept {
  const std::size_t d = boxMin.size();
  T sum = T{0};
  for (std::size_t j = 0; j < d; ++j) {
    T gap = T{0};
    if (point[j] < boxMin[j]) {
      gap = boxMin[j] - point[j];
    } else if (point[j] > boxMax[j]) {
      gap = point[j] - boxMax[j];
    }
    sum += gap * gap;
  }
  return sum;
}

/**
 * @brief Per-component candidate edge with the lowest-weight out-of-component candidate so far.
 *
 * Indexed by the component's representative root as returned by the caller-owned union-find.
 * Weight @c +inf (via @c std::numeric_limits<T>::max) marks "no candidate yet"; both endpoints
 * are @c -1 on an empty slot. Updated in place by @ref nearestOutComponent.
 */
template <class T> struct ComponentBestEdge {
  T weight = std::numeric_limits<T>::max();
  std::int32_t u = -1;
  std::int32_t v = -1;
};

namespace internal {

/**
 * @brief Per-traversal shared state for one Borůvka round.
 *
 * The traversal reads @c componentOf and @c coreDist to classify points and weigh edges, and
 * writes into @c bestW / @c bestU / @c bestV. Each worker owns a private slot so the writes are
 * lock-free; the driver merges per-worker slots after the round completes.
 *
 * @tparam T Scalar element type.
 */
template <class T> struct TraversalCtx {
  const KDTree<T> *tree;
  std::span<const std::size_t> perm;    ///< Reordered-slot -> original-index.
  std::span<const T> reorderedPts;      ///< Flat @c (n*d) reordered-points buffer.
  std::size_t d;                        ///< Dimension of the point cloud.
  std::span<const std::int32_t> compOf; ///< Per-point component root.
  const T *coreDist;                    ///< Length-n core-distance (squared) buffer.
  T *bestW;                             ///< Length-n worker-local component bests.
  std::int32_t *bestU;                  ///< Length-n worker-local endpoint u.
  std::int32_t *bestV;                  ///< Length-n worker-local endpoint v.
};

/**
 * @brief Compute the MRD weight for a point pair and update the query point's best edge.
 *
 * Inlined into the leaf walk so the hot loop over candidate @c j contains exactly the distance
 * kernel plus the three-way max and the per-component update -- no helper call frame. The
 * writes target @c ctx.bestW[compI] directly; concurrent updates to the same @c compI from
 * different workers are ruled out by the per-worker slot contract.
 */
template <class T>
inline void updateBestForPair(TraversalCtx<T> &ctx, std::int32_t origI, std::int32_t compI, T coreI,
                              const T *rowI, std::int32_t origJ, std::int32_t compJ,
                              const T *rowJ) noexcept {
  if (compI == compJ) {
    return;
  }
  const T sq = math::detail::sqEuclideanRowPtr(rowI, rowJ, ctx.d);
  T mrd = sq;
  if (coreI > mrd) {
    mrd = coreI;
  }
  const T coreJ = ctx.coreDist[origJ];
  if (coreJ > mrd) {
    mrd = coreJ;
  }
  T &bestForComp = ctx.bestW[compI];
  if (mrd < bestForComp) {
    bestForComp = mrd;
    ctx.bestU[compI] = origI;
    ctx.bestV[compI] = origJ;
  }
}

/**
 * @brief Traverse the KDTree with a single query point, seeking the nearest out-of-component j.
 *
 * Single-tree depth-first walk with an AABB gap prune: subtrees whose gap distance from @p rowI
 * exceeds the running per-component best for @p compI are skipped. Leaf nodes enumerate their
 * stored points, comparing distances pairwise. Internal-node pivots are processed at the node
 * as singleton updates so every point in the tree is reachable.
 *
 * The bound that prunes subtrees is @c ctx.bestW[compI]; it tightens monotonically as candidates
 * improve. No separate per-subtree cache is required.
 */
template <class T>
void singlePointScan(TraversalCtx<T> &ctx, std::int32_t origI, std::int32_t compI, T coreI,
                     const T *rowI, const KDTreeNode *root) noexcept {
  if (root == nullptr) {
    return;
  }

  // Caller-owned traversal stack pattern matches the KDTree's radius-query driver. Reserve a
  // generous default so the branch-heavy pivots don't realloc mid-walk.
  std::vector<const KDTreeNode *> stack;
  stack.reserve(std::size_t{64});
  stack.push_back(root);

  const T *reordered = ctx.reorderedPts.data();

  while (!stack.empty()) {
    const KDTreeNode *node = stack.back();
    stack.pop_back();
    if (node == nullptr) {
      continue;
    }

    const T bound = ctx.bestW[compI];
    auto [nmin, nmax] = ctx.tree->nodeBounds(node);
    const T gapSq = pointAabbGapSq<T>(rowI, nmin, nmax);
    if (gapSq >= bound) {
      continue;
    }

    const bool isLeaf = (node->m_left == nullptr && node->m_right == nullptr);
    if (isLeaf) {
      const std::size_t base = node->m_index;
      const std::size_t count = node->m_dim;
      const T *leafPts = reordered + (base * ctx.d);
      for (std::size_t p = 0; p < count; ++p) {
        const auto origJ = static_cast<std::int32_t>(ctx.perm[base + p]);
        const std::int32_t compJ = ctx.compOf[origJ];
        const T *rowJ = leafPts + (p * ctx.d);
        updateBestForPair<T>(ctx, origI, compI, coreI, rowI, origJ, compJ, rowJ);
      }
      continue;
    }

    // Internal node: score the pivot first, then descend. The pivot row lives at the node's
    // reordered slot; its original index comes through the permutation.
    const std::size_t pivotSlot = node->m_index;
    const auto origJ = static_cast<std::int32_t>(ctx.perm[pivotSlot]);
    const std::int32_t compJ = ctx.compOf[origJ];
    const T *pivotRow = reordered + (pivotSlot * ctx.d);
    updateBestForPair<T>(ctx, origI, compI, coreI, rowI, origJ, compJ, pivotRow);

    // Descend near-first, far-second against the split axis. Push far before near so the stack
    // pops near first. Distance along the split axis is a cheaper pre-prune than the full gap.
    const std::size_t splitDim = node->m_dim;
    const T diff = rowI[splitDim] - pivotRow[splitDim];
    if (diff < T{0}) {
      if (node->m_right != nullptr) {
        stack.push_back(node->m_right);
      }
      if (node->m_left != nullptr) {
        stack.push_back(node->m_left);
      }
    } else {
      if (node->m_left != nullptr) {
        stack.push_back(node->m_left);
      }
      if (node->m_right != nullptr) {
        stack.push_back(node->m_right);
      }
    }
  }
}

} // namespace internal

/**
 * @brief Find one minimum-weight out-of-component edge per component for one Borůvka round.
 *
 * For every active component (root in @p componentOf), the best candidate out-edge is the
 * minimum-MRD-weight edge connecting a member of the component to a non-member. Every original
 * point is processed as an independent single-tree KDTree walk: for point @c i, the walk visits
 * all out-of-component candidates subject to AABB gap pruning against the running per-component
 * best. The outer loop over points is parallelised; worker-local @c bestW arrays are merged
 * into the output at the end.
 *
 * @tparam T Scalar element type of the point cloud.
 *
 * @param tree        KDTree built on the caller's point cloud; borrowed.
 * @param componentOf Per-point component root as returned by @ref UnionFind::find.
 * @param coreDist    Squared core distances; length @c n.
 * @param pool        Worker pool; @c nullptr runs single-threaded.
 * @param bestOut     One entry per component root; on return, @c bestOut[c] carries the
 *                    minimum-MRD-weight edge leaving component @c c, or
 *                    @c ComponentBestEdge{+inf, -1, -1} when no candidate exists.
 */
template <class T>
void nearestOutComponent(const KDTree<T> &tree, std::span<const std::int32_t> componentOf,
                         const NDArray<T, 1> &coreDist, math::Pool pool,
                         std::vector<ComponentBestEdge<T>> &bestOut) {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                "nearestOutComponent requires T to be float or double");

  const std::size_t n = componentOf.size();
  CLUSTERING_ALWAYS_ASSERT(bestOut.size() == n);
  CLUSTERING_ALWAYS_ASSERT(coreDist.dim(0) == n);

  const KDTreeNode *root = tree.root();
  if (root == nullptr) {
    return;
  }

  const std::size_t dim = tree.dim();
  std::span<const std::size_t> perm = tree.indexPermutation();
  std::span<const T> reordered = tree.reorderedPoints();

  const std::size_t nWorkers = (pool.pool != nullptr) ? pool.workerCount() : std::size_t{1};

  // Per-worker best arrays. Each worker writes to its own slot per component; the merge at the
  // end composes them into @p bestOut. Memory scales linearly in @c nWorkers * n; acceptable at
  // the workloads exercised by the dispatcher which gates large @c n to the Prim backend.
  std::vector<std::vector<T>> workerW(nWorkers, std::vector<T>(n, std::numeric_limits<T>::max()));
  std::vector<std::vector<std::int32_t>> workerU(nWorkers,
                                                 std::vector<std::int32_t>(n, std::int32_t{-1}));
  std::vector<std::vector<std::int32_t>> workerV(nWorkers,
                                                 std::vector<std::int32_t>(n, std::int32_t{-1}));

  auto runChunk = [&](std::size_t lo, std::size_t hi, std::size_t workerSlot) noexcept {
    internal::TraversalCtx<T> ctx{
        .tree = &tree,
        .perm = perm,
        .reorderedPts = reordered,
        .d = dim,
        .compOf = componentOf,
        .coreDist = coreDist.data(),
        .bestW = workerW[workerSlot].data(),
        .bestU = workerU[workerSlot].data(),
        .bestV = workerV[workerSlot].data(),
    };
    const T *reorderedBase = reordered.data();
    for (std::size_t slot = lo; slot < hi; ++slot) {
      const auto origI = static_cast<std::int32_t>(perm[slot]);
      const std::int32_t compI = componentOf[origI];
      const T coreI = coreDist.data()[origI];
      const T *rowI = reorderedBase + (slot * dim);
      internal::singlePointScan<T>(ctx, origI, compI, coreI, rowI, root);
    }
  };

  // Outer fan-out over original points in reordered order. Chunking by slot lets each worker
  // pick up a dense @c (chunk * d) region of the reordered buffer, keeping per-worker memory
  // access largely contiguous in the point cloud. @ref math::Pool::shouldParallelize decides
  // whether the fan-out amortises task dispatch given the per-slot arithmetic cost.
  //
  // The per-slot cost is roughly O(log(n) * d + k * d) where k is the count of surviving
  // out-of-component leaf candidates -- concretely ~16 * d comparisons per leaf hit plus a
  // handful of AABB gaps. The parallelism gate uses @c 64 as the per-slot minimum chunk so
  // very small inputs stay serial.
  const bool useParallel =
      (pool.pool != nullptr) && (nWorkers > std::size_t{1}) && (n >= (nWorkers * std::size_t{64}));
  if (useParallel) {
    pool.pool
        ->submit_blocks(
            std::size_t{0}, n,
            [&](std::size_t lo, std::size_t hi) {
              const std::size_t widx = math::Pool::workerIndex();
              const std::size_t slot = widx < nWorkers ? widx : nWorkers - 1;
              runChunk(lo, hi, slot);
            },
            nWorkers)
        .wait();
  } else {
    runChunk(0, n, /*workerSlot=*/0);
  }

  // Merge per-worker bests into the output. @p bestOut starts at @c {+inf, -1, -1}; the merge
  // compares each worker's candidate against the running best and takes the lower-weight one.
  // Ties are resolved by first-writer (stable across worker indices) so the output is
  // deterministic at a fixed pool size.
  for (std::size_t c = 0; c < n; ++c) {
    bestOut[c] = ComponentBestEdge<T>{std::numeric_limits<T>::max(), -1, -1};
  }
  for (std::size_t w = 0; w < nWorkers; ++w) {
    const T *wW = workerW[w].data();
    const std::int32_t *wU = workerU[w].data();
    const std::int32_t *wV = workerV[w].data();
    for (std::size_t c = 0; c < n; ++c) {
      if (wW[c] < bestOut[c].weight) {
        bestOut[c].weight = wW[c];
        bestOut[c].u = wU[c];
        bestOut[c].v = wV[c];
      }
    }
  }
}

} // namespace clustering::hdbscan::detail
