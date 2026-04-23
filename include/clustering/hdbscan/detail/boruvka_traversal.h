#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <type_traits>
#include <vector>

#include "clustering/always_assert.h"
#include "clustering/index/kdtree.h"
#include "clustering/math/aabb.h"
#include "clustering/math/detail/avx2_helpers.h"
#include "clustering/math/thread.h"
#include "clustering/ndarray.h"

namespace clustering::hdbscan::detail {

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
 * @brief Per-traversal shared state for one Boruvka round.
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
  /// Per-node "single component" id: @c >=0 if every point under the node belongs to one
  /// component, @c -1 if the subtree spans multiple. Indexed by @c KDTreeNode::m_id. Recomputed
  /// once per Boruvka round; lets the walker skip subtrees that cannot supply an out-of-component
  /// candidate for the current query. Becomes very effective in late rounds when components
  /// have grown to enclose most points.
  const std::int32_t *nodeSingleComp;
  T *bestW;            ///< Length-n worker-local component bests.
  std::int32_t *bestU; ///< Length-n worker-local endpoint u.
  std::int32_t *bestV; ///< Length-n worker-local endpoint v.
};

/**
 * @brief Recursive post-order walk that fills @p nodeSingleComp.
 *
 * Returns the single component id when every point under @p node belongs to one component, or
 * @c -1 when the subtree is mixed. The same value lands in @p nodeSingleComp[node->m_id] for
 * the traversal to consume.
 */
template <class T>
std::int32_t computeNodeSingleComp(const KDTreeNode *node, std::span<const std::size_t> perm,
                                   std::span<const std::int32_t> compOf,
                                   std::vector<std::int32_t> &nodeSingleComp) noexcept {
  if (node == nullptr) {
    return std::int32_t{-1};
  }
  const auto nodeId = static_cast<std::size_t>(node->m_id);

  const bool isLeaf = (node->m_left == nullptr && node->m_right == nullptr);
  if (isLeaf) {
    const std::size_t base = node->m_index;
    const std::size_t count = node->m_dim;
    if (count == 0) {
      nodeSingleComp[nodeId] = std::int32_t{-1};
      return std::int32_t{-1};
    }
    const std::int32_t first = compOf[perm[base]];
    for (std::size_t p = 1; p < count; ++p) {
      if (compOf[perm[base + p]] != first) {
        nodeSingleComp[nodeId] = std::int32_t{-1};
        return std::int32_t{-1};
      }
    }
    nodeSingleComp[nodeId] = first;
    return first;
  }

  const std::int32_t pivotComp = compOf[perm[node->m_index]];
  const std::int32_t leftComp =
      computeNodeSingleComp<T>(node->m_left, perm, compOf, nodeSingleComp);
  const std::int32_t rightComp =
      computeNodeSingleComp<T>(node->m_right, perm, compOf, nodeSingleComp);

  std::int32_t result = pivotComp;
  if (node->m_left != nullptr && leftComp != pivotComp) {
    result = std::int32_t{-1};
  }
  if (result != std::int32_t{-1} && node->m_right != nullptr && rightComp != pivotComp) {
    result = std::int32_t{-1};
  }
  nodeSingleComp[nodeId] = result;
  return result;
}

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
                     const T *rowI, const KDTreeNode *root,
                     std::vector<const KDTreeNode *> &stack) noexcept {
  if (root == nullptr) {
    return;
  }

  // Caller-owned scratch stack: reused across every @c singlePointScan invocation in this
  // worker's @ref nearestOutComponent slice. Hoisting the allocation out of the per-point loop
  // turns the round's ~n vector constructions into one per worker.
  stack.clear();
  stack.push_back(root);

  const T *reordered = ctx.reorderedPts.data();

  while (!stack.empty()) {
    const KDTreeNode *node = stack.back();
    stack.pop_back();
    if (node == nullptr) {
      continue;
    }

    // Subtree-level component prune: when every point under this node belongs to the query's
    // own component, the subtree cannot supply an out-of-component candidate for this query.
    // Skip the descent entirely. This dominates the late Boruvka rounds where 95%+ of subtrees
    // are self-component for any given query.
    const std::int32_t nodeComp = ctx.nodeSingleComp[node->m_id];
    if (nodeComp == compI) {
      continue;
    }

    const T bound = ctx.bestW[compI];
    auto [nmin, nmax] = ctx.tree->nodeBounds(node);
    const T gapSq = math::pointAabbGapSq<T>(rowI, nmin, nmax);
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
 * @brief Find one minimum-weight out-of-component edge per component for one Boruvka round.
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
                         const NDArray<T, 1> &coreDist, const NDArray<std::int32_t, 2> &knnIdx,
                         const NDArray<T, 2> &knnSqDist, math::Pool pool,
                         std::vector<ComponentBestEdge<T>> &bestOut,
                         bool allSingletonComponents = false) {
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

  // Populate the per-node single-component cache via one post-order walk. Cost is O(nodeCount)
  // and lives outside the worker fan-out so the workers see a read-only buffer.
  std::vector<std::int32_t> nodeSingleComp(tree.nodeCount(), std::int32_t{-1});
  internal::computeNodeSingleComp<T>(root, perm, componentOf, nodeSingleComp);

  const std::size_t nWorkers = (pool.pool != nullptr) ? pool.workerCount() : std::size_t{1};

  // Per-worker best arrays. Each worker writes to its own slot per component; the merge at the
  // end composes them into @p bestOut. Memory scales linearly in @c nWorkers * n; acceptable at
  // the workloads exercised by the dispatcher which gates large @c n to the Prim backend.
  std::vector<std::vector<T>> workerW(nWorkers, std::vector<T>(n, std::numeric_limits<T>::max()));
  std::vector<std::vector<std::int32_t>> workerU(nWorkers,
                                                 std::vector<std::int32_t>(n, std::int32_t{-1}));
  std::vector<std::vector<std::int32_t>> workerV(nWorkers,
                                                 std::vector<std::int32_t>(n, std::int32_t{-1}));

  // kNN-derived initial bound per component. Each point's kNN list (already in hand from the
  // core-distance query) is the cheapest catalog of nearby candidates. For each point we walk
  // its kNN entries, find the first cross-component neighbour, and use its MRD weight to seed
  // the per-component bound. The KDTree walker below reads @c bestW as its AABB-prune
  // threshold; a tighter starting threshold prunes far more subtrees before any distance work
  // happens. Strictly correct: the seed is itself a real MRD candidate edge, so adopting it as
  // both the bound and the (provisional) best is sound. Workers tighten further per slice.
  const std::size_t kNN = knnIdx.dim(1);
  const T *coreData = coreDist.data();
  std::vector<T> seedW(n, std::numeric_limits<T>::max());
  std::vector<std::int32_t> seedU(n, std::int32_t{-1});
  std::vector<std::int32_t> seedV(n, std::int32_t{-1});
  for (std::size_t i = 0; i < n; ++i) {
    const std::int32_t compI = componentOf[i];
    const T coreI = coreData[i];
    const auto compISize = static_cast<std::size_t>(compI);
    // Walk every kNN entry rather than breaking on the first cross-component match: the MRD
    // lift @c max(coreI, coreJ, sqDist) can make a slightly farther candidate the lower-MRD
    // one when the closer candidate has a larger core distance. A tighter seed = more AABB
    // pruning in the tree walk below.
    for (std::size_t s = 0; s < kNN; ++s) {
      const std::int32_t j = knnIdx(i, s);
      if (j < 0 || std::cmp_equal(j, i)) {
        continue;
      }
      const auto jSize = static_cast<std::size_t>(j);
      const std::int32_t compJ = componentOf[jSize];
      if (compJ == compI) {
        continue;
      }
      T w = knnSqDist(i, s);
      if (coreI > w) {
        w = coreI;
      }
      const T coreJ = coreData[jSize];
      if (coreJ > w) {
        w = coreJ;
      }
      if (w < seedW[compISize]) {
        seedW[compISize] = w;
        seedU[compISize] = static_cast<std::int32_t>(i);
        seedV[compISize] = j;
      }
    }
  }
  // Replicate the seed to every worker's local arrays so each worker's @c singlePointScan
  // walks under the seeded bound from the very first node. The merge step at the end picks
  // the best across workers; identical seeds across slots collapse harmlessly.
  for (std::size_t w = 0; w < nWorkers; ++w) {
    std::copy(seedW.begin(), seedW.end(), workerW[w].begin());
    std::copy(seedU.begin(), seedU.end(), workerU[w].begin());
    std::copy(seedV.begin(), seedV.end(), workerV[w].begin());
  }

  auto runChunk = [&](std::size_t lo, std::size_t hi, std::size_t workerSlot) noexcept {
    internal::TraversalCtx<T> ctx{
        .tree = &tree,
        .perm = perm,
        .reorderedPts = reordered,
        .d = dim,
        .compOf = componentOf,
        .coreDist = coreDist.data(),
        .nodeSingleComp = nodeSingleComp.data(),
        .bestW = workerW[workerSlot].data(),
        .bestU = workerU[workerSlot].data(),
        .bestV = workerV[workerSlot].data(),
    };
    // One scratch stack per worker per round, reused across every @c singlePointScan call in
    // this slice. KDTree depth caps node visits at @c ~log2(n) plus the leaf points; @c 64 is
    // a generous initial reserve that elides reallocs in the worst case for @c n <= 1e6.
    std::vector<const KDTreeNode *> stack;
    stack.reserve(std::size_t{64});
    const T *reorderedBase = reordered.data();
    for (std::size_t slot = lo; slot < hi; ++slot) {
      const auto origI = static_cast<std::int32_t>(perm[slot]);
      const std::int32_t compI = componentOf[origI];
      const T coreI = coreDist.data()[origI];
      const T *rowI = reorderedBase + (slot * dim);
      internal::singlePointScan<T>(ctx, origI, compI, coreI, rowI, root, stack);
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
  (void)allSingletonComponents;
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
