#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <span>
#include <utility>
#include <vector>

#include "clustering/always_assert.h"
#include "clustering/index/range_query.h"
#include "clustering/math/aabb.h"
#include "clustering/math/detail/avx2_helpers.h"
#include "clustering/math/detail/radius_scan.h"
#include "clustering/math/detail/sq_distances_block.h"
#include "clustering/math/detail/top_k_neighbors.h"
#include "clustering/math/thread.h"
#include "clustering/memory/linear_alloc.h"
#include "clustering/ndarray.h"

namespace clustering {

/**
 * @brief Node in a @ref KDTree, sharing the same struct shape between internals and leaves.
 *
 * Internal node: @c m_index is the pivot's slot in the owning tree's reordered point buffer,
 * @c m_dim is the split dimension, and @c m_left / @c m_right point to children.
 *
 * Leaf node: @c m_index is the start offset into the same reordered buffer, @c m_dim is the
 * count of points packed at that offset, and @c m_left == @c m_right == @c nullptr.
 *
 * The leaf-vs-internal test is `m_left == nullptr` && m_right == nullptr. The struct is
 * declared outside @ref KDTree so the allocator can bump-allocate fixed-size slots without
 * seeing the tree's template parameters.
 */
struct KDTreeNode {
  /// Internal: pivot slot in the tree's reordered point buffer. Leaf: base offset into the
  /// same buffer (leaf points live at slots `[m_index, m_index + m_dim))`.
  std::size_t m_index;
  /// Internal: split dimension. Leaf: point count packed at @c m_index.
  std::size_t m_dim;
  /// Left child, or @c nullptr on a leaf.
  KDTreeNode *m_left;
  /// Right child, or @c nullptr on a leaf.
  KDTreeNode *m_right;
  /// Monotonic identifier assigned at construction; keys into the owning tree's per-node
  /// bounds buffer. Stays valid for the tree's lifetime and never collides across nodes of
  /// the same tree. Four bytes is enough: @ref KDTree's node budget never exceeds @c N,
  /// which the HDBSCAN contract caps at the signed 32-bit range.
  std::uint32_t m_id;
};

/**
 * @brief Distance metric a @ref KDTree builds pruning bounds under.
 *
 * Additional flavours (Manhattan, Cosine, etc.) would extend this enum and dispatch new
 * overloads of the pivot-to-child inequality used during pruning.
 */
enum class KDTreeDistanceType : std::uint8_t {
  kEucledian ///< Squared Euclidean; radii and comparisons run on squared distances.
};

/**
 * @brief Implements a KDTree data structure.
 *
 * KDTree is a space-partitioning data structure for organizing points in a K-dimensional space.
 * It is efficient in range-search and nearest neighbor search. This implementation contains
 * a vector of points and an allocator for KDTree nodes, with the root node representing the tree's
 * starting point.
 *
 * @tparam T Data type of the points.
 * @tparam LeafSize Maximum number of points in a leaf node before splitting (default 16).
 * @tparam AllocT Allocator type for KDTree nodes, defaults to LinearAllocator<KDTreeNode>.
 *
 * @warning The KDTree does not manage the lifecycle of the points array. Ensure the array remains
 * valid during KDTree's lifetime.
 *
 * @par Usage Example
 * \code{.cpp}
 * NDArray<float, 2> points = ...;
 * KDTree<float> kdtree(points);
 * auto indices = kdtree.query(query_point, radius);
 * for (size_t index : indices) {
 *   std::cout << index << std::endl;
 * }
 * \endcode
 */
template <class T, KDTreeDistanceType distanceType = KDTreeDistanceType::kEucledian,
          std::size_t LeafSize = 16, class AllocT = LinearAllocator<KDTreeNode>>
class KDTree {
public:
  using value_type = T; ///< Element type of the indexed point cloud.

  /**
   * @brief Constructs a KDTree using a given set of points.
   *
   * @details
   * Initializes the KDTree with a given array of points and prepares the allocator.
   * This constructor builds the KDTree by sorting the points and constructing nodes accordingly.
   *
   * @param points NDArray of points to build the KDTree.
   * @param pool   Parallelism injection for the recursive build; median splits above
   *               @c kParallelBuildFloor fork their subtrees onto the pool. The default
   *               serial mode produces the same tree, so the parameter only changes wall
   *               time, never structure.
   *
   * @note The points array must remain valid for the lifetime of the KDTree, as the tree does not
   * manage the array's lifecycle.
   */
  KDTree(const NDArray<T, 2> &points, math::Pool pool = {})
      : m_allocator(calculatePoolSize(points.dim(0))), m_points(points), m_dim(points.dim(1)) {
    CLUSTERING_ALWAYS_ASSERT(points.isContiguous());
    const std::size_t n = points.dim(0);
    m_indices.resize(n);
    std::iota(m_indices.begin(), m_indices.end(), 0);
    // Subtree node counts are a pure function of range size, so every node's arena slot and
    // id are known before recursion starts; parallel subtree builds write disjoint slots with
    // no shared allocation state and reproduce the serial layout exactly.
    const std::size_t totalNodes = nodeCountFor(n);
    KDTreeNode *arena = (totalNodes > 0) ? m_allocator.allocate(totalNodes) : nullptr;
    m_root = (totalNodes > 0) ? buildAt(0, n, 0, arena, 0, pool) : nullptr;
    m_nextNodeId = static_cast<std::uint32_t>(totalNodes);
    // Materialize points in tree-build order. After @c buildAt rewrites @c m_indices into a
    // permutation matching the tree layout, a leaf's points live at @c m_points_reordered slots
    // `[leaf.m_index, leaf.m_index + leaf.m_dim)`. Contiguous access there replaces the
    // scatter-indirection `m_points[m_indices[k]`] had, which at low d was the cache-miss
    // ceiling: every leaf-brute-force iteration landed on a random row of @c m_points.
    // Rows land in disjoint slots, so the copy fans out over the pool.
    m_points_reordered.resize(n * m_dim);
    const T *src = points.data();
    T *dst = m_points_reordered.data();
    pool.parallelForBlocks(std::size_t{0}, n, std::size_t{0}, [&](std::size_t lo, std::size_t hi) {
      for (std::size_t k = lo; k < hi; ++k) {
        const T *s = src + (m_indices[k] * m_dim);
        T *d = dst + (k * m_dim);
        for (std::size_t j = 0; j < m_dim; ++j) {
          d[j] = s[j];
        }
      }
    });
    // Populate per-node axis-aligned bounding boxes once the tree is built and the reordered
    // points buffer is materialized. Layout is `(numNodes * 2 * d)` flat: row @c 2*id holds the
    // min-coords vector, row @c 2*id + 1 holds the max-coords vector. Dual-tree walkers consume
    // this through @ref nodeBounds as a pair of @c std::span views; leaving @ref KDTreeNode's
    // size unchanged past the monotonic @c m_id keeps leaf-scan cache behaviour stable at
    // high @c d.
    //
    // Leaf boxes carry the point sweep, so they fan out over the arena; internal boxes are
    // unions of their children, and the post-order ids let a flat ascending-id pass visit
    // every child before its parent without recursion.
    m_nodeBounds.assign(static_cast<std::size_t>(m_nextNodeId) * 2 * m_dim, T{});
    if (m_root != nullptr) {
      KDTreeNode *arenaNodes = m_root;
      pool.parallelForBlocks(std::size_t{0}, totalNodes, std::size_t{0},
                             [&](std::size_t lo, std::size_t hi) {
                               for (std::size_t s = lo; s < hi; ++s) {
                                 const KDTreeNode &node = arenaNodes[s];
                                 if (node.m_left == nullptr && node.m_right == nullptr) {
                                   populateLeafBounds(node);
                                 }
                               }
                             });
      std::vector<const KDTreeNode *> byId(totalNodes, nullptr);
      for (std::size_t s = 0; s < totalNodes; ++s) {
        byId[arenaNodes[s].m_id] = &arenaNodes[s];
      }
      for (std::size_t id = 0; id < totalNodes; ++id) {
        const KDTreeNode &node = *byId[id];
        if (node.m_left != nullptr || node.m_right != nullptr) {
          unionChildBounds(node);
        }
      }
    }
  }

  /**
   * @brief Finds points within a specified radius of a query point.
   *
   * @details
   * This method returns the indices of points within a given radius from the query point.
   * It searches the KDTree efficiently by pruning branches that fall outside the search radius.
   *
   * @param query_point The point around which to search.
   * @param radius The search radius.
   * @param limit Maximum number of points to return. If -1, returns all points within the radius.
   * @return Vector of indices of points within the search radius.
   */
  std::vector<std::size_t> query(const NDArray<T, 1> &query_point, T radius,
                                 std::int64_t limit = -1) const {
    std::vector<std::size_t> indices;
    const T radius_sq = radius * radius;
    ensureLeafSoa();
    std::vector<KDTreeNode *> stack;
    stack.reserve(kDefaultStackReserve);
    // Copy the borrowed row into a scratch buffer so the core walker can assume a contiguous
    // `d`-element block regardless of the source layout. The copy is d stores -- free at d=2.
    std::vector<T> qbuf(m_dim);
    for (std::size_t k = 0; k < m_dim; ++k) {
      qbuf[k] = query_point[k];
    }
    queryImpl(m_root, qbuf.data(), radius_sq, indices, stack, limit);
    return indices;
  }

  /**
   * @brief Returns the full radius-neighborhood adjacency over the indexed point cloud.
   *
   * Walks the tree once per row, fanning the outer loop out over @p pool. Lets DBSCAN consume
   * the whole neighbor graph in one call rather than threading a per-point query loop through
   * the caller. Every row is complete in both directions -- the walk finds all neighbours of
   * its query point -- so the core flag is a per-row size check taken in the same pass.
   *
   * @param radius Non-negative neighbourhood radius; comparison runs on the squared distance.
   * @param minPts Core threshold on the self-inclusive neighbour count.
   * @param pool   Parallelism injection used to fan the outer row loop out across workers.
   * @return Rows and core flags per the @ref clustering::index::CoreAdjacency contract.
   */
  [[nodiscard]] index::CoreAdjacency query(T radius, std::size_t minPts, math::Pool pool) const {
    const std::size_t n = m_points.dim(0);
    index::CoreAdjacency out;
    out.rows.resize(n);
    out.isCore.assign(n, 0);
    if (n == 0) {
      return out;
    }

    const T radius_sq = radius * radius;
    ensureLeafSoa(pool);

    // Above the dim floor, one box-pruned walk per leaf replaces one walk per point: the
    // visited-node count collapses by the leaf occupancy while the pair tests grow only by the
    // box inflation, which the higher per-pair cost at larger d amortizes. Below the floor the
    // per-pair scan is a few ops and the inflation dominates, so points walk individually.
    if (m_dim >= kBlockQueryDimFloor && m_root != nullptr) {
      blockQuery(radius_sq, minPts, pool, out);
      return out;
    }

    // Leaves whose bounding-box diagonal fits inside the radius are cliques: every member is
    // within eps of every other, and with at least minPts members every member is a core. A
    // query ball that swallows such a leaf can take one representative edge instead of
    // materializing the members, so flag them once up front.
    const std::size_t totalNodes = nodeCount();
    const KDTreeNode *arenaNodes = m_root;
    std::vector<std::uint8_t> leafAllCore(totalNodes, 0);
    pool.parallelForBlocks(
        std::size_t{0}, totalNodes, std::size_t{0}, [&](std::size_t lo, std::size_t hi) {
          for (std::size_t nodeSlot = lo; nodeSlot < hi; ++nodeSlot) {
            const KDTreeNode &node = arenaNodes[nodeSlot];
            if (node.m_left != nullptr || node.m_right != nullptr || node.m_dim < minPts) {
              continue;
            }
            const auto [bmin, bmax] = nodeBounds(&node);
            T diagSq = T{0};
            for (std::size_t j = 0; j < m_dim; ++j) {
              const T ext = bmax[j] - bmin[j];
              diagSq += ext * ext;
            }
            if (diagSq <= radius_sq) {
              leafAllCore[node.m_id] = 1;
            }
          }
        });

    const std::size_t workers = pool.workerCount();
    std::vector<std::vector<std::pair<std::int32_t, std::int32_t>>> workerEdges(workers);

    auto runRange = [&](std::size_t lo, std::size_t hi) {
      // Reuse the traversal stack across every query in this chunk. Tree depth stays below
      // log2(n / LeafSize) + a few for spilled internal pushes, so kDefaultStackReserve rarely
      // grows past its initial capacity; clearing keeps the allocation alive between queries.
      std::vector<KDTreeNode *> stack;
      stack.reserve(kDefaultStackReserve);
      const std::size_t adjReserveFloor =
          std::min(n, (m_dim == kWideAdjReserveDim) ? kWideAdjReserveFloor : kAdjReserveFloor);
      const T *reordered = m_points_reordered.data();
      std::vector<std::pair<std::int32_t, std::int32_t>> &edges =
          workerEdges[math::Pool::workerIndex()];
      const bool useSoa = !m_leafSoa.empty();
      for (std::size_t k = lo; k < hi; ++k) {
        // Walk in tree-build order so consecutive queries share tree paths and keep the visited
        // nodes warm in cache; m_indices[k] maps the reordered row back to its original slot.
        const T *qp = reordered + (k * m_dim);
        const std::size_t rowIdx = m_indices[k];
        std::vector<std::int32_t> &row = out.rows[rowIdx];
        // Each row is filled by exactly this query. Seed a reserve floor so the first survivors do
        // not walk the vector-doubling reallocation cascade from a zero-capacity start.
        row.reserve(adjReserveFloor);

        // Clique-leaf shortcut: a swallowed all-core leaf contributes its whole population to
        // the degree and one representative edge to the component build. Taking the shortcut
        // proves this point core (its final degree can only exceed the running guard), so the
        // thinned row stays within the core-row contract; a point that never clears the guard
        // scans normally and keeps a complete row.
        std::size_t bulkDegree = 0;
        stack.clear();
        stack.push_back(m_root);
        while (!stack.empty()) {
          KDTreeNode *node = stack.back();
          stack.pop_back();
          if (node == nullptr) {
            continue;
          }
          const auto [bmin, bmax] = nodeBounds(node);
          if (math::pointAabbGapSq(qp, bmin, bmax) > radius_sq) {
            continue;
          }
          if (node->m_left == nullptr && node->m_right == nullptr) {
            const std::size_t base = node->m_index;
            const std::size_t count = node->m_dim;
            if (leafAllCore[node->m_id] != 0 && row.size() + bulkDegree + count >= minPts &&
                math::pointAabbFarthestSq(qp, bmin, bmax) <= radius_sq) {
              bulkDegree += count;
              const auto rep = static_cast<std::int32_t>(m_indices[base]);
              const auto self = static_cast<std::int32_t>(rowIdx);
              if (rep != self) {
                edges.emplace_back(self, rep);
              }
              continue;
            }
            const T *leafPts = reordered + (base * m_dim);
            auto emit = [&](std::size_t i) noexcept {
              row.push_back(static_cast<std::int32_t>(m_indices[base + i]));
            };
            if (useSoa) {
              math::detail::radiusScanSoa(qp, m_leafSoa.data() + (base * m_dim), count, m_dim,
                                          radius_sq, emit);
            } else {
              math::detail::radiusScan(qp, leafPts, count, m_dim, radius_sq, emit);
            }
            continue;
          }
          const std::size_t pivotSlot = node->m_index;
          const T *pivotRow = reordered + (pivotSlot * m_dim);
          if (math::detail::sqEuclideanRowPtr(qp, pivotRow, m_dim) <= radius_sq) {
            row.push_back(static_cast<std::int32_t>(m_indices[pivotSlot]));
          }
          stack.push_back(node->m_left);
          stack.push_back(node->m_right);
        }
        out.isCore[rowIdx] =
            (row.size() + bulkDegree >= minPts) ? std::uint8_t{1} : std::uint8_t{0};
      }
    };

    if (pool.shouldParallelize(n, 4, 2)) {
      // Oversubscribe blocks so dynamic stealing balances the skewed per-query degree; one block
      // per worker lets a dense-neighbourhood region gate the join while the rest idle. Each query
      // is a full tree walk, so the block floor is finer than the row-light default to keep enough
      // blocks per worker at small n.
      pool.parallelForBlocks<citor::HintsDefaults>(
          std::size_t{0}, n, pool.stealBlocks(n, 64),
          [&](std::size_t lo, std::size_t hi) { runRange(lo, hi); });
    } else {
      runRange(0, n);
    }
    for (const auto &edges : workerEdges) {
      out.extraEdges.insert(out.extraEdges.end(), edges.begin(), edges.end());
    }
    return out;
  }

  /**
   * @brief Returns the k nearest neighbours of every indexed point, self-excluded.
   *
   * For every row @c i of the original point cloud, the method finds the @c k original-index
   * points minimizing squared Euclidean distance to row @c i (excluding @c i itself) and writes
   * the result as two parallel `(n x k)` arrays: `indices[i]`[j] is the original index of the
   * @c j-th closest neighbour, `sqDists[i]`[j] is its squared distance. Each row is sorted
   * ascending by @c sqDist. Ties in distance resolve on smaller neighbour index so results are
   * reproducible bit-for-bit across runs at matched input.
   *
   * Traversal is depth-first with the bounded max-heap's current worst retained distance serving
   * as the pruning bound (`heap.top()`.first). Until the heap fills, the bound is
   * @c std::numeric_limits<T>::max() and pruning is inactive.
   *
   * @pre `k >= 1` and @c k < n.
   *
   * @param k    Number of neighbours per point (self-excluded). The signed 32-bit argument mirrors
   *             the neighbour-index type carried through the `(indices, sqDists)` output; the
   *             preconditions clamp it to `[1, n)`.
   * @param pool Parallelism injection for the outer per-point loop; follows the
   *             @c shouldParallelize policy the radius-query path uses.
   * @return Pair of two arrays: @c .first is an `(n x k)` @c std::int32_t array of neighbour
   *         indices; @c .second is an `(n x k)` @c T array of squared distances.
   */
  [[nodiscard]] std::pair<NDArray<std::int32_t, 2>, NDArray<T, 2>> knnQuery(std::int32_t k,
                                                                            math::Pool pool) const {
    const std::size_t n = m_points.dim(0);
    CLUSTERING_ALWAYS_ASSERT(k >= 1);
    CLUSTERING_ALWAYS_ASSERT(std::cmp_less(k, n));

    const auto kSz = static_cast<std::size_t>(k);
    NDArray<std::int32_t, 2> indices({n, kSz});
    NDArray<T, 2> sqDists({n, kSz});

    auto runRange = [&](std::size_t lo, std::size_t hi) {
      // Reuse the traversal stack and top-k tracker across every query in this chunk; both are
      // reset at the head of each per-point walk.
      std::vector<KDTreeNode *> stack;
      stack.reserve(kDefaultStackReserve);
      math::detail::TopKNeighbors<T, std::int32_t> topK(kSz);
      const T *sourceData = m_points.data();
      std::int32_t *idxOut = indices.data();
      T *distOut = sqDists.data();
      for (std::size_t i = lo; i < hi; ++i) {
        const auto iOriginal = static_cast<std::int32_t>(i);
        const T *qp = sourceData + (i * m_dim);
        topK.clear();
        knnQueryImpl(m_root, qp, iOriginal, topK, stack);
        topK.drainAscending(distOut + (i * kSz), idxOut + (i * kSz));
      }
    };

    if (pool.shouldParallelize(n, 4, 2)) {
      pool.parallelForBlocks<citor::HintsDefaults>(
          std::size_t{0}, n, std::size_t{0},
          [&](std::size_t lo, std::size_t hi) { runRange(lo, hi); });
    } else {
      runRange(0, n);
    }
    return {std::move(indices), std::move(sqDists)};
  }

  /**
   * @brief Axis-aligned bounding box of the points routed through @p node.
   *
   * For a leaf, the bounds enclose every point stored at the leaf. For an internal node, the
   * bounds enclose the union of the child boxes plus the pivot. Callers consume the result as
   * two @c d-element spans into the tree's contiguous bounds buffer; the pointers are valid for
   * the tree's lifetime.
   *
   * @param node Node to query; must belong to this tree. Passing @c nullptr is a precondition
   *             violation.
   * @return Pair of spans `(min, max)`, each of length @c d.
   */
  [[nodiscard]] std::pair<std::span<const T>, std::span<const T>>
  nodeBounds(const KDTreeNode *node) const noexcept {
    assert(node != nullptr && "KDTree::nodeBounds on null node");
    const std::size_t base = static_cast<std::size_t>(node->m_id) * 2 * m_dim;
    const T *bounds = m_nodeBounds.data() + base;
    return {std::span<const T>(bounds, m_dim), std::span<const T>(bounds + m_dim, m_dim)};
  }

  /**
   * @brief Permutation from reordered-slot index to original point index.
   *
   * Element @c k is the original row index of the point stored at reordered slot @c k. A leaf
   * with @c m_index = base and @c m_dim = count owns slots `[base, base + count)`; its
   * original indices are `permutation[base .. base + count - 1]`. Stable for the tree's
   * lifetime; pointer invalidation follows the tree's move and destruction.
   *
   * @return Length- @c N span over the permutation buffer.
   */
  [[nodiscard]] std::span<const std::size_t> indexPermutation() const noexcept {
    return {m_indices.data(), m_indices.size()};
  }

  /**
   * @brief Points in reordered (tree-build) order as a flat row-major buffer.
   *
   * The element at flat offset `(slot * d + j)` is the @c j-th coordinate of the point stored
   * at reordered slot @c slot; this equals `originalPoints(permutation[slot], j)`. Consumers
   * that already have a @c KDTreeNode in hand can index through this buffer contiguously rather
   * than chasing the permutation; leaf ranges are therefore a @c count x d block of neighbouring
   * cache lines with no scatter on the leaf scan.
   *
   * @return Length- @c N*d span over the reordered-points buffer.
   */
  [[nodiscard]] std::span<const T> reorderedPoints() const noexcept {
    return {m_points_reordered.data(), m_points_reordered.size()};
  }

  /**
   * @brief Total node count, equal to one past the largest @c m_id assigned during construction.
   *
   * Callers that side-table per-node state (e.g. the Boruvka per-round single-component cache)
   * size their buffer to this count and index by `node->m_id`.
   */
  [[nodiscard]] std::size_t nodeCount() const noexcept {
    return static_cast<std::size_t>(m_nextNodeId);
  }

  /// Root of the tree; @c nullptr for an empty point set.
  [[nodiscard]] const KDTreeNode *root() const noexcept { return m_root; }

  /// Dimension count of the indexed point cloud.
  [[nodiscard]] std::size_t dim() const noexcept { return m_dim; }

  /**
   * @brief Destroys the KDTree, deallocating its nodes.
   *
   * @details
   * Deallocates all nodes of the KDTree, if the allocator supports deallocation.
   * This is to ensure that no memory leaks occur from dynamically allocated nodes.
   */
  ~KDTree() {
    if (m_allocator.isDeallocSupported()) {
      doRecDealloc(m_root);
    }
  }

private:
  /**
   * @brief Calculates the required memory pool size for the KDTree's allocator.
   *
   * @details
   * Ensures that the allocator has sufficient memory to accommodate all nodes of the KDTree.
   * The calculated pool size is based on the number of points and the size of a KDTreeNode.
   *
   * @param numPoints Number of points in the KDTree.
   * @return Number of nodes to allocate.
   */
  static size_t calculatePoolSize(size_t numPoints) {
    if (numPoints == 0) {
      return 0;
    }
    // With leaf-size tuning, the number of nodes is at most numPoints
    // (much less than the original 2*numPoints-1, since leaf nodes batch
    // up to LeafSize points each).
    return numPoints;
  }

  /**
   * @brief Recursively deallocates KDTree nodes.
   *
   * @details
   * Used by the KDTree destructor to free memory allocated for each node in the tree.
   *
   * @param node Current node to deallocate.
   */
  static void doRecDealloc(KDTreeNode *node) {
    if (node == nullptr) {
      return;
    }

    doRecDealloc(node->m_left);
    doRecDealloc(node->m_right);

    delete node;
  }

  /// Subtree size at or below which @c buildAt stops forking and recurses inline. Around
  /// this size one median split costs about as much as a task spawn plus steal, so finer
  /// forking only adds queue traffic.
  static constexpr std::size_t kParallelBuildFloor = 512;

  /**
   * @brief Node count of the subtree @c buildAt produces for a range of @p m points.
   *
   * Mirrors the split recursion exactly: the pivot leaves `(m - 1) / 2` points on the left
   * and the remainder on the right. @c buildAt leans on this to pre-partition arena slots
   * and node ids at every fork.
   */
  static std::size_t nodeCountFor(std::size_t m) noexcept {
    if (m == 0) {
      return 0;
    }
    if (m <= LeafSize) {
      return 1;
    }
    const std::size_t mLeft = (m - 1) / 2;
    return 1 + nodeCountFor(mLeft) + nodeCountFor(m - 1 - mLeft);
  }

  /**
   * @brief Recursively builds the KDTree over `m_indices[start, end)` into pre-assigned slots.
   *
   * Selects the median along the depth-cycled dimension, partitions the range, and recurses
   * into both halves. The subtree's nodes occupy the contiguous arena block
   * `[arena, arena + nodeCountFor(end - start))` with the parent at @c arena in front of the
   * left then right child blocks, and carry post-order ids from @p idBase up; both follow
   * from @c nodeCountFor alone, so sibling subtrees touch disjoint arena slots, disjoint
   * id ranges, and disjoint @c m_indices subranges. Splits above @c kParallelBuildFloor
   * hand the two halves to `pool.forkJoin2`; the tree is identical either way, and identical
   * to what the fully serial recursion produced.
   *
   * @param start  First index of the range in @c m_indices.
   * @param end    One past the last index of the range.
   * @param depth  Current depth; selects the split dimension.
   * @param arena  First arena slot of this subtree's contiguous node block.
   * @param idBase First node id of this subtree's id range.
   * @param pool   Parallelism injection for the fork decision.
   * @return A pointer to the constructed subtree root, or @c nullptr for an empty range.
   */
  KDTreeNode *buildAt(std::size_t start, std::size_t end, std::size_t depth, KDTreeNode *arena,
                      std::uint32_t idBase, math::Pool pool) {
    if (start >= end) {
      return nullptr;
    }

    // Leaf node: store range into m_indices, brute-force at query time
    if (end - start <= LeafSize) {
      *arena = {.m_index = start,     // offset into m_indices
                .m_dim = end - start, // count of points
                .m_left = nullptr,
                .m_right = nullptr,
                .m_id = idBase};
      return arena;
    }

    // Internal node: split on median
    const std::size_t dim = depth % m_points.dim(1);
    const std::size_t median = start + (((end - start) - 1) / 2);

    using diff_t = std::vector<std::size_t>::difference_type;
    std::nth_element(m_indices.begin() + static_cast<diff_t>(start),
                     m_indices.begin() + static_cast<diff_t>(median),
                     m_indices.begin() + static_cast<diff_t>(end),
                     [this, dim](std::size_t lhs, std::size_t rhs) {
                       return m_points[lhs][dim] < m_points[rhs][dim];
                     });

    const std::size_t nLeft = nodeCountFor(median - start);
    const std::size_t nRight = nodeCountFor(end - median - 1);
    KDTreeNode *left = nullptr;
    KDTreeNode *right = nullptr;
    auto buildLeft = [&] { left = buildAt(start, median, depth + 1, arena + 1, idBase, pool); };
    auto buildRight = [&] {
      right = buildAt(median + 1, end, depth + 1, arena + 1 + nLeft,
                      idBase + static_cast<std::uint32_t>(nLeft), pool);
    };
    if (end - start > kParallelBuildFloor && pool.pool != nullptr) {
      pool.forkJoin2(buildLeft, buildRight);
    } else {
      buildLeft();
      buildRight();
    }

    *arena = {.m_index = median, // reordered slot of the pivot
              .m_dim = dim,
              .m_left = left,
              .m_right = right,
              .m_id = idBase + static_cast<std::uint32_t>(nLeft + nRight)};

    return arena;
  }

  /// Dimension at or below which the radius sweep keeps one walk per point; from here up a
  /// leaf's points share one box-pruned walk.
  static constexpr std::size_t kBlockQueryDimFloor = 4;

  /**
   * @brief Radius sweep that walks the tree once per source leaf instead of once per point.
   *
   * For every source leaf, one depth-first walk prunes on the box-to-box gap between the leaf's
   * bounds and each visited node; surviving target leaves are scanned once per source point and
   * surviving pivots are distance-tested against the whole source block. Every neighbour of a
   * source point sits inside some visited node because the box gap lower-bounds the point
   * distance, so rows come out complete and the core flags fall out of the same pass. Pivot
   * points are not covered by any leaf's slot range; they keep their per-point walks.
   *
   * Rows are owned by their source leaf's walk (pivot rows by the pivot's own walk), so the
   * fan-out over leaves stays race-free.
   */
  void blockQuery(T radius_sq, std::size_t minPts, math::Pool pool,
                  index::CoreAdjacency &out) const {
    const std::size_t n = m_points.dim(0);
    const std::size_t totalNodes = nodeCount();
    const KDTreeNode *arenaNodes = m_root;
    std::vector<const KDTreeNode *> leaves;
    std::vector<const KDTreeNode *> pivots;
    leaves.reserve(totalNodes);
    for (std::size_t nodeSlot = 0; nodeSlot < totalNodes; ++nodeSlot) {
      const KDTreeNode &node = arenaNodes[nodeSlot];
      if (node.m_left == nullptr && node.m_right == nullptr) {
        leaves.push_back(&node);
      } else {
        pivots.push_back(&node);
      }
    }

    const T *reordered = m_points_reordered.data();
    const std::size_t adjReserveFloor =
        std::min(n, (m_dim == kWideAdjReserveDim) ? kWideAdjReserveFloor : kAdjReserveFloor);
    const bool useSoa = !m_leafSoa.empty();

    auto runLeafRange = [&](std::size_t lo, std::size_t hi) {
      std::vector<const KDTreeNode *> stack;
      stack.reserve(kDefaultStackReserve);
      for (std::size_t li = lo; li < hi; ++li) {
        const KDTreeNode &source = *leaves[li];
        const std::size_t base = source.m_index;
        const std::size_t count = source.m_dim;
        const auto [srcMin, srcMax] = nodeBounds(&source);
        for (std::size_t i = 0; i < count; ++i) {
          out.rows[m_indices[base + i]].reserve(adjReserveFloor);
        }

        stack.clear();
        stack.push_back(m_root);
        while (!stack.empty()) {
          const KDTreeNode *node = stack.back();
          stack.pop_back();
          if (node == nullptr) {
            continue;
          }
          const auto [nodeMin, nodeMax] = nodeBounds(node);
          if (math::aabbAabbGapSq(srcMin, srcMax, nodeMin, nodeMax) > radius_sq) {
            continue;
          }
          if (node->m_left == nullptr && node->m_right == nullptr) {
            const std::size_t targetBase = node->m_index;
            const std::size_t targetCount = node->m_dim;
            const T *targetPts = reordered + (targetBase * m_dim);
            const T *targetSoa = useSoa ? m_leafSoa.data() + (targetBase * m_dim) : nullptr;
            std::size_t i = 0;
            if (useSoa) {
              // Paired sources share each target-column load and one call setup.
              for (; i + 2 <= count; i += 2) {
                std::vector<std::int32_t> &row0 = out.rows[m_indices[base + i]];
                std::vector<std::int32_t> &row1 = out.rows[m_indices[base + i + 1]];
                math::detail::radiusScanSoaPair(
                    reordered + ((base + i) * m_dim), reordered + ((base + i + 1) * m_dim),
                    targetSoa, targetCount, m_dim, radius_sq,
                    [&](std::size_t j) noexcept {
                      row0.push_back(static_cast<std::int32_t>(m_indices[targetBase + j]));
                    },
                    [&](std::size_t j) noexcept {
                      row1.push_back(static_cast<std::int32_t>(m_indices[targetBase + j]));
                    });
              }
            }
            for (; i < count; ++i) {
              const T *qp = reordered + ((base + i) * m_dim);
              std::vector<std::int32_t> &row = out.rows[m_indices[base + i]];
              auto emit = [&](std::size_t j) noexcept {
                row.push_back(static_cast<std::int32_t>(m_indices[targetBase + j]));
              };
              if (useSoa) {
                math::detail::radiusScanSoa(qp, targetSoa, targetCount, m_dim, radius_sq, emit);
              } else {
                math::detail::radiusScan(qp, targetPts, targetCount, m_dim, radius_sq, emit);
              }
            }
            continue;
          }
          const T *pivotRow = reordered + (node->m_index * m_dim);
          const auto pivotIdx = static_cast<std::int32_t>(m_indices[node->m_index]);
          for (std::size_t i = 0; i < count; ++i) {
            const T *qp = reordered + ((base + i) * m_dim);
            if (math::detail::sqEuclideanRowPtr(qp, pivotRow, m_dim) <= radius_sq) {
              out.rows[m_indices[base + i]].push_back(pivotIdx);
            }
          }
          stack.push_back(node->m_left);
          stack.push_back(node->m_right);
        }

        for (std::size_t i = 0; i < count; ++i) {
          const std::size_t rowIdx = m_indices[base + i];
          out.isCore[rowIdx] =
              (out.rows[rowIdx].size() >= minPts) ? std::uint8_t{1} : std::uint8_t{0};
        }
      }
    };

    auto runPivotRange = [&](std::size_t lo, std::size_t hi) {
      std::vector<KDTreeNode *> stack;
      stack.reserve(kDefaultStackReserve);
      for (std::size_t pi = lo; pi < hi; ++pi) {
        const std::size_t slot = pivots[pi]->m_index;
        const T *qp = reordered + (slot * m_dim);
        const std::size_t rowIdx = m_indices[slot];
        std::vector<std::int32_t> &row = out.rows[rowIdx];
        row.reserve(adjReserveFloor);
        queryImpl(m_root, qp, radius_sq, row, stack, /*limit=*/-1);
        out.isCore[rowIdx] = (row.size() >= minPts) ? std::uint8_t{1} : std::uint8_t{0};
      }
    };

    if (pool.shouldParallelize(n, 4, 2)) {
      pool.parallelForBlocks<citor::HintsDefaults>(
          std::size_t{0}, leaves.size(), pool.stealBlocks(leaves.size(), 1), runLeafRange);
      pool.parallelForBlocks(std::size_t{0}, pivots.size(), std::size_t{0}, runPivotRange);
    } else {
      runLeafRange(0, leaves.size());
      runPivotRange(0, pivots.size());
    }
  }

  /**
   * @brief Core iterative range-search driver.
   *
   * Takes a contiguous @c d-element query pointer and a caller-owned traversal stack so the
   * adjacency sweep can amortize the stack buffer across every query in a chunk. @p indices is
   * an output buffer of any integral type the caller wants labels in (rank-1 adjacency wants
   * @c std::int32_t; the public one-shot wrapper wants @c std::size_t). @p limit preserves the
   * behaviour of the historical overload: `-1` emits everything, otherwise stop once the
   * output reaches the cap.
   *
   * @tparam OutIdx Integral type used to store hit indices. The cast from @c std::size_t is
   *                explicit so downstream narrowing is localized here, not in callers.
   *
   * @param root      Root node of the subtree to search.
   * @param qp        Pointer to the contiguous query point; size-@c m_dim.
   * @param radius_sq Squared radius; comparisons are squared-distance.
   * @param indices   Output vector of hit indices.
   * @param stack     Caller-owned traversal scratch; cleared at the start of each call, reused
   *                  across subsequent calls within a range chunk.
   * @param limit     `-1` for unbounded; otherwise stop once `indices.size()` reaches it.
   */
  template <class OutIdx>
  void queryImpl(KDTreeNode *root, const T *qp, T radius_sq, std::vector<OutIdx> &indices,
                 std::vector<KDTreeNode *> &stack, std::int64_t limit = -1) const {
    if (root == nullptr) {
      return;
    }

    stack.clear();
    stack.push_back(root);

    const T *reorderedBase = m_points_reordered.data();

    while (!stack.empty()) {
      const KDTreeNode *node = stack.back();
      stack.pop_back();

      if (node == nullptr) {
        continue;
      }

      if (limit != -1 && indices.size() == static_cast<std::size_t>(limit)) {
        break;
      }

      // Leaf node: brute-force all points in the range. @c m_points_reordered lays points in
      // tree-build order, so a leaf's entries are a contiguous @c count x d block -- one or
      // two cache lines at small @c d. @ref math::detail::radiusScan dispatches into a SIMD
      // kernel when @c d matches a batched width (today @c f32, `d == 2`) and otherwise
      // falls back to the scalar @ref sqEuclideanRowPtr per-row primitive.
      if (node->m_left == nullptr && node->m_right == nullptr) {
        const std::size_t base = node->m_index;
        const std::size_t count = node->m_dim;
        const T *leafPts = reorderedBase + (base * m_dim);
        // The SoA leaf copy exists only when @c d clears @ref kSoaLeafDimFloor; it is built lazily
        // on the first radius query, and when present scans without a horizontal sum; otherwise the
        // AoS scan runs.
        const bool useSoa = !m_leafSoa.empty();
        const T *leafSoa = useSoa ? m_leafSoa.data() + (base * m_dim) : nullptr;
        const bool isBounded = (limit != -1);
        if (!isBounded) {
          // Unbounded radius query (the DBSCAN adjacency sweep): every survivor is kept, so the
          // leaf-scan emit skips the per-point capacity test the bounded path needs.
          auto emit = [&](std::size_t i) noexcept {
            indices.push_back(static_cast<OutIdx>(m_indices[base + i]));
          };
          if (useSoa) {
            math::detail::radiusScanSoa(qp, leafSoa, count, m_dim, radius_sq, emit);
          } else {
            math::detail::radiusScan(qp, leafPts, count, m_dim, radius_sq, emit);
          }
          continue;
        }
        const auto cap = static_cast<std::size_t>(limit);
        auto emit = [&](std::size_t i) noexcept {
          if (indices.size() >= cap) {
            return;
          }
          indices.push_back(static_cast<OutIdx>(m_indices[base + i]));
        };
        if (useSoa) {
          math::detail::radiusScanSoa(qp, leafSoa, count, m_dim, radius_sq, emit);
        } else {
          math::detail::radiusScan(qp, leafPts, count, m_dim, radius_sq, emit);
        }
        if (indices.size() >= cap) {
          break;
        }
        continue;
      }

      // Internal node: check the split point and traverse children. `node->m_index` is the
      // pivot's slot in @c m_points_reordered, so the pivot's row lives at
      // @c reorderedBase + slot * m_dim. The original point index (needed for @c indices)
      // comes from `m_indices[slot]`.
      const std::size_t pivotSlot = node->m_index;
      const std::size_t splitDim = node->m_dim;
      const T *pivotRow = reorderedBase + (pivotSlot * m_dim);
      const T dist_sq = math::detail::sqEuclideanRowPtr(qp, pivotRow, m_dim);
      if (dist_sq <= radius_sq) {
        indices.push_back(static_cast<OutIdx>(m_indices[pivotSlot]));
      }

      const T pivotCoord = pivotRow[splitDim];
      const T diff = qp[splitDim] - pivotCoord;
      // Range queries scan the near side unconditionally and the far side only when the split
      // plane lies within the radius. Discovery order does not matter here -- every matching leaf
      // is scanned and the adjacency is order independent -- so the symmetric near / far branches
      // collapse into two straight-line conditional pushes. `diff < 0` selects which child is the
      // near side; `farWithinRadius` admits the other when the plane is reachable.
      const bool farWithinRadius = diff * diff <= radius_sq;
      if (node->m_left != nullptr && (diff < 0 || farWithinRadius)) {
        stack.push_back(node->m_left);
      }
      if (node->m_right != nullptr && (diff >= 0 || farWithinRadius)) {
        stack.push_back(node->m_right);
      }
    }
  }

  /**
   * @brief Depth-first kNN walker with small top-@c k tracker pruning.
   *
   * Admits every candidate point to @p topK, whose retained worst-key serves as the pruning
   * bound. Until the tracker fills (holds @c k entries), the bound is `+inf` and the walker
   * accepts all subtrees; once full, subtrees whose minimum possible distance along the split
   * axis exceeds the bound are skipped. The tracker's `O(1)` reject path on the common
   * "new distance is not smaller than the current worst" case replaces the heap sift that
   * dominated this walker's cycle budget at small @c k.
   *
   * @param root       Root node of the subtree to search.
   * @param qp         Pointer to the contiguous query point; size-@c m_dim.
   * @param selfIndex  Original-index slot to exclude from the result (the query's own row).
   * @param topK       Caller-owned top-@c k tracker; cleared by the public @ref knnQuery driver.
   * @param stack      Caller-owned traversal scratch; cleared at the head of each call, reused
   *                   across subsequent calls within a range chunk.
   */
  void knnQueryImpl(KDTreeNode *root, const T *qp, std::int32_t selfIndex,
                    math::detail::TopKNeighbors<T, std::int32_t> &topK,
                    std::vector<KDTreeNode *> &stack) const {
    if (root == nullptr) {
      return;
    }

    stack.clear();
    stack.push_back(root);

    const T *reorderedBase = m_points_reordered.data();

    while (!stack.empty()) {
      const KDTreeNode *node = stack.back();
      stack.pop_back();

      if (node == nullptr) {
        continue;
      }

      // Current pruning bound. Until the tracker fills, accept everything; once full, the
      // retained worst-key serves as the upper bound.
      const T bound = topK.boundKey();

      // AABB gap prune: the minimum possible squared distance from the query to any point
      // under @c node is @c pointAabbGapSq against the subtree's bounding box. The parent's
      // single-axis prune is a strictly weaker lower bound; at d>=4 the full-AABB gap skips
      // subtrees the axis prune lets through, which at d=8 is the dominant share of the kNN
      // walker's remaining internal-node visits.
      if (bound != std::numeric_limits<T>::max()) {
        auto [nmin, nmax] = this->nodeBounds(node);
        const T gapSq = math::pointAabbGapSq<T>(qp, nmin, nmax);
        if (gapSq >= bound) {
          continue;
        }
      }

      if (node->m_left == nullptr && node->m_right == nullptr) {
        // Leaf: walk the contiguous count x d block and admit each candidate. Self-exclusion
        // is by original index so a candidate is skipped iff it is the query row. Distances
        // are computed in blocks of four so the horizontal-sum epilogue is shared across
        // neighbours; the top-k admit then runs over the precomputed dsq vector.
        const std::size_t base = node->m_index;
        const std::size_t count = node->m_dim;
        const T *leafPts = reorderedBase + (base * m_dim);
        std::array<T, LeafSize> dsqBuf{};
        math::detail::sqDistancesAosBlock<T>(qp, leafPts, count, m_dim, dsqBuf.data());
        // Batch-level prune: if every distance in the leaf exceeds the current worst retained
        // bound, no candidate can enter the tracker. Scan for the minimum once; the common
        // case at LeafSize=64 with k in [4, 32] is that every distance falls above the bound
        // and the per-entry admit loop (self-exclude cmp, index load, topK.push) is skipped
        // entirely. The scan runs scalar because LeafSize is known at compile time and the
        // auto-vectoriser produces a tight min-reduce with no hsum epilogue of its own.
        if (topK.full()) {
          T minDsq = dsqBuf[0];
          for (std::size_t i = 1; i < count; ++i) {
            if (dsqBuf[i] < minDsq) {
              minDsq = dsqBuf[i];
            }
          }
          if (minDsq >= bound) {
            continue;
          }
        }
        for (std::size_t i = 0; i < count; ++i) {
          const auto pointIdx = static_cast<std::int32_t>(m_indices[base + i]);
          if (pointIdx == selfIndex) {
            continue;
          }
          topK.push(dsqBuf[i], pointIdx);
        }
        continue;
      }

      // Internal: test the pivot, then descend near child first so the bound tightens before
      // the far-child prune test. Compute the split-axis delta squared first so both the pivot
      // admit and the descend-far gate can share it: the full d-wide pivot distance is at least
      // the split-axis contribution, so a larger delta squared proves the pivot cannot enter
      // the retained set and the d-wide hsum is skipped entirely.
      const std::size_t pivotSlot = node->m_index;
      const std::size_t splitDim = node->m_dim;
      const T *pivotRow = reorderedBase + (pivotSlot * m_dim);
      const auto pivotIdx = static_cast<std::int32_t>(m_indices[pivotSlot]);
      const T pivotCoord = pivotRow[splitDim];
      const T diff = qp[splitDim] - pivotCoord;
      const T diffSq = diff * diff;
      if (pivotIdx != selfIndex && diffSq <= bound) {
        const T dist_sq = math::detail::sqEuclideanRowPtr(qp, pivotRow, m_dim);
        topK.push(dist_sq, pivotIdx);
      }
      // DFS descends near-first, far-second. Stack is LIFO, so push far before near.
      if (diff < 0) {
        if (diffSq <= bound && node->m_right != nullptr) {
          stack.push_back(node->m_right);
        }
        if (node->m_left != nullptr) {
          stack.push_back(node->m_left);
        }
      } else {
        if (diffSq <= bound && node->m_left != nullptr) {
          stack.push_back(node->m_left);
        }
        if (node->m_right != nullptr) {
          stack.push_back(node->m_right);
        }
      }
    }
  }

  /// Build the feature-major leaf copy on first use; a no-op when @c d is below the floor or the
  /// copy already exists. Called single-threaded at a radius query's entry, before any fan-out.
  void ensureLeafSoa(math::Pool pool = {}) const {
    if (m_dim < kSoaLeafDimFloor || m_root == nullptr || !m_leafSoa.empty()) {
      return;
    }
    m_leafSoa.resize(m_points_reordered.size());
    // Leaves transpose disjoint slices, so the pass fans out over the contiguous node arena.
    const KDTreeNode *arenaNodes = m_root;
    pool.parallelForBlocks(std::size_t{0}, nodeCount(), std::size_t{0},
                           [&](std::size_t lo, std::size_t hi) {
                             for (std::size_t nodeSlot = lo; nodeSlot < hi; ++nodeSlot) {
                               const KDTreeNode &node = arenaNodes[nodeSlot];
                               if (node.m_left == nullptr && node.m_right == nullptr) {
                                 transposeLeafSoa(node);
                               }
                             }
                           });
  }

  /// Transpose one leaf's contiguous AoS rows into the feature-major @ref m_leafSoa block at
  /// the same base offset, so the SoA radius scan reads a feature's value for the leaf's points
  /// from one contiguous span.
  void transposeLeafSoa(const KDTreeNode &leaf) const noexcept {
    const std::size_t base = leaf.m_index;
    const std::size_t count = leaf.m_dim;
    const T *aos = m_points_reordered.data() + (base * m_dim);
    T *soa = m_leafSoa.data() + (base * m_dim);
    for (std::size_t p = 0; p < count; ++p) {
      for (std::size_t f = 0; f < m_dim; ++f) {
        soa[(f * count) + p] = aos[(p * m_dim) + f];
      }
    }
  }

  /// Leaf box: seed from the first stored row, then fold the remaining rows into min/max.
  void populateLeafBounds(const KDTreeNode &node) noexcept {
    T *minOut = m_nodeBounds.data() + (static_cast<std::size_t>(node.m_id) * 2 * m_dim);
    T *maxOut = minOut + m_dim;
    const std::size_t base = node.m_index;
    const std::size_t count = node.m_dim;
    const T *leafPts = m_points_reordered.data() + (base * m_dim);
    for (std::size_t j = 0; j < m_dim; ++j) {
      minOut[j] = leafPts[j];
      maxOut[j] = leafPts[j];
    }
    for (std::size_t i = 1; i < count; ++i) {
      const T *row = leafPts + (i * m_dim);
      for (std::size_t j = 0; j < m_dim; ++j) {
        if (row[j] < minOut[j]) {
          minOut[j] = row[j];
        }
        if (row[j] > maxOut[j]) {
          maxOut[j] = row[j];
        }
      }
    }
  }

  /// Internal box: union of the children's boxes plus the pivot row. Requires both child
  /// boxes to be populated already; the ctor's ascending-id pass guarantees it because a
  /// parent's post-order id exceeds every descendant's.
  void unionChildBounds(const KDTreeNode &node) noexcept {
    T *minOut = m_nodeBounds.data() + (static_cast<std::size_t>(node.m_id) * 2 * m_dim);
    T *maxOut = minOut + m_dim;

    const KDTreeNode *const seed = (node.m_left != nullptr) ? node.m_left : node.m_right;
    const T *seedMin = m_nodeBounds.data() + (static_cast<std::size_t>(seed->m_id) * 2 * m_dim);
    const T *seedMax = seedMin + m_dim;
    for (std::size_t j = 0; j < m_dim; ++j) {
      minOut[j] = seedMin[j];
      maxOut[j] = seedMax[j];
    }

    if (node.m_left != nullptr && node.m_right != nullptr) {
      const T *otherMin =
          m_nodeBounds.data() + (static_cast<std::size_t>(node.m_right->m_id) * 2 * m_dim);
      const T *otherMax = otherMin + m_dim;
      for (std::size_t j = 0; j < m_dim; ++j) {
        if (otherMin[j] < minOut[j]) {
          minOut[j] = otherMin[j];
        }
        if (otherMax[j] > maxOut[j]) {
          maxOut[j] = otherMax[j];
        }
      }
    }

    // Union-in the pivot's own row so the internal-node box encloses the pivot point too.
    const std::size_t pivotSlot = node.m_index;
    const T *pivotRow = m_points_reordered.data() + (pivotSlot * m_dim);
    for (std::size_t j = 0; j < m_dim; ++j) {
      if (pivotRow[j] < minOut[j]) {
        minOut[j] = pivotRow[j];
      }
      if (pivotRow[j] > maxOut[j]) {
        maxOut[j] = pivotRow[j];
      }
    }
  }

  /// Initial capacity for the per-chunk traversal stack. Tree depth stays below
  /// `log2(n / LeafSize)` in the balanced case; 64 slots absorb the worst-case spillover
  /// from unbalanced subtrees so the vector almost never grows past this reserve.
  static constexpr std::size_t kDefaultStackReserve = 64;

  /// Per-row adjacency capacity floor seeded before a radius query fills a row, sized to absorb a
  /// typical eps-neighbourhood so early survivors skip the vector-doubling reallocation cascade.
  static constexpr std::size_t kAdjReserveFloor = 8;
  static constexpr std::size_t kWideAdjReserveDim = 8;
  static constexpr std::size_t kWideAdjReserveFloor = 128;

  /// Feature-count floor at or above which the leaf scan uses the SoA copy. At `d=1` there is one
  /// feature and the scalar scan already does no horizontal work; from `d=2` the feature-major
  /// scan accumulates eight points in lanes and outruns the AoS scan, so the transpose pays.
  static constexpr std::size_t kSoaLeafDimFloor = 2;

  AllocT m_allocator; ///< Bump-allocator for @ref KDTreeNode instances owned by the tree.

  KDTreeNode *m_root = nullptr;       ///< Root of the tree; @c nullptr for an empty point set.
  const NDArray<T, 2> &m_points;      ///< Borrowed view of the caller-owned point cloud.
  std::size_t m_dim = 0;              ///< Cached dimension count (`m_points.dim(1)`).
  std::vector<std::size_t> m_indices; ///< Permutation: `m_indices[k]` is the original point
                                      ///< index of the point at reordered slot @c k.
  std::vector<T> m_points_reordered;  ///< Points in tree-build order; row @c k is
                                      ///< `m_points[m_indices[k]`]. Makes each leaf's
                                      ///< coordinates a contiguous @c count x d block.
  /// Feature-major copy of each leaf's points; the block at base @c b holds feature @c f for the
  /// leaf's point @c p at `b * d + f * count + p`. Built lazily by @ref ensureLeafSoa on the first
  /// radius query so a kNN-only tree never pays for it; empty below @ref kSoaLeafDimFloor.
  mutable std::vector<T> m_leafSoa;
  /// Monotonic node identifier assigned at construction; equals the final node count once the
  /// tree is fully constructed. Keys into @ref m_nodeBounds.
  std::uint32_t m_nextNodeId = 0;
  /// Flat per-node bounds buffer; row @c 2*id holds min-coords, row @c 2*id + 1 holds max-
  /// coords, each of length @c m_dim. Populated after the tree is built: leaf boxes fan out
  /// over the pool; internal boxes union up in ascending-id order.
  std::vector<T> m_nodeBounds;
};

} // namespace clustering
