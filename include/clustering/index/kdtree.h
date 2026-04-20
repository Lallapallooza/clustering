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
#include "clustering/math/detail/avx2_helpers.h"
#include "clustering/math/detail/bounded_max_heap.h"
#include "clustering/math/detail/radius_scan.h"
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
 * The leaf-vs-internal test is @c m_left == nullptr && m_right == nullptr. The struct is
 * declared outside @ref KDTree so the allocator can bump-allocate fixed-size slots without
 * seeing the tree's template parameters.
 */
struct KDTreeNode {
  /// Internal: pivot slot in the tree's reordered point buffer. Leaf: base offset into the
  /// same buffer (leaf points live at slots @c [m_index, m_index + m_dim)).
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
   *
   * @note The points array must remain valid for the lifetime of the KDTree, as the tree does not
   * manage the array's lifecycle.
   */
  KDTree(const NDArray<T, 2> &points)
      : m_allocator(calculatePoolSize(points.dim(0))), m_points(points), m_dim(points.dim(1)) {
    CLUSTERING_ALWAYS_ASSERT(points.isContiguous());
    const std::size_t n = points.dim(0);
    m_indices.resize(n);
    std::iota(m_indices.begin(), m_indices.end(), 0);
    m_root = build(0, n, 0);
    // Materialize points in tree-build order. After @ref build rewrites @c m_indices into a
    // permutation matching the tree layout, a leaf's points live at @c m_points_reordered slots
    // @c [leaf.m_index, leaf.m_index + leaf.m_dim). Contiguous access there replaces the
    // scatter-indirection @c m_points[m_indices[k]] had, which at low d was the cache-miss
    // ceiling: every leaf-brute-force iteration landed on a random row of @c m_points.
    m_points_reordered.resize(n * m_dim);
    const T *src = points.data();
    T *dst = m_points_reordered.data();
    for (std::size_t k = 0; k < n; ++k) {
      const std::size_t src_row = m_indices[k];
      const T *s = src + (src_row * m_dim);
      T *d = dst + (k * m_dim);
      for (std::size_t j = 0; j < m_dim; ++j) {
        d[j] = s[j];
      }
    }
    // Populate per-node axis-aligned bounding boxes once the tree is built and the reordered
    // points buffer is materialized. Layout is @c (numNodes * 2 * d) flat: row @c 2*id holds the
    // min-coords vector, row @c 2*id + 1 holds the max-coords vector. Dual-tree walkers consume
    // this through @ref nodeBounds as a pair of @c std::span views; leaving @ref KDTreeNode's
    // size unchanged past the monotonic @c m_id keeps leaf-scan cache behaviour stable at
    // high @c d.
    m_nodeBounds.assign(static_cast<std::size_t>(m_nextNodeId) * 2 * m_dim, T{});
    if (m_root != nullptr) {
      populateBounds(m_root);
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
   * the caller.
   *
   * @param radius Non-negative neighbourhood radius; comparison runs on the squared distance.
   * @param pool   Parallelism injection used to fan the outer row loop out across workers.
   * @return Length-@c n vector where element @c i lists every @c j with
   *         @c ||x_i - x_j||^2 <= radius^2.
   */
  [[nodiscard]] std::vector<std::vector<std::int32_t>> query(T radius, math::Pool pool) const {
    const std::size_t n = m_points.dim(0);
    std::vector<std::vector<std::int32_t>> adj(n);
    if (n == 0) {
      return adj;
    }

    const T radius_sq = radius * radius;

    auto runRange = [&](std::size_t lo, std::size_t hi) {
      // Reuse the traversal stack across every query in this chunk. Tree depth stays below
      // log2(n / LeafSize) + a few for spilled internal pushes, so kDefaultStackReserve rarely
      // grows past its initial capacity; clearing keeps the allocation alive between queries.
      std::vector<KDTreeNode *> stack;
      stack.reserve(kDefaultStackReserve);
      const T *sourceData = m_points.data();
      for (std::size_t i = lo; i < hi; ++i) {
        // @p i iterates the source ordering; @c sourceData + i*m_dim is the caller's row.
        const T *qp = sourceData + (i * m_dim);
        queryImpl(m_root, qp, radius_sq, adj[i], stack, /*limit=*/-1);
      }
    };

    if (pool.shouldParallelize(n, 4, 2) && pool.pool != nullptr) {
      pool.pool
          ->submit_blocks(std::size_t{0}, n,
                          [&](std::size_t lo, std::size_t hi) { runRange(lo, hi); })
          .wait();
    } else {
      runRange(0, n);
    }
    return adj;
  }

  /**
   * @brief Returns the k nearest neighbours of every indexed point, self-excluded.
   *
   * For every row @c i of the original point cloud, the method finds the @c k original-index
   * points minimizing squared Euclidean distance to row @c i (excluding @c i itself) and writes
   * the result as two parallel @c (n x k) arrays: @c indices[i][j] is the original index of the
   * @c j-th closest neighbour, @c sqDists[i][j] is its squared distance. Each row is sorted
   * ascending by @c sqDist. Ties in distance resolve on smaller neighbour index so results are
   * reproducible bit-for-bit across runs at matched input.
   *
   * Traversal is depth-first with the bounded max-heap's current worst retained distance serving
   * as the pruning bound (@c heap.top().first). Until the heap fills, the bound is
   * @c std::numeric_limits<T>::max() and pruning is inactive.
   *
   * @pre @c k >= 1 and @c k < n.
   *
   * @param k    Number of neighbours per point (self-excluded). The signed 32-bit argument mirrors
   *             the neighbour-index type carried through the @c (indices, sqDists) output; the
   *             preconditions clamp it to @c [1, n).
   * @param pool Parallelism injection for the outer per-point loop; follows the
   *             @c shouldParallelize policy the radius-query path uses.
   * @return Pair of two arrays: @c .first is an @c (n x k) @c std::int32_t array of neighbour
   *         indices; @c .second is an @c (n x k) @c T array of squared distances.
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
      // Reuse the traversal stack and heap across every query in this chunk; both are reset at
      // the head of each per-point walk.
      std::vector<KDTreeNode *> stack;
      stack.reserve(kDefaultStackReserve);
      math::detail::BoundedMaxHeap<T, std::int32_t> heap(kSz);
      const T *sourceData = m_points.data();
      std::int32_t *idxOut = indices.data();
      T *distOut = sqDists.data();
      for (std::size_t i = lo; i < hi; ++i) {
        const auto iOriginal = static_cast<std::int32_t>(i);
        const T *qp = sourceData + (i * m_dim);
        heap.clear();
        knnQueryImpl(m_root, qp, iOriginal, heap, stack);
        // Drain the heap into descending-distance order (root is largest), then reverse to get
        // ascending order. Drain slots are written to the tail of the row so the reverse step is
        // just index arithmetic.
        std::size_t slot = heap.size();
        while (slot > 0) {
          --slot;
          const auto &top = heap.top();
          idxOut[(i * kSz) + slot] = top.second;
          distOut[(i * kSz) + slot] = top.first;
          heap.pop();
        }
      }
    };

    if (pool.shouldParallelize(n, 4, 2) && pool.pool != nullptr) {
      pool.pool
          ->submit_blocks(std::size_t{0}, n,
                          [&](std::size_t lo, std::size_t hi) { runRange(lo, hi); })
          .wait();
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
   * @return Pair of spans @c (min, max), each of length @c d.
   */
  [[nodiscard]] std::pair<std::span<const T>, std::span<const T>>
  nodeBounds(const KDTreeNode *node) const noexcept {
    assert(node != nullptr && "KDTree::nodeBounds on null node");
    const std::size_t base = static_cast<std::size_t>(node->m_id) * 2 * m_dim;
    const T *bounds = m_nodeBounds.data() + base;
    return {std::span<const T>(bounds, m_dim), std::span<const T>(bounds + m_dim, m_dim)};
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

  /**
   * @brief Recursively builds the KDTree from a set of point indices.
   *
   * This method constructs the KDTree by selecting a median point at each level,
   * partitioning the set of points into two subsets, and then recursively building
   * left and right subtrees. The dimension for comparison at each level of the tree
   * is determined by the depth, ensuring that different dimensions are used at each
   * level of the tree for balancing.
   *
   * @param indices Indices of the points to be included in the current subtree.
   * @param depth Current depth in the tree, used to determine the comparison dimension.
   * @param node The current node being constructed.
   * @return A pointer to the constructed KDTreeNode.
   */
  KDTreeNode *build(std::size_t start, std::size_t end, std::size_t depth) {
    if (start >= end) {
      return nullptr;
    }

    // Leaf node: store range into m_indices, brute-force at query time
    if (end - start <= LeafSize) {
      KDTreeNode *node = m_allocator.allocate();
      *node = {.m_index = start,     // offset into m_indices
               .m_dim = end - start, // count of points
               .m_left = nullptr,
               .m_right = nullptr,
               .m_id = m_nextNodeId++};
      return node;
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

    KDTreeNode *node = m_allocator.allocate();
    *node = {.m_index = median, // reordered slot of the pivot
             .m_dim = dim,
             .m_left = build(start, median, depth + 1),
             .m_right = build(median + 1, end, depth + 1),
             .m_id = m_nextNodeId++};

    return node;
  }

  /**
   * @brief Core iterative range-search driver.
   *
   * Takes a contiguous @c d-element query pointer and a caller-owned traversal stack so the
   * adjacency sweep can amortize the stack buffer across every query in a chunk. @p indices is
   * an output buffer of any integral type the caller wants labels in (rank-1 adjacency wants
   * @c std::int32_t; the public one-shot wrapper wants @c std::size_t). @p limit preserves the
   * behaviour of the historical overload: @c -1 emits everything, otherwise stop once the
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
   * @param limit     @c -1 for unbounded; otherwise stop once @c indices.size() reaches it.
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
      // kernel when @c d matches a batched width (today @c f32, @c d == 2) and otherwise
      // falls back to the scalar @ref sqEuclideanRowPtr per-row primitive.
      if (node->m_left == nullptr && node->m_right == nullptr) {
        const std::size_t base = node->m_index;
        const std::size_t count = node->m_dim;
        const T *leafPts = reorderedBase + (base * m_dim);
        const bool isBounded = (limit != -1);
        const std::size_t cap =
            isBounded ? static_cast<std::size_t>(limit) : std::numeric_limits<std::size_t>::max();
        math::detail::radiusScan(qp, leafPts, count, m_dim, radius_sq, [&](std::size_t i) noexcept {
          if (indices.size() >= cap) {
            return;
          }
          indices.push_back(static_cast<OutIdx>(m_indices[base + i]));
        });
        if (isBounded && indices.size() >= cap) {
          break;
        }
        continue;
      }

      // Internal node: check the split point and traverse children. @c node->m_index is the
      // pivot's slot in @c m_points_reordered, so the pivot's row lives at
      // @c reorderedBase + slot * m_dim. The original point index (needed for @c indices)
      // comes from @c m_indices[slot].
      const std::size_t pivotSlot = node->m_index;
      const std::size_t splitDim = node->m_dim;
      const T *pivotRow = reorderedBase + (pivotSlot * m_dim);
      const T dist_sq = math::detail::sqEuclideanRowPtr(qp, pivotRow, m_dim);
      if (dist_sq <= radius_sq) {
        indices.push_back(static_cast<OutIdx>(m_indices[pivotSlot]));
      }

      const T pivotCoord = pivotRow[splitDim];
      const T diff = qp[splitDim] - pivotCoord;
      if (diff < 0) {
        if (node->m_left != nullptr) {
          stack.push_back(node->m_left);
        }
        if (diff * diff <= radius_sq && node->m_right != nullptr) {
          stack.push_back(node->m_right);
        }
      } else {
        if (node->m_right != nullptr) {
          stack.push_back(node->m_right);
        }
        if (diff * diff <= radius_sq && node->m_left != nullptr) {
          stack.push_back(node->m_left);
        }
      }
    }
  }

  /**
   * @brief Depth-first kNN walker with bounded-max-heap pruning.
   *
   * Pushes every candidate point into the heap, using the heap's largest-key entry as the
   * pruning bound. Until the heap fills (holds @c k entries), the bound is
   * @c std::numeric_limits<T>::max() and the walker accepts all subtrees; once full, subtrees
   * whose minimum possible distance along the split axis exceeds the bound are skipped.
   *
   * @param root       Root node of the subtree to search.
   * @param qp         Pointer to the contiguous query point; size-@c m_dim.
   * @param selfIndex  Original-index slot to exclude from the result (the query's own row).
   * @param heap       Caller-owned bounded max-heap; cleared by the public @ref knnQuery driver.
   * @param stack      Caller-owned traversal scratch; cleared at the head of each call, reused
   *                   across subsequent calls within a range chunk.
   */
  void knnQueryImpl(KDTreeNode *root, const T *qp, std::int32_t selfIndex,
                    math::detail::BoundedMaxHeap<T, std::int32_t> &heap,
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

      // Current pruning bound. Until the heap is full, accept everything; once full, the heap
      // root's squared distance is the worst retained and serves as the upper bound.
      const T bound =
          (heap.size() < heap.capacity()) ? std::numeric_limits<T>::max() : heap.top().first;

      if (node->m_left == nullptr && node->m_right == nullptr) {
        // Leaf: walk the contiguous count x d block and push each candidate into the heap. Self-
        // exclusion is by original index so a candidate is skipped iff it is the query row.
        const std::size_t base = node->m_index;
        const std::size_t count = node->m_dim;
        const T *leafPts = reorderedBase + (base * m_dim);
        for (std::size_t i = 0; i < count; ++i) {
          const auto pointIdx = static_cast<std::int32_t>(m_indices[base + i]);
          if (pointIdx == selfIndex) {
            continue;
          }
          const T dsq = math::detail::sqEuclideanRowPtr(qp, leafPts + (i * m_dim), m_dim);
          heap.push(dsq, pointIdx);
        }
        continue;
      }

      // Internal: test the pivot, then descend near child first so the bound tightens before the
      // far-child prune test. The pivot's pointwise distance competes with the heap directly.
      const std::size_t pivotSlot = node->m_index;
      const std::size_t splitDim = node->m_dim;
      const T *pivotRow = reorderedBase + (pivotSlot * m_dim);
      const auto pivotIdx = static_cast<std::int32_t>(m_indices[pivotSlot]);
      if (pivotIdx != selfIndex) {
        const T dist_sq = math::detail::sqEuclideanRowPtr(qp, pivotRow, m_dim);
        heap.push(dist_sq, pivotIdx);
      }

      const T pivotCoord = pivotRow[splitDim];
      const T diff = qp[splitDim] - pivotCoord;
      const T diffSq = diff * diff;
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

  /**
   * @brief Post-order walk that fills @ref m_nodeBounds for every node in @p node's subtree.
   *
   * Leaves seed their box from the contiguous points they store; internal nodes take the
   * element-wise min/max of their children's boxes and then union-in the pivot's row so the
   * reported box truly encloses every point routed through the node (including the pivot at
   * the internal node's own position).
   *
   * @param node Root of the subtree to populate; must be non-null.
   */
  void populateBounds(KDTreeNode *node) noexcept {
    T *minOut = m_nodeBounds.data() + (static_cast<std::size_t>(node->m_id) * 2 * m_dim);
    T *maxOut = minOut + m_dim;

    if (node->m_left == nullptr && node->m_right == nullptr) {
      // Leaf: seed from the first stored row, then fold the remaining rows into min/max.
      const std::size_t base = node->m_index;
      const std::size_t count = node->m_dim;
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
      return;
    }

    // Internal: recurse first so children's bounds are ready, then take their union. The tree
    // invariant guarantees every internal node has at least one child -- the build routine only
    // returns @c nullptr when @p start >= @p end, and the internal-node branch is only entered
    // when the range exceeds @c LeafSize, which is @c >= 1.
    if (node->m_left != nullptr) {
      populateBounds(node->m_left);
    }
    if (node->m_right != nullptr) {
      populateBounds(node->m_right);
    }

    const KDTreeNode *const seed = (node->m_left != nullptr) ? node->m_left : node->m_right;
    const T *seedMin = m_nodeBounds.data() + (static_cast<std::size_t>(seed->m_id) * 2 * m_dim);
    const T *seedMax = seedMin + m_dim;
    for (std::size_t j = 0; j < m_dim; ++j) {
      minOut[j] = seedMin[j];
      maxOut[j] = seedMax[j];
    }

    if (node->m_left != nullptr && node->m_right != nullptr) {
      const T *otherMin =
          m_nodeBounds.data() + (static_cast<std::size_t>(node->m_right->m_id) * 2 * m_dim);
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
    const std::size_t pivotSlot = node->m_index;
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
  /// @c log2(n / LeafSize) in the balanced case; 64 slots absorb the worst-case spillover
  /// from unbalanced subtrees so the vector almost never grows past this reserve.
  static constexpr std::size_t kDefaultStackReserve = 64;

  AllocT m_allocator; ///< Bump-allocator for @ref KDTreeNode instances owned by the tree.

  KDTreeNode *m_root = nullptr;       ///< Root of the tree; @c nullptr for an empty point set.
  const NDArray<T, 2> &m_points;      ///< Borrowed view of the caller-owned point cloud.
  std::size_t m_dim = 0;              ///< Cached dimension count (@c m_points.dim(1)).
  std::vector<std::size_t> m_indices; ///< Permutation: @c m_indices[k] is the original point
                                      ///< index of the point at reordered slot @c k.
  std::vector<T> m_points_reordered;  ///< Points in tree-build order; row @c k is
                                      ///< @c m_points[m_indices[k]]. Makes each leaf's
                                      ///< coordinates a contiguous @c count x d block.
  /// Monotonic node identifier assigned at @ref build; equals the final node count once the
  /// tree is fully constructed. Keys into @ref m_nodeBounds.
  std::uint32_t m_nextNodeId = 0;
  /// Flat per-node bounds buffer; row @c 2*id holds min-coords, row @c 2*id + 1 holds max-
  /// coords, each of length @c m_dim. Populated by @ref populateBounds after the tree is built.
  std::vector<T> m_nodeBounds;
};

} // namespace clustering
