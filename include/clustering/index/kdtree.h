#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <stack>
#include <vector>

#include "clustering/math/distance.h"
#include "clustering/math/thread.h"
#include "clustering/memory/linear_alloc.h"
#include "clustering/ndarray.h"

namespace clustering {

/**
 * @brief Represents a node in a KDTree.
 *
 * @details
 * KDTreeNode contains essential information about a node in a KDTree.
 * For internal nodes: m_index is the point index, m_dim is the split dimension,
 * and m_left/m_right point to children.
 * For leaf nodes: m_index is the start offset into the shared indices array,
 * m_dim is the count of points in the leaf, and m_left/m_right are both nullptr.
 * A node is a leaf when m_left == nullptr && m_right == nullptr.
 * This struct is separate from the KDTree class to facilitate usage in different allocators.
 */
struct KDTreeNode {
  std::size_t m_index; ///< For internal: point index. For leaf: offset into m_indices.
  std::size_t m_dim;   ///< For internal: split dimension. For leaf: count of points.
  KDTreeNode *m_left;  ///< Pointer to the left child (nullptr for leaf nodes)
  KDTreeNode *m_right; ///< Pointer to the right child (nullptr for leaf nodes)
};

enum class KDTreeDistanceType : std::uint8_t { kEucledian };

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
  using value_type = T; ///< value_type is the type of the points in the KDTree

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
      : m_allocator(calculatePoolSize(points.dim(0))), m_points(points) {
    m_indices.resize(points.dim(0));
    std::iota(m_indices.begin(), m_indices.end(), 0);
    m_root = build(0, m_indices.size(), 0);
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
    query(m_root, query_point, radius_sq, indices, limit);
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

    const std::size_t d = m_points.dim(1);
    const T radius_sq = radius * radius;

    auto runRange = [&](std::size_t lo, std::size_t hi) {
      NDArray<T, 1> q({d});
      std::vector<std::size_t> hits;
      for (std::size_t i = lo; i < hi; ++i) {
        for (std::size_t k = 0; k < d; ++k) {
          q[k] = m_points[i][k];
        }
        hits.clear();
        query(m_root, q, radius_sq, hits, /*limit=*/-1);
        adj[i].reserve(hits.size());
        for (const std::size_t h : hits) {
          adj[i].push_back(static_cast<std::int32_t>(h));
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
    return adj;
  }

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
               .m_right = nullptr};
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
    *node = {.m_index = m_indices[median],
             .m_dim = dim,
             .m_left = build(start, median, depth + 1),
             .m_right = build(median + 1, end, depth + 1)};

    return node;
  }

  /**
   * @brief Searches the KDTree for points within a specified radius of a query point.
   *
   * This method implements an efficient search algorithm using a stack to manage traversal
   * of the KDTree nodes. It checks each node's distance to the query point and accumulates
   * the indices of points falling within the specified radius. The search is optimized by
   * pruning branches of the tree that cannot contain points within the search radius.
   *
   * @param root Root node of the KDTree or current node in recursive calls.
   * @param query_point The point around which the search is performed.
   * @param radius The radius within which to search for points.
   * @param indices Vector to store the indices of points found within the search radius.
   * @param limit Optional limit on the number of points to find. If -1, no limit is applied.
   */
  void query(KDTreeNode *root, const NDArray<T, 1> &query_point, T radius_sq,
             std::vector<std::size_t> &indices, int64_t limit = -1) const {
    if (root == nullptr) {
      return;
    }

    std::vector<KDTreeNode *> stack;
    stack.reserve(256);
    stack.push_back(root);

    while (!stack.empty()) {
      const KDTreeNode *node = stack.back();
      stack.pop_back();

      if (node == nullptr) {
        continue;
      }

      if (limit != -1 && indices.size() == static_cast<std::size_t>(limit)) {
        break;
      }

      // Leaf node: brute-force all points in the range
      if (node->m_left == nullptr && node->m_right == nullptr) {
        for (std::size_t i = 0; i < node->m_dim; ++i) {
          const std::size_t idx = m_indices[node->m_index + i];
          T dist_sq = distanceSquared(query_point, idx);
          if (dist_sq <= radius_sq) {
            indices.push_back(idx);
          }
          if (limit != -1 && indices.size() == static_cast<std::size_t>(limit)) {
            break;
          }
        }
        continue;
      }

      // Internal node: check the split point and traverse children
      T dist_sq = distanceSquared(query_point, node->m_index);
      if (dist_sq <= radius_sq) {
        indices.push_back(node->m_index);
      }

      T diff = query_point[node->m_dim] - m_points[node->m_index][node->m_dim];
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
   * @brief Calculates the distance between a query point and a point in a KDTree.
   *
   * Routes through @c clustering::math::distance::pointwiseSq; the CPO's @c SqEuclideanTag
   * overload owns the AVX2 / scalar dispatch, including the 8-lane floor and the 32-byte
   * alignment check on both operands.
   *
   * @param query_point A reference to an NDArray representing the query point.
   * @param index The index of the point in the KDTree to compare with the query point.
   * @return The squared Euclidean distance between the query point and the specified point.
   */
  T distanceSquared(const NDArray<T, 1> &query_point, std::size_t index) const noexcept {
    if constexpr (distanceType == KDTreeDistanceType::kEucledian) {
      return math::distance::pointwiseSq(math::distance::SqEuclideanTag{}, query_point,
                                         m_points.row(index));
    }
  }

  AllocT m_allocator; ///< Allocator for the KDTree

  KDTreeNode *m_root = nullptr;       ///< Root of the KDTree
  const NDArray<T, 2> &m_points;      ///< NDArray of points in the KDTree
  std::vector<std::size_t> m_indices; ///< Shared index array for build
};

} // namespace clustering
