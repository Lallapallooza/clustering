#pragma once

#include <cmath>
#include <cassert>
#include <numeric>
#include <algorithm>
#include <stack>

#include "clustering/ndarray.h"
#include "clustering/memory/linear_alloc.h"

#ifdef CLUSTERING_USE_AVX2
#include <immintrin.h>
#endif

/**
 * @brief Represents a node in a KDTree.
 *
 * @details
 * KDTreeNode contains essential information about a node in a KDTree,
 * including its index, the dimension it represents, and pointers to its
 * left and right children. This struct is separate from the KDTree class
 * to facilitate usage in different allocators.
 */
struct KDTreeNode {
  std::size_t m_index;  ///< Index of the point in the KDTree
  std::size_t m_dim;    ///< Dimension of the point
  KDTreeNode  *m_left;  ///< Pointer to the left child of the node
  KDTreeNode  *m_right; ///< Pointer to the right child of the node
};

enum class KDTreeDistanceType {
  kEucledian
};

/**
 * @brief Implements a KDTree data structure.
 *
 * KDTree is a space-partitioning data structure for organizing points in a K-dimensional space.
 * It is efficient in range-search and nearest neighbor search. This implementation contains
 * a vector of points and an allocator for KDTree nodes, with the root node representing the tree's starting point.
 *
 * @tparam T Data type of the points.
 * @tparam AllocT Allocator type for KDTree nodes, defaults to LinearAllocator<KDTreeNode>.
 *
 * @warning The KDTree does not manage the lifecycle of the points array. Ensure the array remains valid during KDTree's lifetime.
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
template<class T, KDTreeDistanceType distanceType = KDTreeDistanceType::kEucledian, class AllocT=LinearAllocator<KDTreeNode>>
class KDTree {
 public:
  using value_type = T; ///< value_type is the type of the points in the KDTree

 public:
  /**
   * @brief Constructs a KDTree using a given set of points.
   *
   * @details
   * Initializes the KDTree with a given array of points and prepares the allocator.
   * This constructor builds the KDTree by sorting the points and constructing nodes accordingly.
   *
   * @param points NDArray of points to build the KDTree.
   *
   * @note The points array must remain valid for the lifetime of the KDTree, as the tree does not manage the array's lifecycle.
   */
  KDTree(const NDArray<T, 2> &points)
    : m_allocator(calculatePoolSize(points.dim(0))), m_points(points), m_root(nullptr) {
    std::vector<std::size_t> indices(points.dim(0));
    std::iota(indices.begin(), indices.end(), 0);
    m_root = m_allocator.allocate();
    m_root = build(indices, 0, m_root);
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
  std::vector<std::size_t> query(const NDArray<T, 1> &query_point, T radius, int64_t limit = -1) const {
    std::vector<std::size_t> indices;
    query(m_root, query_point, radius, indices, limit);
    return indices;
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
   * @return Calculated pool size in bytes.
   */
  static size_t calculatePoolSize(size_t numPoints) {
    return (2 * numPoints - 1) * sizeof(KDTreeNode);
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
    if (!node) {
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
  KDTreeNode *build(std::vector<std::size_t> &indices, std::size_t depth, KDTreeNode *node) {
    if (indices.empty()) {
      return nullptr; // Base case for recursion.
    }

    std::size_t dim    = depth % m_points.dim(1); // Select dimension based on current depth.
    std::size_t median = (indices.size() - 1) / 2;

    // Partition the indices around the median element.
    std::nth_element(indices.begin(), indices.begin() + median, indices.end(),
                     [this, dim](std::size_t lhs, std::size_t rhs) {
                       return m_points[lhs][dim] < m_points[rhs][dim];
                     });

    // Construct the node with the median element.
    *node = {
      /*m_index=*/indices[median],
      /*m_dim=*/dim,
      /*m_left=*/m_allocator.allocate(),
      /*m_right=*/m_allocator.allocate()
    };

    // Recursively build the left and right subtrees.
    std::vector<std::size_t> left_indices(indices.begin(), indices.begin() + median);
    std::vector<std::size_t> right_indices(indices.begin() + median + 1, indices.end());

    node->m_left  = build(left_indices, depth + 1, node->m_left);
    node->m_right = build(right_indices, depth + 1, node->m_right);

    return node; // Return the constructed node.
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
  void query(KDTreeNode *root,
             const NDArray<T, 1> &query_point,
             T radius,
             std::vector<std::size_t> &indices,
             int64_t limit = -1) const {
    if (!root) {
      return;
    }

    std::vector<KDTreeNode *> stack;
    stack.reserve(16); // Pre-allocate memory. TODO: Maybe runtime-tuned?
    stack.push_back(root);

    while (!stack.empty()) {
      KDTreeNode *node = stack.back();
      stack.pop_back();

      if (!node)
        continue;

      if (limit != -1 && indices.size() == limit) {
        break; // Stop if the limit is reached.
      }

      T dist = distance(query_point, node->m_index);
      if (dist <= radius) {
        indices.push_back(node->m_index); // Include this point as it's within the radius.
      }

      T diff = query_point[node->m_dim] - m_points[node->m_index][node->m_dim];
      // Decide which child nodes to traverse next.
      if (diff < 0) {
        if (node->m_left)
          stack.push_back(node->m_left);
        if (diff * diff <= radius * radius && node->m_right)
          stack.push_back(node->m_right);
      } else {
        if (node->m_right)
          stack.push_back(node->m_right);
        if (diff * diff <= radius * radius && node->m_left)
          stack.push_back(node->m_left);
      }
    }
  }


  /**
   * @brief Calculates the distance between a query point and a point in a KDTree.
   *
   * This function serves as a wrapper to select the appropriate distance calculation
   * method (either EucledianDistanceScalar or EucledianDistanceAVX2) based on whether AVX2 instructions
   * are being used. If CLUSTERING_USE_AVX2 is defined, EucledianDistanceAVX2 is used; otherwise,
   * EucledianDistanceScalar is used.
   *
   * @tparam T The data type of the elements in the NDArray.
   * @param query_point A reference to an NDArray representing the query point.
   * @param index The index of the point in the KDTree to compare with the query point.
   * @return The Euclidean distance between the query point and the specified point in the KDTree.
   */
  inline T distance(const NDArray<T, 1> &query_point, std::size_t index) const noexcept {
    if constexpr (distanceType == KDTreeDistanceType::kEucledian) {
      #ifdef CLUSTERING_USE_AVX2
      return EucledianDistanceAVX2(query_point, index);
      #else
      return EucledianDistanceScalar(query_point, index);
      #endif
    }
  }

  /**
   * @brief Calculates the Euclidean distance between a query point and a point in a KDTree using scalar operations.
   *
   * This method is a fallback for environments where AVX2 instructions are not available.
   * It calculates the Euclidean distance by iterating through each dimension, computing
   * the difference between corresponding elements of the query point and the KDTree point,
   * and summing their squares.
   *
   * @tparam T The data type of the elements in the NDArray.
   * @param query_point A reference to an NDArray representing the query point.
   * @param index The index of the point in the KDTree to compare with the query point.
   * @return The Euclidean distance between the query point and the specified point in the KDTree.
   * @exception noexcept This function is marked noexcept, meaning it is not expected to throw exceptions.
   */
  T EucledianDistanceScalar(const NDArray<T, 1> &query_point, std::size_t index) const noexcept {
    T sum = 0.0f;

    for (std::size_t dim = 0; dim < m_points.dim(1); ++dim) {
      T diff = query_point[dim] - m_points[index][dim];
      sum += diff * diff;
    }
    return std::sqrt(sum);
  }

#ifdef CLUSTERING_USE_AVX2
  /**
   * @brief Calculates the Euclidean distance between a query point and a point in a KDTree using AVX2 vectorized operations.
   *
   * This method leverages AVX2 instructions to process multiple dimensions simultaneously
   * for faster computation. It is used when CLUSTERING_USE_AVX2 is defined. The method
   * calculates the distance by vectorized operations on blocks of 8 dimensions and
   * handles any remaining dimensions scalarly.
   *
   * @tparam T The data type of the elements in the NDArray.
   * @param query_point A reference to an NDArray representing the query point.
   * @param index The index of the point in the KDTree to compare with the query point.
   * @return The Euclidean distance between the query point and the specified point in the KDTree.
   * @exception noexcept This function is marked noexcept, meaning it is not expected to throw exceptions.
   */
  T EucledianDistanceAVX2(const NDArray<T, 1> &query_point, std::size_t index) const noexcept {
    // Initialize a vector to accumulate squared differences, using AVX2 for efficient computation.
    __m256 sum_vec = _mm256_setzero_ps();

    std::size_t dim;
    std::size_t point_dim = m_points.dim(1); // Determine the total number of dimensions.

    // Process dimensions in groups of 8 using AVX2 instructions for vectorized computation.
    for (dim = 0; dim + 8 <= point_dim; dim += 8) {
      // Load groups of 8 elements from the query point and KDTree point.
      __m256 v1 = _mm256_load_ps(query_point[dim].data());
      __m256 v2 = _mm256_load_ps(m_points[index][dim].data());

      // Compute and accumulate the squared differences between elements.
      __m256 diff    = _mm256_sub_ps(v1, v2);
      __m256 sq_diff = _mm256_mul_ps(diff, diff);
      sum_vec = _mm256_add_ps(sum_vec, sq_diff);
    }

    // Residual sum with loop unrolling
    T residual_sum = 0.0f;
    for (; dim < point_dim; ++dim) {
      T diff = query_point[dim] - m_points[index][dim];
      residual_sum += diff * diff;
    }

    // Efficient horizontal sum of AVX2 vector
    __m256 permute = _mm256_permute2f128_ps(sum_vec, sum_vec, 1);
    sum_vec = _mm256_add_ps(sum_vec, permute);
    sum_vec = _mm256_hadd_ps(sum_vec, sum_vec);
    sum_vec = _mm256_hadd_ps(sum_vec, sum_vec);

    T simd_sum = _mm_cvtss_f32(_mm256_castps256_ps128(sum_vec));
    return std::sqrt(simd_sum + residual_sum);
  }
#endif

 private:
  AllocT m_allocator;            ///< Allocator for the KDTree

  KDTreeNode          *m_root;   ///< Root of the KDTree
  const NDArray<T, 2> &m_points; ///< NDArray of points in the KDTree
};