#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

#include "clustering/math/pairwise.h"
#include "clustering/math/thread.h"
#include "clustering/ndarray.h"

namespace clustering {

/**
 * @brief Range-index backend that builds the full eps-neighborhood adjacency in one fused
 *        pairwise sweep.
 *
 * At high dim, where tree pruning collapses, a blocked pairwise sweep with the eps-threshold
 * fused into the microkernel epilogue is the right primitive for DBSCAN: core-point detection
 * and cluster expansion both reduce to adjacency lookups, so one N*N sweep retires all the
 * pairwise compute the algorithm needs.
 *
 * @tparam T Element type of the point cloud (@c float or @c double).
 *
 * @warning The instance borrows the input matrix. The caller must keep @p points alive for the
 *          lifetime of the @c BruteForcePairwise.
 */
template <class T> class BruteForcePairwise {
public:
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                "BruteForcePairwise<T> requires T to be float or double");

  /**
   * @brief Constructs the backend over a borrowed point matrix.
   *
   * @param points Row-major @c n x @c d point matrix. Must outlive the instance.
   */
  explicit BruteForcePairwise(const NDArray<T, 2> &points) noexcept : m_points(points) {}

  /**
   * @brief Returns the full radius-neighborhood adjacency over the indexed point cloud.
   *
   * Emits surviving @c (i, j) pairs directly from the fused AVX2 threshold kernel; the outer
   * driver partitions X rows across @p pool so per-row pushes are race-free.
   *
   * @param radius Non-negative neighbourhood radius; comparison runs on the squared distance.
   * @param pool   Parallelism injection forwarded to the thresholded sweep.
   * @return Length-@c n vector where element @c i lists every @c j with
   *         @c ||x_i - x_j||^2 <= radius^2.
   */
  [[nodiscard]] std::vector<std::vector<std::int32_t>> query(T radius, math::Pool pool) const {
    const std::size_t n = m_points.dim(0);
    std::vector<std::vector<std::int32_t>> adj(n);
    if (n == 0) {
      return adj;
    }

    // Reserve a small floor per row so the first push_backs do not trigger the vector-doubling
    // reallocation cascade that otherwise dominates adjacency construction on dense fixtures.
    for (auto &v : adj) {
      v.reserve(16);
    }

    const T radiusSq = radius * radius;
    auto emit = [&adj](std::size_t row, std::size_t col) {
      adj[row].push_back(static_cast<std::int32_t>(col));
    };
    math::pairwiseSqEuclideanThresholded(m_points, m_points, radiusSq, pool, emit);
    return adj;
  }

private:
  const NDArray<T, 2> &m_points; ///< Borrowed point matrix; the caller owns the storage.
};

} // namespace clustering
