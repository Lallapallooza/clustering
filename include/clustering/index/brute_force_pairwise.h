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
   * Emits surviving `(i, j)` pairs directly from the fused AVX2 threshold kernel; the outer
   * driver partitions X rows across @p pool so per-row pushes are race-free.
   *
   * @param radius Non-negative neighbourhood radius; comparison runs on the squared distance.
   * @param pool   Parallelism injection forwarded to the thresholded sweep.
   * @return Length-@c n vector where element @c i lists every @c j with
   *         `||x_i - x_j||^2 <= radius^2`.
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
    // Symmetric eps-neighbour graph: the kernel emits each unique upper-triangular cell once
    // with `row <= col`. The kernel-side emit only touches `adj[row]` so workers writing
    // disjoint row chunks remain race-free; the mirror push (`adj[col]`.push(row)) runs as a
    // single-threaded post-pass below. Halves the pairwise compute on the brute-force path.
    auto emit = [&adj](std::size_t row, std::size_t col) {
      adj[row].push_back(static_cast<std::int32_t>(col));
    };
    math::pairwiseSqEuclideanThresholdedSymmetric(m_points, radiusSq, pool, emit);

    // Mirror pass: each surviving upper-triangular pair `(i, j)` lives in `adj[i]`; push
    // @c i to `adj[j]` sequentially so the parallel sweep above never crosses worker
    // boundaries. The size snapshot stops the loop from walking the freshly-mirrored entries
    // when the next outer @c i lands on a row that earlier iterations already augmented.
    for (std::size_t i = 0; i < n; ++i) {
      const std::size_t origSize = adj[i].size();
      for (std::size_t k = 0; k < origSize; ++k) {
        const auto jSigned = adj[i][k];
        const auto j = static_cast<std::size_t>(jSigned);
        if (j > i) {
          adj[j].push_back(static_cast<std::int32_t>(i));
        }
      }
    }
    return adj;
  }

private:
  const NDArray<T, 2> &m_points; ///< Borrowed point matrix; the caller owns the storage.
};

} // namespace clustering
