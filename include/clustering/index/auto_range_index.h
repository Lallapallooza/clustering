#pragma once

#include <cstddef>
#include <cstdint>
#include <variant>
#include <vector>

#include "clustering/index/brute_force_pairwise.h"
#include "clustering/index/kdtree.h"
#include "clustering/math/thread.h"
#include "clustering/ndarray.h"

namespace clustering::index {

/**
 * @brief Range-index policy that picks @ref KDTree below a dimension threshold and
 *        @ref BruteForcePairwise above it.
 *
 * Above @ref AutoRangeIndex::bruteForceDimFloor KDTree pruning collapses on the curse of
 * dimensionality and a blocked pairwise sweep wins; below it the tree's log-depth walks are cheaper
 * than an N*N sweep. The choice is made once at construction.
 *
 * @tparam T Element type of the point cloud.
 */
template <class T> class AutoRangeIndex {
public:
#ifdef CLUSTERING_DBSCAN_BRUTE_FORCE_DIM_FLOOR
  /**
   * @brief Dimension threshold at or above which the brute-force backend is selected.
   *
   * Override at configure time with @c -DCLUSTERING_DBSCAN_BRUTE_FORCE_DIM_FLOOR=<value>.
   */
  static constexpr std::size_t bruteForceDimFloor = CLUSTERING_DBSCAN_BRUTE_FORCE_DIM_FLOOR;
#else
  /// Dimension threshold at or above which the brute-force backend is selected.
  static constexpr std::size_t bruteForceDimFloor = 16;
#endif

  /**
   * @brief Constructs the policy, picking the backend once against `points.dim(1)`.
   *
   * @param points Row-major @c n x @c d point matrix. Must outlive the instance.
   * @param pool   Parallelism injection forwarded to the KDTree backend's build; the
   *               brute-force backend has no construction work to fan out.
   */
  explicit AutoRangeIndex(const NDArray<T, 2> &points, math::Pool pool = {})
      : m_held(pick(points, pool)) {}

  /**
   * @brief Returns the core-aware radius adjacency from the held backend.
   *
   * @param radius Non-negative neighbourhood radius; comparison runs on the squared distance.
   * @param minPts Core threshold on the self-inclusive neighbour count.
   * @param pool   Parallelism injection forwarded to the held backend.
   * @return Rows and core flags per the @ref clustering::index::CoreAdjacency contract.
   */
  [[nodiscard]] CoreAdjacency query(T radius, std::size_t minPts, math::Pool pool) const {
    return std::visit([&](const auto &idx) { return idx.query(radius, minPts, pool); }, m_held);
  }

private:
  using Tree = KDTree<T, KDTreeDistanceType::kEucledian, 64>;
  using Brute = BruteForcePairwise<T>;
  using Held = std::variant<Tree, Brute>;

  static Held pick(const NDArray<T, 2> &points, math::Pool pool) {
    if (points.dim(1) >= bruteForceDimFloor) {
      return Held(std::in_place_type<Brute>, points);
    }
    return Held(std::in_place_type<Tree>, points, pool);
  }

  Held m_held; ///< Concrete backend picked at construction.
};

} // namespace clustering::index
