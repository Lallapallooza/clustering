#pragma once

#include <concepts>
#include <cstdint>
#include <vector>

#include "clustering/math/thread.h"
#include "clustering/ndarray.h"

namespace clustering::index {

/**
 * @brief Contract for spatial indexes that can surface the radius-neighborhood adjacency
 *        over a borrowed point cloud in one call.
 *
 * DBSCAN reduces core-point detection and cluster expansion to neighbor lookups on the radius
 * graph; any satisfying backend is free to build that graph however its geometry favours
 * (tree walk for low dim, blocked pairwise for high dim).
 *
 * @tparam Q Candidate index type.
 * @tparam T Element type of the point cloud.
 */
template <class Q, class T>
concept RangeIndex = std::constructible_from<Q, const NDArray<T, 2> &> &&
                     requires(const Q &q, T radius, math::Pool pool) {
                       {
                         q.query(radius, pool)
                       } -> std::same_as<std::vector<std::vector<std::int32_t>>>;
                     };

} // namespace clustering::index
