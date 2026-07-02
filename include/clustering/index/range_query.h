#pragma once

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "clustering/math/thread.h"
#include "clustering/ndarray.h"

namespace clustering::index {

/**
 * @brief Radius-neighborhood adjacency with per-point core flags.
 *
 * `rows[i]` lists neighbours of point @c i within the query radius, always including @c i
 * itself and every neighbour @c j with `j > i`. Rows of core points may omit neighbours with
 * `j < i`: their degree already carries the core verdict, and the component build reads only
 * the `j > i` half. Rows of non-core points are complete in both directions so border
 * assignment can scan every adjacent core.
 *
 * `isCore[i]` is nonzero iff point @c i has at least @c minPts neighbours within the radius,
 * counting itself, measured on the full two-sided degree regardless of what @c rows stores.
 */
struct CoreAdjacency {
  std::vector<std::vector<std::int32_t>> rows; ///< Per-point neighbour lists.
  std::vector<std::uint8_t> isCore;            ///< Per-point core flag from the full degree.
};

/**
 * @brief Contract for spatial indexes that can surface the radius-neighborhood adjacency
 *        over a borrowed point cloud in one call.
 *
 * DBSCAN reduces core-point detection and cluster expansion to neighbor lookups on the radius
 * graph; any satisfying backend is free to build that graph however its geometry favours
 * (tree walk for low dim, blocked pairwise for high dim). Carrying the core threshold into
 * the query lets a backend skip materializing edge halves no consumer reads.
 *
 * @tparam Q Candidate index type.
 * @tparam T Element type of the point cloud.
 */
template <class Q, class T>
concept RangeIndex = std::constructible_from<Q, const NDArray<T, 2> &> &&
                     requires(const Q &q, T radius, std::size_t minPts, math::Pool pool) {
                       { q.query(radius, minPts, pool) } -> std::same_as<CoreAdjacency>;
                     };

} // namespace clustering::index
