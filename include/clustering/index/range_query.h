#pragma once

#include <concepts>
#include <cstdint>
#include <vector>

#include "clustering/ndarray.h"

namespace clustering::index {

/**
 * @brief Contract for spatial indices that answer radius queries over a point cloud.
 *
 * A @c RangeQuery is constructed from a borrowed point matrix and exposes two radius-query
 * overloads that return the indices of points lying inside a ball of the given radius. The
 * two-argument overload returns every point inside the ball; the three-argument overload returns
 * at most @p limit of them.
 *
 * @tparam Q Candidate index type.
 * @tparam T Element type of the point cloud.
 */
template <class Q, class T>
concept RangeQuery = requires(const Q &q, const NDArray<T, 2> &points, const NDArray<T, 1> &p, T r,
                              std::int64_t limit) {
  Q(points);
  { q.query(p, r, limit) } -> std::same_as<std::vector<std::size_t>>;
  { q.query(p, r) } -> std::same_as<std::vector<std::size_t>>;
};

} // namespace clustering::index
