#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "clustering/ndarray.h"

namespace clustering::hdbscan {

/**
 * @brief One edge of the minimum spanning tree of mutual-reachability distances.
 *
 * Endpoints are @c std::int32_t because the pipeline contract caps @c N at the signed 32-bit
 * range. The weight is the mutual-reachability distance between the two endpoints under the
 * configured @c minSamples.
 *
 * @tparam T Element type of the point cloud. Only @c float is supported; a @c double
 *         specialization is out of scope.
 */
template <class T> struct MstEdge {
  /// First endpoint of the MST edge (0-based vertex index).
  std::int32_t u = 0;
  /// Second endpoint of the MST edge (0-based vertex index).
  std::int32_t v = 0;
  /// Mutual-reachability distance between @c u and @c v.
  T weight = T{};
};

/**
 * @brief Frozen output contract of every MST backend.
 *
 * The MST boundary is the one axis of variation across backends; everything downstream
 * (single-linkage tree, condensed tree, cluster extraction, outlier scoring) is monomorphic and
 * reads from this shape. Fields default to well-defined empty values so a @c MstOutput produced
 * by the default constructor is already in a valid "no fit yet" state.
 *
 * @tparam T Element type of the point cloud.
 */
template <class T> struct MstOutput {
  /// The @c N - 1 MST edges, in insertion order.
  std::vector<MstEdge<T>> edges;
  /// Per-point core distance (length @c N; self-excluded kNN distance at @c minSamples).
  NDArray<T, 1> coreDistances{std::array<std::size_t, 1>{0}};
};

} // namespace clustering::hdbscan
