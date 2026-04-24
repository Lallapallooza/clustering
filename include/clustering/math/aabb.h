#pragma once

#include <cstddef>
#include <span>
#include <type_traits>

#include "clustering/always_assert.h"
#ifdef CLUSTERING_USE_AVX2
#include "clustering/math/detail/aabb_gap_avx2.h"
#endif

namespace clustering::math {

/**
 * @brief Squared gap distance between a point and an axis-aligned bounding box.
 *
 * Per dimension, either the point lies inside the extent (zero gap) or the gap is the signed
 * distance to the nearer face. The return is the sum of squared gaps, a strict lower bound on
 * the squared Euclidean distance from the point to any point inside the box. Used as the prune
 * key in single-tree KDTree traversals.
 *
 * When @c CLUSTERING_USE_AVX2 is defined dispatches to the AVX2 kernel which processes @c d in
 * `ceil(d/8)` ymm-wide tiles plus a scalar tail. The vector path is branch-free per dimension;
 * the tail loop matches the scalar reference exactly so dimensions entirely below 8 (e.g. @c d=2
 * or @c d=4) get the same answer as the scalar fallback.
 *
 * Numerically equivalent to the scalar reference within `(ceil(d/8)+1)`*ULP-2 on finite inputs:
 * one ULP-2 budget per accumulated 8-wide partial plus one for the horizontal reduction. Strict
 * ULP-2 holds at `d<=8` (single tile, two roundings).
 *
 * @warning Bit-equivalence with the scalar reference is not guaranteed: the AVX2 horizontal
 *          reduction differs from a scalar left-fold under FMA. NaN inputs also diverge: the
 *          scalar predicate treats NaN as "inside" (gap = 0), the vector path propagates NaN
 *          through the @c max chain. Callers requiring NaN-safe behaviour must pre-filter.
 *
 * @tparam T Scalar element type; @c float only (a @c double specialization is out of scope).
 *
 * @param point  Length-@c d query coordinates.
 * @param boxMin AABB minimum coordinates; length defines @c d.
 * @param boxMax AABB maximum coordinates; length must equal `boxMin.size()`.
 * @return `sum_{j=0..d-1}` max(0, max(boxMin[j]-point[j], point[j]-boxMax[j]))^2.
 */
template <class T>
[[nodiscard]] inline T pointAabbGapSq(const T *point, std::span<const T> boxMin,
                                      std::span<const T> boxMax) noexcept {
  static_assert(std::is_same_v<T, float>,
                "pointAabbGapSq: T must be float; a double specialization is out of scope.");
  CLUSTERING_ALWAYS_ASSERT(boxMin.size() == boxMax.size());
  const std::size_t d = boxMin.size();
#ifdef CLUSTERING_USE_AVX2
  return detail::pointAabbGapSqAvx2F32(point, boxMin.data(), boxMax.data(), d);
#else
  T sum = T{0};
  for (std::size_t j = 0; j < d; ++j) {
    T gap = T{0};
    if (point[j] < boxMin[j]) {
      gap = boxMin[j] - point[j];
    } else if (point[j] > boxMax[j]) {
      gap = point[j] - boxMax[j];
    }
    sum += gap * gap;
  }
  return sum;
#endif
}

} // namespace clustering::math
