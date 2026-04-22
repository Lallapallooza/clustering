#pragma once

#ifdef CLUSTERING_USE_AVX2

#include <immintrin.h>

#include <cstddef>

#include "clustering/math/pairwise.h"

namespace clustering::math::detail {

/**
 * @brief Squared point-to-AABB gap distance for f32 via AVX2.
 *
 * Per dimension the scalar reference is @c gap = max(0, max(boxMin - p, p - boxMax)) and the
 * return is @c sum(gap^2). The vector form computes the two signed deltas in lane-parallel,
 * folds them into a single non-negative gap with a pair of @c max instructions, and accumulates
 * @c gap*gap into an FMA-driven 8-lane accumulator. The branch-free fold replaces the scalar
 * if/else-if chain, which was the dominant per-dimension cost in the kdtree-traversal hot loop.
 *
 * Processes @c d in @c ceil(d/8) ymm-wide tiles plus a scalar tail (where @c d % 8 != 0). The
 * tail loop matches the scalar reference exactly so dimensions entirely below 8 (e.g. @c d=2 or
 * @c d=4) get the same answer the scalar fallback would give. Loads are unaligned
 * (@c _mm256_loadu_ps); the kdtree's reordered points and node-bounds buffers are not aligned
 * to 32 bytes by construction so an aligned load would be UB on the existing call sites.
 *
 * @param point  Length-@p d query coordinates.
 * @param boxMin Length-@p d AABB minimum coordinates.
 * @param boxMax Length-@p d AABB maximum coordinates.
 * @param d      Number of dimensions.
 * @return @c sum_{j=0..d-1} max(0, max(boxMin[j]-point[j], point[j]-boxMax[j]))^2.
 */
[[nodiscard]] inline float pointAabbGapSqAvx2F32(const float *__restrict__ point,
                                                 const float *__restrict__ boxMin,
                                                 const float *__restrict__ boxMax,
                                                 std::size_t d) noexcept {
  __m256 acc = _mm256_setzero_ps();
  const __m256 zero = _mm256_setzero_ps();

  std::size_t j = 0;
  for (; j + 8 <= d; j += 8) {
    const __m256 vp = _mm256_loadu_ps(point + j);
    const __m256 vmin = _mm256_loadu_ps(boxMin + j);
    const __m256 vmax = _mm256_loadu_ps(boxMax + j);
    const __m256 lo = _mm256_sub_ps(vmin, vp); // boxMin - point: positive when point below box
    const __m256 hi = _mm256_sub_ps(vp, vmax); // point - boxMax: positive when point above box
    // At most one of lo/hi is positive in any lane (boxMin <= boxMax is the AABB invariant); the
    // other is non-positive, and both are non-positive when the point lies inside the extent.
    // max(lo, hi) keeps the positive side (or a non-positive value if inside); max(0, ...)
    // clamps the inside case to zero.
    const __m256 gap = _mm256_max_ps(zero, _mm256_max_ps(lo, hi));
    acc = _mm256_fmadd_ps(gap, gap, acc);
  }

  float tail = 0.0F;
  for (; j < d; ++j) {
    float gap = 0.0F;
    if (point[j] < boxMin[j]) {
      gap = boxMin[j] - point[j];
    } else if (point[j] > boxMax[j]) {
      gap = point[j] - boxMax[j];
    }
    tail += gap * gap;
  }

  return horizontalSumAvx2(acc) + tail;
}

} // namespace clustering::math::detail

#endif // CLUSTERING_USE_AVX2
