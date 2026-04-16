#pragma once

#include <cstddef>
#include <type_traits>

#include "clustering/math/pairwise.h"

namespace clustering::math::detail {

/**
 * @brief Squared Euclidean distance between two contiguous rows of equal length @p d.
 *
 * Routes to the AVX2-vectorized @c sqEuclideanRowAvx2 when the build is AVX2-enabled and
 * @p d clears one lane width, otherwise a scalar fallback.
 */
template <class T>
[[nodiscard]] inline T sqEuclideanRowPtr(const T *a, const T *b, std::size_t d) noexcept {
#ifdef CLUSTERING_USE_AVX2
  if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
    if (d >= kAvx2Lanes<T>) {
      return sqEuclideanRowAvx2(a, b, d);
    }
  }
#endif
  T s = T{0};
  for (std::size_t t = 0; t < d; ++t) {
    const T diff = a[t] - b[t];
    s += diff * diff;
  }
  return s;
}

} // namespace clustering::math::detail
