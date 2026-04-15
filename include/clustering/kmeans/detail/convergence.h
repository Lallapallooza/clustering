#pragma once

#include <cstddef>
#include <type_traits>

#include "clustering/ndarray.h"

namespace clustering::kmeans::detail {

/**
 * @brief Kahan-compensated total squared shift across all centroids.
 *
 * @p shiftSq(c) holds the squared L2 shift of centroid @c c between iterations; this routine
 * returns their compensated sum. Kahan avoids the large-running-total + many-small-addends
 * cancellation that would bias the convergence check near-tol at large @c k.
 */
template <class T> [[nodiscard]] inline T totalShiftSqKahan(const NDArray<T, 1> &shiftSq) noexcept {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                "totalShiftSqKahan<T> requires T to be float or double");
  const std::size_t k = shiftSq.dim(0);
  T s = T{0};
  T c = T{0};
  for (std::size_t i = 0; i < k; ++i) {
    const T y = shiftSq(i) - c;
    const T t = s + y;
    c = (t - s) - y;
    s = t;
  }
  return s;
}

} // namespace clustering::kmeans::detail
