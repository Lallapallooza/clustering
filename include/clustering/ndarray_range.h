#pragma once

#include <cstddef>
#include <limits>

namespace clustering {

/**
 * @brief Half-open index range with optional positive step for slicing an NDArray axis.
 *
 * @c end defaults to the axis-size sentinel; at slice time it is clamped to @c m_shape[axis].
 * Negative @c step is rejected (no reversed views in v1).
 */
struct Range {
  std::size_t begin = 0;
  std::size_t end = std::numeric_limits<std::size_t>::max();
  std::ptrdiff_t step = 1;
};

/**
 * @brief Sentinel value meaning "take every element of this axis".
 */
constexpr Range all() { return {}; }

} // namespace clustering
