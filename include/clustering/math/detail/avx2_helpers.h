#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "clustering/math/pairwise.h"

#ifdef CLUSTERING_USE_AVX2
#include <immintrin.h>
#endif

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

/**
 * @brief Linear existence test over a contiguous @c std::int32_t slab.
 *
 * Returns @c true if any of the @p size slots starting at @p data holds @p target. AVX2 path
 * consumes the slab 8 lanes per vector compare, so short dedup scans over small slabs collapse
 * from @c O(size) scalar compares to @c O(size / 8) vector ops. Scalar fallback when AVX2 is
 * not compiled in.
 *
 * @param data   Pointer to the slab; may be unaligned.
 * @param size   Slab length in @c std::int32_t elements.
 * @param target Value to locate.
 */
[[nodiscard]] inline bool containsInt32(const std::int32_t *data, std::size_t size,
                                        std::int32_t target) noexcept {
#ifdef CLUSTERING_USE_AVX2
  const __m256i needle = _mm256_set1_epi32(target);
  std::size_t s = 0;
  for (; s + 8 <= size; s += 8) {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    const __m256i haystack = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(data + s));
    const __m256i eq = _mm256_cmpeq_epi32(haystack, needle);
    if (_mm256_movemask_epi8(eq) != 0) {
      return true;
    }
  }
  for (; s < size; ++s) {
    if (data[s] == target) {
      return true;
    }
  }
  return false;
#else
  for (std::size_t s = 0; s < size; ++s) {
    if (data[s] == target) {
      return true;
    }
  }
  return false;
#endif
}

} // namespace clustering::math::detail
