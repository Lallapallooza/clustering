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
 * @brief Dot product of two contiguous rows of equal length @p d, 32-byte-aligned inputs only.
 *
 * Uses @c _mm256_load_ps / @c _mm256_load_pd unconditionally. Passing a pointer that is not
 * 32-byte aligned is undefined behaviour and on x86 faults. Callers that have proved row
 * alignment once at the matrix level (base pointer aligned AND row stride a multiple of 32
 * bytes) can amortise the check across all row pairs by calling this variant; otherwise use
 * @ref dotRowPtr, whose per-operand alignment check is required for correctness.
 */
template <class T>
[[nodiscard]] inline T dotRowAligned32Ptr(const T *a, const T *b, std::size_t d) noexcept {
#ifdef CLUSTERING_USE_AVX2
  if constexpr (std::is_same_v<T, float>) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    std::size_t k = 0;
    for (; k + 16 <= d; k += 16) {
      const __m256 va0 = _mm256_load_ps(a + k);
      const __m256 vb0 = _mm256_load_ps(b + k);
      const __m256 va1 = _mm256_load_ps(a + k + 8);
      const __m256 vb1 = _mm256_load_ps(b + k + 8);
      acc0 = _mm256_fmadd_ps(va0, vb0, acc0);
      acc1 = _mm256_fmadd_ps(va1, vb1, acc1);
    }
    for (; k + 8 <= d; k += 8) {
      const __m256 va = _mm256_load_ps(a + k);
      const __m256 vb = _mm256_load_ps(b + k);
      acc0 = _mm256_fmadd_ps(va, vb, acc0);
    }
    float tail = 0.0F;
    for (; k < d; ++k) {
      tail += a[k] * b[k];
    }
    return horizontalSumAvx2(_mm256_add_ps(acc0, acc1)) + tail;
  } else if constexpr (std::is_same_v<T, double>) {
    __m256d acc0 = _mm256_setzero_pd();
    __m256d acc1 = _mm256_setzero_pd();
    std::size_t k = 0;
    for (; k + 8 <= d; k += 8) {
      const __m256d va0 = _mm256_load_pd(a + k);
      const __m256d vb0 = _mm256_load_pd(b + k);
      const __m256d va1 = _mm256_load_pd(a + k + 4);
      const __m256d vb1 = _mm256_load_pd(b + k + 4);
      acc0 = _mm256_fmadd_pd(va0, vb0, acc0);
      acc1 = _mm256_fmadd_pd(va1, vb1, acc1);
    }
    for (; k + 4 <= d; k += 4) {
      const __m256d va = _mm256_load_pd(a + k);
      const __m256d vb = _mm256_load_pd(b + k);
      acc0 = _mm256_fmadd_pd(va, vb, acc0);
    }
    double tail = 0.0;
    for (; k < d; ++k) {
      tail += a[k] * b[k];
    }
    return horizontalSumAvx2(_mm256_add_pd(acc0, acc1)) + tail;
  }
#endif
  T s = T{0};
  for (std::size_t t = 0; t < d; ++t) {
    s += a[t] * b[t];
  }
  return s;
}

/**
 * @brief Dot product of two contiguous rows of equal length @p d, any alignment.
 *
 * Checks each operand's alignment once and picks @c _mm256_load_ps vs @c _mm256_loadu_ps per
 * load accordingly; the branch is stable across the loop and predicts perfectly. Use this
 * variant when either operand's alignment cannot be proved at the call site (caller-provided
 * buffers, odd row strides, NumPy-owned storage). When both operands are guaranteed 32-byte
 * aligned at every call, @ref dotRowAligned32Ptr skips the check.
 */
template <class T>
[[nodiscard]] inline T dotRowPtr(const T *a, const T *b, std::size_t d) noexcept {
#ifdef CLUSTERING_USE_AVX2
  if constexpr (std::is_same_v<T, float>) {
    const bool aAligned = (reinterpret_cast<std::uintptr_t>(a) % 32) == 0;
    const bool bAligned = (reinterpret_cast<std::uintptr_t>(b) % 32) == 0;
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    std::size_t k = 0;
    for (; k + 16 <= d; k += 16) {
      const __m256 va0 = aAligned ? _mm256_load_ps(a + k) : _mm256_loadu_ps(a + k);
      const __m256 vb0 = bAligned ? _mm256_load_ps(b + k) : _mm256_loadu_ps(b + k);
      const __m256 va1 = aAligned ? _mm256_load_ps(a + k + 8) : _mm256_loadu_ps(a + k + 8);
      const __m256 vb1 = bAligned ? _mm256_load_ps(b + k + 8) : _mm256_loadu_ps(b + k + 8);
      acc0 = _mm256_fmadd_ps(va0, vb0, acc0);
      acc1 = _mm256_fmadd_ps(va1, vb1, acc1);
    }
    for (; k + 8 <= d; k += 8) {
      const __m256 va = aAligned ? _mm256_load_ps(a + k) : _mm256_loadu_ps(a + k);
      const __m256 vb = bAligned ? _mm256_load_ps(b + k) : _mm256_loadu_ps(b + k);
      acc0 = _mm256_fmadd_ps(va, vb, acc0);
    }
    float tail = 0.0F;
    for (; k < d; ++k) {
      tail += a[k] * b[k];
    }
    return horizontalSumAvx2(_mm256_add_ps(acc0, acc1)) + tail;
  } else if constexpr (std::is_same_v<T, double>) {
    const bool aAligned = (reinterpret_cast<std::uintptr_t>(a) % 32) == 0;
    const bool bAligned = (reinterpret_cast<std::uintptr_t>(b) % 32) == 0;
    __m256d acc0 = _mm256_setzero_pd();
    __m256d acc1 = _mm256_setzero_pd();
    std::size_t k = 0;
    for (; k + 8 <= d; k += 8) {
      const __m256d va0 = aAligned ? _mm256_load_pd(a + k) : _mm256_loadu_pd(a + k);
      const __m256d vb0 = bAligned ? _mm256_load_pd(b + k) : _mm256_loadu_pd(b + k);
      const __m256d va1 = aAligned ? _mm256_load_pd(a + k + 4) : _mm256_loadu_pd(a + k + 4);
      const __m256d vb1 = bAligned ? _mm256_load_pd(b + k + 4) : _mm256_loadu_pd(b + k + 4);
      acc0 = _mm256_fmadd_pd(va0, vb0, acc0);
      acc1 = _mm256_fmadd_pd(va1, vb1, acc1);
    }
    for (; k + 4 <= d; k += 4) {
      const __m256d va = aAligned ? _mm256_load_pd(a + k) : _mm256_loadu_pd(a + k);
      const __m256d vb = bAligned ? _mm256_load_pd(b + k) : _mm256_loadu_pd(b + k);
      acc0 = _mm256_fmadd_pd(va, vb, acc0);
    }
    double tail = 0.0;
    for (; k < d; ++k) {
      tail += a[k] * b[k];
    }
    return horizontalSumAvx2(_mm256_add_pd(acc0, acc1)) + tail;
  }
#endif
  T s = T{0};
  for (std::size_t t = 0; t < d; ++t) {
    s += a[t] * b[t];
  }
  return s;
}

template <class T> [[nodiscard]] inline T sqEuclideanFromDot(T normA, T normB, T dot) noexcept {
  const T sq = normA + normB - (T{2} * dot);
  return sq > T{0} ? sq : T{0};
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
