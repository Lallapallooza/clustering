#pragma once

#include <cstddef>
#include <type_traits>

#include "clustering/math/pairwise.h"

#ifdef CLUSTERING_USE_AVX2
#include <immintrin.h>
#endif

namespace clustering::math::detail {

/**
 * @brief Sum-reduce a contiguous @c float / @c double array via two independent AVX2
 *        accumulator chains.
 *
 * A single-accumulator scalar reduction walks the inputs at one add per cycle (latency-bound on
 * the running total). Two AVX2 chains halve the dependency height and the lane-parallel adds
 * give an `8x` (float) / `4x` (double) throughput boost over scalar.
 *
 * @tparam T Element type (@c float or @c double).
 * @param p Length-@p n contiguous input.
 * @param n Element count.
 * @return Sum of the inputs.
 */
template <class T>
[[gnu::always_inline]] inline T sumReduceAvx2(const T *p, std::size_t n) noexcept {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                "sumReduceAvx2 requires float or double");
#ifdef CLUSTERING_USE_AVX2
  if constexpr (std::is_same_v<T, float>) {
    __m256 ae = _mm256_setzero_ps();
    __m256 ao = _mm256_setzero_ps();
    std::size_t i = 0;
    for (; i + 16 <= n; i += 16) {
      ae = _mm256_add_ps(ae, _mm256_loadu_ps(p + i));
      ao = _mm256_add_ps(ao, _mm256_loadu_ps(p + i + 8));
    }
    if (i + 8 <= n) {
      ae = _mm256_add_ps(ae, _mm256_loadu_ps(p + i));
      i += 8;
    }
    T s = horizontalSumAvx2(_mm256_add_ps(ae, ao));
    for (; i < n; ++i) {
      s += p[i];
    }
    return s;
  } else {
    __m256d ae = _mm256_setzero_pd();
    __m256d ao = _mm256_setzero_pd();
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) {
      ae = _mm256_add_pd(ae, _mm256_loadu_pd(p + i));
      ao = _mm256_add_pd(ao, _mm256_loadu_pd(p + i + 4));
    }
    if (i + 4 <= n) {
      ae = _mm256_add_pd(ae, _mm256_loadu_pd(p + i));
      i += 4;
    }
    T s = horizontalSumAvx2(_mm256_add_pd(ae, ao));
    for (; i < n; ++i) {
      s += p[i];
    }
    return s;
  }
#else
  T s = T{0};
  for (std::size_t i = 0; i < n; ++i) {
    s += p[i];
  }
  return s;
#endif
}

/**
 * @brief In-place affine transform `p[i] = p[i] * a + b` via one FMA per AVX2 lane.
 *
 * @tparam T Element type (@c float or @c double).
 * @param p Length-@p n in-place buffer; both source and destination.
 * @param n Element count.
 * @param a Multiplicative scalar applied to every element.
 * @param b Additive scalar applied after the multiply.
 */
template <class T>
[[gnu::always_inline]] inline void affineInPlaceAvx2(T *p, std::size_t n, T a, T b) noexcept {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                "affineInPlaceAvx2 requires float or double");
#ifdef CLUSTERING_USE_AVX2
  if constexpr (std::is_same_v<T, float>) {
    const __m256 va = _mm256_set1_ps(a);
    const __m256 vb = _mm256_set1_ps(b);
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) {
      _mm256_storeu_ps(p + i, _mm256_fmadd_ps(_mm256_loadu_ps(p + i), va, vb));
    }
    for (; i < n; ++i) {
      p[i] = (p[i] * a) + b;
    }
    return;
  } else {
    const __m256d va = _mm256_set1_pd(a);
    const __m256d vb = _mm256_set1_pd(b);
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4) {
      _mm256_storeu_pd(p + i, _mm256_fmadd_pd(_mm256_loadu_pd(p + i), va, vb));
    }
    for (; i < n; ++i) {
      p[i] = (p[i] * a) + b;
    }
    return;
  }
#else
  for (std::size_t i = 0; i < n; ++i) {
    p[i] = (p[i] * a) + b;
  }
#endif
}

/**
 * @brief Broadcast-fill `out[i] = v` for `i` in `[0, n)` via AVX2 stores.
 *
 * @tparam T Element type (@c float or @c double).
 * @param out Destination buffer.
 * @param n Element count.
 * @param v Value to broadcast.
 */
template <class T>
[[gnu::always_inline]] inline void fillAvx2(T *out, std::size_t n, T v) noexcept {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                "fillAvx2 requires float or double");
#ifdef CLUSTERING_USE_AVX2
  if constexpr (std::is_same_v<T, float>) {
    const __m256 vv = _mm256_set1_ps(v);
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) {
      _mm256_storeu_ps(out + i, vv);
    }
    for (; i < n; ++i) {
      out[i] = v;
    }
    return;
  } else {
    const __m256d vv = _mm256_set1_pd(v);
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4) {
      _mm256_storeu_pd(out + i, vv);
    }
    for (; i < n; ++i) {
      out[i] = v;
    }
    return;
  }
#else
  for (std::size_t i = 0; i < n; ++i) {
    out[i] = v;
  }
#endif
}

/**
 * @brief Scaled copy `out[i] = src[i] * a` via AVX2.
 *
 * Source and destination may alias.
 *
 * @tparam T Element type (@c float or @c double).
 * @param src Source buffer of length @p n.
 * @param n Element count.
 * @param a Multiplicative scalar.
 * @param out Destination buffer of length @p n.
 */
template <class T>
[[gnu::always_inline]] inline void scaleAvx2(const T *src, std::size_t n, T a, T *out) noexcept {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                "scaleAvx2 requires float or double");
#ifdef CLUSTERING_USE_AVX2
  if constexpr (std::is_same_v<T, float>) {
    const __m256 va = _mm256_set1_ps(a);
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) {
      _mm256_storeu_ps(out + i, _mm256_mul_ps(_mm256_loadu_ps(src + i), va));
    }
    for (; i < n; ++i) {
      out[i] = src[i] * a;
    }
    return;
  } else {
    const __m256d va = _mm256_set1_pd(a);
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4) {
      _mm256_storeu_pd(out + i, _mm256_mul_pd(_mm256_loadu_pd(src + i), va));
    }
    for (; i < n; ++i) {
      out[i] = src[i] * a;
    }
    return;
  }
#else
  for (std::size_t i = 0; i < n; ++i) {
    out[i] = src[i] * a;
  }
#endif
}

} // namespace clustering::math::detail
