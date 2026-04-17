#pragma once

#include <cstddef>

#ifdef CLUSTERING_USE_AVX2
#include <immintrin.h>

namespace clustering::math::distance::detail {

/**
 * @brief Squared Euclidean distance between two f32 vectors via AVX2.
 *
 * Processes 8 lanes per iteration with @c _mm256_load_ps, accumulates lane-parallel squared
 * differences into one @c __m256, and finishes any remainder in a scalar tail. The horizontal
 * reduction uses @c permute2f128 + two @c hadd_ps, matching the epilogue in @c index/kdtree.h so
 * the summation order is preserved for downstream bit-identity callers.
 *
 * Precondition: @p a and @p b must be 32-byte aligned. @c _mm256_load_ps on a misaligned pointer
 * is undefined behavior. The intended caller path is a pointer lifted from
 * @c NDArray::alignedData<32>() on an Owned / contiguous array; the public @c pointwiseSq entry
 * runtime-checks both operands before dispatching here so strided or externally-borrowed data
 * cannot reach this kernel through that surface.
 *
 * @param a First vector (32-byte aligned, @p n elements).
 * @param b Second vector (32-byte aligned, @p n elements).
 * @param n Number of elements in @p a and @p b.
 * @return @c sum_{i=0..n-1} (a[i] - b[i])^2 as @c float.
 */
inline float sqEuclideanAvx2F32(const float *__restrict__ a, const float *__restrict__ b,
                                std::size_t n) noexcept {
  __m256 sum_vec = _mm256_setzero_ps();

  std::size_t i = 0;
  for (; i + 8 <= n; i += 8) {
    const __m256 v1 = _mm256_load_ps(a + i);
    const __m256 v2 = _mm256_load_ps(b + i);
    const __m256 diff = _mm256_sub_ps(v1, v2);
    const __m256 sq_diff = _mm256_mul_ps(diff, diff);
    sum_vec = _mm256_add_ps(sum_vec, sq_diff);
  }

  float residual_sum = 0.0F;
  for (; i < n; ++i) {
    const float diff = a[i] - b[i];
    residual_sum += diff * diff;
  }

  const __m256 permute = _mm256_permute2f128_ps(sum_vec, sum_vec, 1);
  sum_vec = _mm256_add_ps(sum_vec, permute);
  sum_vec = _mm256_hadd_ps(sum_vec, sum_vec);
  sum_vec = _mm256_hadd_ps(sum_vec, sum_vec);

  const float simd_sum = _mm_cvtss_f32(_mm256_castps256_ps128(sum_vec));
  return simd_sum + residual_sum;
}

} // namespace clustering::math::distance::detail

#endif // CLUSTERING_USE_AVX2
