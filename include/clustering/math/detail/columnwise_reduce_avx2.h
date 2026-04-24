#pragma once

#include <cstddef>
#include <type_traits>

#ifdef CLUSTERING_USE_AVX2
#include <immintrin.h>
#endif

namespace clustering::math::detail {

#ifdef CLUSTERING_USE_AVX2

/**
 * @brief Accumulate one row into per-column @c sum and @c sum-of-squares running totals (f32).
 *
 * Fused pattern used by variance-of-columns computations: each lane of @p colSum accumulates
 * @p row[t] and each lane of @p colSumSq accumulates @p row[t]^2 across all rows. Processes 8
 * features per iteration; scalar tail for @p d not a multiple of 8. Writes are unaligned so the
 * caller's column buffers may come from any @c NDArray allocation.
 */
inline void columnwiseAccumSumSqAvx2F32(const float *row, std::size_t d, float *colSum,
                                        float *colSumSq) noexcept {
  std::size_t t = 0;
  for (; t + 8 <= d; t += 8) {
    const __m256 v = _mm256_loadu_ps(row + t);
    __m256 s = _mm256_loadu_ps(colSum + t);
    __m256 sq = _mm256_loadu_ps(colSumSq + t);
    s = _mm256_add_ps(s, v);
    sq = _mm256_fmadd_ps(v, v, sq);
    _mm256_storeu_ps(colSum + t, s);
    _mm256_storeu_ps(colSumSq + t, sq);
  }
  for (; t < d; ++t) {
    const float v = row[t];
    colSum[t] += v;
    colSumSq[t] += v * v;
  }
}

/**
 * @brief Kahan-compensated per-column add of one row into running per-column totals (f32).
 *
 * Adds @p row element-wise into @p sumRow using Kahan compensation held in @p compRow; each lane
 * is independent so SIMD along the feature axis is bit-identical to the scalar loop per column.
 * The subtractions that recover the rounding residual (`(t - sum)` - y) are emitted as discrete
 * @c _mm256_sub_ps calls so the compiler cannot fuse them into an FMA and collapse the Kahan
 * correction. Tail handled with scalar loop when @p d is not a multiple of 8.
 */
inline void kahanAddRowAvx2F32(const float *row, std::size_t d, float *sumRow,
                               float *compRow) noexcept {
  std::size_t t = 0;
  for (; t + 8 <= d; t += 8) {
    const __m256 v = _mm256_loadu_ps(row + t);
    const __m256 sum = _mm256_loadu_ps(sumRow + t);
    const __m256 comp = _mm256_loadu_ps(compRow + t);
    const __m256 y = _mm256_sub_ps(v, comp);
    const __m256 tv = _mm256_add_ps(sum, y);
    const __m256 diff = _mm256_sub_ps(tv, sum);
    const __m256 compNew = _mm256_sub_ps(diff, y);
    _mm256_storeu_ps(sumRow + t, tv);
    _mm256_storeu_ps(compRow + t, compNew);
  }
  for (; t < d; ++t) {
    const float y = row[t] - compRow[t];
    const float tv = sumRow[t] + y;
    compRow[t] = (tv - sumRow[t]) - y;
    sumRow[t] = tv;
  }
}

#endif // CLUSTERING_USE_AVX2

/**
 * @brief Accumulate one row into per-column sum and sum-of-squares totals.
 *
 * Dispatches to the AVX2 f32 kernel when the build is AVX2-enabled, otherwise a scalar loop.
 */
template <class T>
inline void columnwiseAccumSumSq(const T *row, std::size_t d, T *colSum, T *colSumSq) noexcept {
#ifdef CLUSTERING_USE_AVX2
  if constexpr (std::is_same_v<T, float>) {
    columnwiseAccumSumSqAvx2F32(row, d, colSum, colSumSq);
    return;
  }
#endif
  for (std::size_t t = 0; t < d; ++t) {
    const T v = row[t];
    colSum[t] += v;
    colSumSq[t] += v * v;
  }
}

/**
 * @brief Kahan-compensated per-column add of one row into running totals.
 *
 * Dispatches to the AVX2 f32 kernel when the build is AVX2-enabled, otherwise a scalar loop.
 * SIMD and scalar paths produce bit-identical results because each column's Kahan state is
 * independent of the others.
 */
template <class T>
inline void kahanAddRow(const T *row, std::size_t d, T *sumRow, T *compRow) noexcept {
#ifdef CLUSTERING_USE_AVX2
  if constexpr (std::is_same_v<T, float>) {
    kahanAddRowAvx2F32(row, d, sumRow, compRow);
    return;
  }
#endif
  for (std::size_t t = 0; t < d; ++t) {
    const T y = row[t] - compRow[t];
    const T tv = sumRow[t] + y;
    compRow[t] = (tv - sumRow[t]) - y;
    sumRow[t] = tv;
  }
}

} // namespace clustering::math::detail
