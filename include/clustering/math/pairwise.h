#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "clustering/always_assert.h"
#include "clustering/math/gemm.h"
#include "clustering/math/thread.h"
#include "clustering/ndarray.h"

#ifdef CLUSTERING_USE_AVX2
#include <immintrin.h>
#endif

namespace clustering::math {

namespace detail {

#ifdef CLUSTERING_USE_AVX2

inline float horizontalSumAvx2(__m256 v) noexcept {
  const __m256 permute = _mm256_permute2f128_ps(v, v, 1);
  const __m256 s1 = _mm256_add_ps(v, permute);
  const __m256 s2 = _mm256_hadd_ps(s1, s1);
  const __m256 s3 = _mm256_hadd_ps(s2, s2);
  return _mm_cvtss_f32(_mm256_castps256_ps128(s3));
}

inline double horizontalSumAvx2(__m256d v) noexcept {
  const __m256d permute = _mm256_permute2f128_pd(v, v, 1);
  const __m256d s1 = _mm256_add_pd(v, permute);
  const __m256d s2 = _mm256_hadd_pd(s1, s1);
  return _mm_cvtsd_f64(_mm256_castpd256_pd128(s2));
}

inline float sqEuclideanRowAvx2(const float *xRow, const float *yRow, std::size_t d) noexcept {
  __m256 acc = _mm256_setzero_ps();
  const bool xAligned = (reinterpret_cast<std::uintptr_t>(xRow) % 32) == 0;
  const bool yAligned = (reinterpret_cast<std::uintptr_t>(yRow) % 32) == 0;
  std::size_t k = 0;
  for (; k + 8 <= d; k += 8) {
    const __m256 vx = xAligned ? _mm256_load_ps(xRow + k) : _mm256_loadu_ps(xRow + k);
    const __m256 vy = yAligned ? _mm256_load_ps(yRow + k) : _mm256_loadu_ps(yRow + k);
    const __m256 diff = _mm256_sub_ps(vx, vy);
    acc = _mm256_add_ps(acc, _mm256_mul_ps(diff, diff));
  }
  float tail = 0.0F;
  for (; k < d; ++k) {
    const float diff = xRow[k] - yRow[k];
    tail += diff * diff;
  }
  return horizontalSumAvx2(acc) + tail;
}

inline double sqEuclideanRowAvx2(const double *xRow, const double *yRow, std::size_t d) noexcept {
  __m256d acc = _mm256_setzero_pd();
  const bool xAligned = (reinterpret_cast<std::uintptr_t>(xRow) % 32) == 0;
  const bool yAligned = (reinterpret_cast<std::uintptr_t>(yRow) % 32) == 0;
  std::size_t k = 0;
  for (; k + 4 <= d; k += 4) {
    const __m256d vx = xAligned ? _mm256_load_pd(xRow + k) : _mm256_loadu_pd(xRow + k);
    const __m256d vy = yAligned ? _mm256_load_pd(yRow + k) : _mm256_loadu_pd(yRow + k);
    const __m256d diff = _mm256_sub_pd(vx, vy);
    acc = _mm256_add_pd(acc, _mm256_mul_pd(diff, diff));
  }
  double tail = 0.0;
  for (; k < d; ++k) {
    const double diff = xRow[k] - yRow[k];
    tail += diff * diff;
  }
  return horizontalSumAvx2(acc) + tail;
}

#endif // CLUSTERING_USE_AVX2

template <class T> constexpr std::size_t kAvx2Lanes = std::is_same_v<T, float> ? 8 : 4;

template <class T, Layout LX, Layout LY>
inline T sqEuclideanRow(const NDArray<T, 2, LX> &X, std::size_t i, const NDArray<T, 2, LY> &Y,
                        std::size_t j) noexcept {
  const std::size_t d = X.dim(1);
#ifdef CLUSTERING_USE_AVX2
  if constexpr (LX == Layout::Contig && LY == Layout::Contig) {
    if (d >= kAvx2Lanes<T>) {
      const T *xRow = X.data() + (i * d);
      const T *yRow = Y.data() + (j * d);
      return sqEuclideanRowAvx2(xRow, yRow, d);
    }
  }
#endif
  T sum = T{0};
  for (std::size_t k = 0; k < d; ++k) {
    const T diff = X(i, k) - Y(j, k);
    sum += diff * diff;
  }
  return sum;
}

#ifdef CLUSTERING_USE_AVX2

inline float sqNormRowAvx2(const float *xRow, std::size_t d) noexcept {
  __m256 acc = _mm256_setzero_ps();
  const bool aligned = (reinterpret_cast<std::uintptr_t>(xRow) % 32) == 0;
  std::size_t k = 0;
  for (; k + 8 <= d; k += 8) {
    const __m256 v = aligned ? _mm256_load_ps(xRow + k) : _mm256_loadu_ps(xRow + k);
    acc = _mm256_add_ps(acc, _mm256_mul_ps(v, v));
  }
  float tail = 0.0F;
  for (; k < d; ++k) {
    tail += xRow[k] * xRow[k];
  }
  return horizontalSumAvx2(acc) + tail;
}

inline double sqNormRowAvx2(const double *xRow, std::size_t d) noexcept {
  __m256d acc = _mm256_setzero_pd();
  const bool aligned = (reinterpret_cast<std::uintptr_t>(xRow) % 32) == 0;
  std::size_t k = 0;
  for (; k + 4 <= d; k += 4) {
    const __m256d v = aligned ? _mm256_load_pd(xRow + k) : _mm256_loadu_pd(xRow + k);
    acc = _mm256_add_pd(acc, _mm256_mul_pd(v, v));
  }
  double tail = 0.0;
  for (; k < d; ++k) {
    tail += xRow[k] * xRow[k];
  }
  return horizontalSumAvx2(acc) + tail;
}

#endif // CLUSTERING_USE_AVX2

template <class T, Layout LX>
inline T sqNormRow(const NDArray<T, 2, LX> &X, std::size_t i) noexcept {
  const std::size_t d = X.dim(1);
#ifdef CLUSTERING_USE_AVX2
  if constexpr (LX == Layout::Contig) {
    if (d >= kAvx2Lanes<T>) {
      const T *xRow = X.data() + (i * d);
      return sqNormRowAvx2(xRow, d);
    }
  }
#endif
  T sum = T{0};
  for (std::size_t k = 0; k < d; ++k) {
    const T v = X(i, k);
    sum += v * v;
  }
  return sum;
}

/**
 * @brief Row-wise sum of squares: @c norms(i) = sum_k X(i, k)^2.
 *
 * Inner reduction mirrors @ref sqEuclideanRow: AVX2 when @p LX is @c Layout::Contig and the row
 * fills at least one lane, scalar otherwise. The outer row loop fans out over @p pool when the
 * workload clears @c shouldParallelize; per-row arithmetic is untouched across the fan-out.
 *
 * @tparam T Element type (@c float or @c double).
 * @tparam LX Layout tag of @p X; CTAD-resolved so strided views (e.g. @c Z.t()) bind without
 *         explicit template arguments.
 * @param X Input matrix (n x d).
 * @param norms Rank-1 output of length n; @c isMutable() must be true.
 * @param pool Parallelism injection for the outer row loop.
 */
template <class T, Layout LX>
void rowNormsSq(const NDArray<T, 2, LX> &X, NDArray<T, 1> &norms, Pool pool) {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                "rowNormsSq<T> requires T to be float or double");

  CLUSTERING_ALWAYS_ASSERT(norms.isMutable());
  CLUSTERING_ALWAYS_ASSERT(norms.dim(0) == X.dim(0));

  const std::size_t n = X.dim(0);
  if (n == 0) {
    return;
  }

  auto runRowRange = [&](std::size_t lo, std::size_t hi) noexcept {
    for (std::size_t i = lo; i < hi; ++i) {
      norms(i) = sqNormRow<T, LX>(X, i);
    }
  };

  if (pool.shouldParallelize(n, 4, 2)) {
    pool.pool
        ->submit_blocks(std::size_t{0}, n,
                        [&](std::size_t lo, std::size_t hi) { runRowRange(lo, hi); })
        .wait();
  } else {
    runRowRange(0, n);
  }
}

/**
 * @brief Large-path pairwise squared Euclidean via the GEMM identity.
 *
 * Computes @c out(i, j) = ||X(i) - Y(j)||^2 by reconstructing
 * @c ||x||^2 + ||y||^2 - 2 x . y^T. The dot-product matrix is evaluated with the public
 * @ref gemm entry at @c alpha = -2 and @c beta = 0; row-norm vectors are produced by
 * @ref rowNormsSq; a final elementwise broadcast adds the norms and clamps to zero.
 *
 * @tparam T Element type (@c float or @c double).
 * @tparam LX Layout tag of @p X; CTAD-resolved.
 * @tparam LY Layout tag of @p Y; CTAD-resolved.
 * @param X Left operand (n x d).
 * @param Y Right operand (m x d).
 * @param out Output matrix (n x m); @c isMutable() must be true.
 * @param pool Parallelism injection; forwarded to @c rowNormsSq and @c gemm and used for the
 *        broadcast-and-clamp sweep.
 */
template <class T, Layout LX, Layout LY>
void pairwiseSqEuclideanGemm(const NDArray<T, 2, LX> &X, const NDArray<T, 2, LY> &Y,
                             NDArray<T, 2> &out, Pool pool) {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                "pairwiseSqEuclideanGemm<T> requires T to be float or double");

  CLUSTERING_ALWAYS_ASSERT(out.isMutable());
  CLUSTERING_ALWAYS_ASSERT(X.dim(1) == Y.dim(1));
  CLUSTERING_ALWAYS_ASSERT(out.dim(0) == X.dim(0));
  CLUSTERING_ALWAYS_ASSERT(out.dim(1) == Y.dim(0));

  const std::size_t n = X.dim(0);
  const std::size_t m = Y.dim(0);
  if (n == 0 || m == 0) {
    return;
  }

  NDArray<T, 1> xNorms({n});
  NDArray<T, 1> yNorms({m});
  rowNormsSq(X, xNorms, pool);
  rowNormsSq(Y, yNorms, pool);

  gemm(X, Y.t(), out, pool, T{-2}, T{0});

  auto runBroadcastRange = [&](std::size_t lo, std::size_t hi) noexcept {
    for (std::size_t i = lo; i < hi; ++i) {
      const T xi = xNorms(i);
      for (std::size_t j = 0; j < m; ++j) {
        // Cancellation in ||x||^2 + ||y||^2 - 2 x . y can produce tiny negatives when x ~= y;
        // squared distance is non-negative by definition, so clamp.
        const T v = (out(i, j) + xi) + yNorms(j);
        out(i, j) = std::max(v, T{0});
      }
    }
  };

  const std::size_t totalCells = n * m;
  if (pool.shouldParallelize(totalCells, 64, 2)) {
    pool.pool
        ->submit_blocks(std::size_t{0}, n,
                        [&](std::size_t lo, std::size_t hi) { runBroadcastRange(lo, hi); })
        .wait();
  } else {
    runBroadcastRange(0, n);
  }
}

} // namespace detail

/**
 * @brief Pairwise squared Euclidean distances between rows of two matrices.
 *
 * Writes @c out(i, j) = sum_k (X(i, k) - Y(j, k))^2 for every row pair. @p out must be
 * mutable-owned and contiguous; shape mismatches trigger a release-active assert.
 *
 * @tparam T Element type (@c float or @c double).
 * @tparam LX Layout tag of @p X; CTAD-resolved so strided views (e.g. @c Z.t()) bind without
 *         explicit template arguments.
 * @tparam LY Layout tag of @p Y.
 * @param X Left operand (n x d).
 * @param Y Right operand (m x d).
 * @param out Output matrix (n x m). @c isMutable() must be true.
 * @param pool Parallelism injection; when the workload clears @c shouldParallelize the outer
 *        row loop fans out over the attached pool.
 */
template <class T, Layout LX = Layout::Contig, Layout LY = Layout::Contig>
void pairwiseSqEuclidean(const NDArray<T, 2, LX> &X, const NDArray<T, 2, LY> &Y, NDArray<T, 2> &out,
                         Pool pool) {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                "pairwiseSqEuclidean<T> requires T to be float or double");

  CLUSTERING_ALWAYS_ASSERT(out.isMutable());

  CLUSTERING_ALWAYS_ASSERT(X.dim(1) == Y.dim(1));
  CLUSTERING_ALWAYS_ASSERT(out.dim(0) == X.dim(0));
  CLUSTERING_ALWAYS_ASSERT(out.dim(1) == Y.dim(0));

  const std::size_t n = X.dim(0);
  const std::size_t m = Y.dim(0);
  if (n == 0 || m == 0) {
    return;
  }

  auto runRowRange = [&](std::size_t lo, std::size_t hi) noexcept {
    for (std::size_t i = lo; i < hi; ++i) {
      for (std::size_t j = 0; j < m; ++j) {
        out(i, j) = detail::sqEuclideanRow<T, LX, LY>(X, i, Y, j);
      }
    }
  };

  if (pool.shouldParallelize(n, 4, 2)) {
    pool.pool
        ->submit_blocks(std::size_t{0}, n,
                        [&](std::size_t lo, std::size_t hi) { runRowRange(lo, hi); })
        .wait();
  } else {
    runRowRange(0, n);
  }
}

} // namespace clustering::math
