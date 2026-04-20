#pragma once

#include <cstddef>
#include <type_traits>

#include "clustering/math/detail/avx2_helpers.h"

#ifdef CLUSTERING_USE_AVX2
#include <immintrin.h>
#endif

namespace clustering::math::detail {

/**
 * @brief Scans @p count contiguous points (AoS, row-major with @p d floats per row) and
 *        calls @p emit(i) for each @c i in @c [0, count) whose squared distance to @p qp is
 *        at most @p radius_sq.
 *
 * Dispatches to SIMD-specialized inner kernels when the build is AVX2-enabled and @p d clears
 * a width the kernel can exploit. Today the batched kernel exists for @c f32 at @c d == 2
 * (4 pairs per iteration via shuffle); every other @c (T, d) combination falls through to the
 * scalar-per-row loop driven by @ref sqEuclideanRowPtr. Keeping the fallback in the same header
 * lets downstream kernels take the shortcut without re-implementing the scalar baseline.
 *
 * Intended consumer: @c KDTree leaf brute force, where @p pts_aos points at the leaf's slice of
 * the tree-reordered point buffer so the row-major iteration lands on sequential cache lines.
 *
 * @tparam T      Element type (@c float or @c double).
 * @tparam EmitFn Functor invoked with an in-leaf row index @c i; typically appends the matching
 *                original point id to the caller's adjacency row.
 *
 * @param qp         Query point; @p d contiguous elements.
 * @param pts_aos    Base pointer to the leaf's points; @c count*d contiguous elements.
 * @param count      Number of candidate rows in the leaf.
 * @param d          Row length (also the stride in elements inside @p pts_aos).
 * @param radius_sq  Squared radius; comparisons are inclusive.
 * @param emit       Callback invoked with @c (std::size_t i) for each matching row.
 */
template <class T, class EmitFn>
inline void radiusScan(const T *qp, const T *pts_aos, std::size_t count, std::size_t d, T radius_sq,
                       EmitFn &&emit) noexcept {
#ifdef CLUSTERING_USE_AVX2
  if constexpr (std::is_same_v<T, float>) {
    if (d == 2) {
      // 4-wide AVX2 f32: load two xmm pairs of AoS x/y, shuffle to SoA, FMA to dsq, mask-compare,
      // emit set lanes. The shuffle epilogue amortizes the load so the inner loop runs at
      // 4 distances / ~5 cycles on Zen 4 / Ice Lake-class cores.
      const __m128 qx = _mm_set1_ps(qp[0]);
      const __m128 qy = _mm_set1_ps(qp[1]);
      const __m128 rsq = _mm_set1_ps(radius_sq);
      std::size_t i = 0;
      for (; i + 4 <= count; i += 4) {
        const __m128 lo = _mm_loadu_ps(pts_aos + (i * 2));       // x0,y0,x1,y1
        const __m128 hi = _mm_loadu_ps(pts_aos + ((i + 2) * 2)); // x2,y2,x3,y3
        const __m128 xs = _mm_shuffle_ps(lo, hi, _MM_SHUFFLE(2, 0, 2, 0));
        const __m128 ys = _mm_shuffle_ps(lo, hi, _MM_SHUFFLE(3, 1, 3, 1));
        const __m128 dx = _mm_sub_ps(xs, qx);
        const __m128 dy = _mm_sub_ps(ys, qy);
        const __m128 sq = _mm_fmadd_ps(dy, dy, _mm_mul_ps(dx, dx));
        const __m128 cmp = _mm_cmp_ps(sq, rsq, _CMP_LE_OS);
        const int mask = _mm_movemask_ps(cmp);
        if ((mask & 1) != 0) {
          emit(i);
        }
        if ((mask & 2) != 0) {
          emit(i + 1);
        }
        if ((mask & 4) != 0) {
          emit(i + 2);
        }
        if ((mask & 8) != 0) {
          emit(i + 3);
        }
      }
      // Scalar tail for the last 0-3 rows.
      for (; i < count; ++i) {
        const float ex = pts_aos[i * 2] - qp[0];
        const float ey = pts_aos[(i * 2) + 1] - qp[1];
        const float sq = (ex * ex) + (ey * ey);
        if (sq <= radius_sq) {
          emit(i);
        }
      }
      return;
    }
  }
#endif
  for (std::size_t i = 0; i < count; ++i) {
    const T dsq = sqEuclideanRowPtr(qp, pts_aos + (i * d), d);
    if (dsq <= radius_sq) {
      emit(i);
    }
  }
}

} // namespace clustering::math::detail
