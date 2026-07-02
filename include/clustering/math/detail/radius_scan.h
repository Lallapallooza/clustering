#pragma once

#include <array>
#include <cstddef>
#include <type_traits>

#include "clustering/math/detail/avx2_helpers.h"
#include "clustering/math/detail/sq_distances_block.h"

#ifdef CLUSTERING_USE_AVX2
#include <immintrin.h>
#endif

namespace clustering::math::detail {

/**
 * @brief Scans @p count contiguous points (AoS, row-major with @p d floats per row) and
 *        calls @p emit(i) for each @c i in `[0, count)` whose squared distance to @p qp is
 *        at most @p radius_sq.
 *
 * Dispatches to SIMD-specialized inner kernels when the build is AVX2-enabled and @p d clears
 * a width the kernel can exploit. For @c f32 the `d == 2` path packs 4 pairs per iteration via
 * shuffle, and `d >= 8` routes through @ref sqDistancesAosBlock so one horizontal-sum tree is
 * amortised across four neighbours instead of one per row; the remaining `(T, d)` pairs fall
 * through to the scalar-per-row loop driven by @ref sqEuclideanRowPtr.
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
 * @param emit       Callback invoked with `(std::size_t i)` for each matching row.
 */
template <class T, class EmitFn>
inline void radiusScan(const T *qp, const T *pts_aos, std::size_t count, std::size_t d, T radius_sq,
                       EmitFn &&emit) noexcept {
#ifdef CLUSTERING_USE_AVX2
  if constexpr (std::is_same_v<T, float>) {
    if (d == 2) {
      // 4-wide AVX2 f32: load two xmm pairs of AoS x/y, shuffle to SoA, FMA to dsq, mask-compare,
      // emit set lanes. The shuffle epilogue amortises the load so the inner loop runs at
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
        // Walk the 4-lane survivor mask one set bit at a time. The common cluster-interior block
        // has no survivors, so the single mask-empty test replaces four per-lane branches that
        // each mispredict on a cluster edge.
        auto m = static_cast<unsigned>(_mm_movemask_ps(cmp));
        while (m != 0) {
          emit(i + static_cast<std::size_t>(__builtin_ctz(m)));
          m &= m - 1;
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
    if (d >= kAvx2Lanes<T>) {
      // For d >= 8 the per-point horizontal-sum epilogue dominates the scan. Route the leaf
      // through the block kernel, which amortises one hsum tree across four neighbours, then
      // threshold the buffered distances. Chunked so the scratch stays one cache line.
      constexpr std::size_t kBlk = 16;
      alignas(32) std::array<T, kBlk> dsq;
      std::array<std::size_t, kBlk> hits;
      for (std::size_t base = 0; base < count; base += kBlk) {
        const std::size_t m = count - base < kBlk ? count - base : kBlk;
        sqDistancesAosBlock<T>(qp, pts_aos + (base * d), m, d, dsq.data());
        // Branchless survivor compaction: the per-point threshold is data-dependent and
        // mispredicts on cluster-edge leaves. Advance the write cursor by the comparison result
        // so only the predictable emit loop over the survivor count carries a branch.
        std::size_t h = 0;
        for (std::size_t j = 0; j < m; ++j) {
          hits[h] = base + j;
          h += static_cast<std::size_t>(dsq[j] <= radius_sq);
        }
        for (std::size_t t = 0; t < h; ++t) {
          emit(hits[t]);
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

/**
 * @brief Range scan over a leaf laid out feature-major (SoA): @p leaf_soa holds feature @c f for
 *        every point at `leaf_soa[f * count + p]`.
 *
 * The SoA layout puts each feature's value for eight points in one contiguous vector, so the
 * squared distance accumulates straight down the feature axis with the eight points held in lanes
 * and never needs a horizontal sum. That trades the AoS block kernel's shuffle-port-bound hsum
 * tree for a load-and-FMA loop, which is the dominant cost of the low-dimension range query.
 *
 * @tparam T      Element type (@c float fast path; scalar otherwise).
 * @tparam EmitFn Functor invoked with an in-leaf point index for each match.
 * @param qp        Query point; @p d contiguous elements.
 * @param leaf_soa  Feature-major leaf block; `d * count` contiguous elements.
 * @param count     Point count in the leaf.
 * @param d         Feature count.
 * @param radius_sq Squared radius; comparison is inclusive.
 * @param emit      Callback invoked with `(std::size_t p)` for each matching point.
 */
template <class T, class EmitFn>
inline void radiusScanSoa(const T *qp, const T *leaf_soa, std::size_t count, std::size_t d,
                          T radius_sq, EmitFn &&emit) noexcept {
#ifdef CLUSTERING_USE_AVX2
  if constexpr (std::is_same_v<T, float>) {
    const __m256 rsq = _mm256_set1_ps(radius_sq);
    std::size_t p = 0;
    const auto emitGroup = [&](__m256 acc, std::size_t outBase) noexcept {
      auto m = static_cast<unsigned>(_mm256_movemask_ps(_mm256_cmp_ps(acc, rsq, _CMP_LE_OS)));
      while (m != 0) {
        emit(outBase + static_cast<std::size_t>(__builtin_ctz(m)));
        m &= m - 1;
      }
    };
    // Four point-groups in flight: the feature-axis FMA chains overlap to hide multiply-add
    // latency, and the one q broadcast per feature amortises across thirty-two distances, which
    // keeps the loop load bound rather than stalled on the chain or starved on the broadcast.
    for (; p + 32 <= count; p += 32) {
      __m256 acc0 = _mm256_setzero_ps();
      __m256 acc1 = _mm256_setzero_ps();
      __m256 acc2 = _mm256_setzero_ps();
      __m256 acc3 = _mm256_setzero_ps();
      for (std::size_t f = 0; f < d; ++f) {
        const __m256 q = _mm256_broadcast_ss(qp + f);
        const float *col = leaf_soa + (f * count) + p;
        const __m256 e0 = _mm256_sub_ps(_mm256_loadu_ps(col), q);
        const __m256 e1 = _mm256_sub_ps(_mm256_loadu_ps(col + 8), q);
        const __m256 e2 = _mm256_sub_ps(_mm256_loadu_ps(col + 16), q);
        const __m256 e3 = _mm256_sub_ps(_mm256_loadu_ps(col + 24), q);
        acc0 = _mm256_fmadd_ps(e0, e0, acc0);
        acc1 = _mm256_fmadd_ps(e1, e1, acc1);
        acc2 = _mm256_fmadd_ps(e2, e2, acc2);
        acc3 = _mm256_fmadd_ps(e3, e3, acc3);
      }
      emitGroup(acc0, p);
      emitGroup(acc1, p + 8);
      emitGroup(acc2, p + 16);
      emitGroup(acc3, p + 24);
    }
    for (; p + 8 <= count; p += 8) {
      __m256 acc = _mm256_setzero_ps();
      for (std::size_t f = 0; f < d; ++f) {
        const __m256 q = _mm256_broadcast_ss(qp + f);
        const __m256 e = _mm256_sub_ps(_mm256_loadu_ps(leaf_soa + (f * count) + p), q);
        acc = _mm256_fmadd_ps(e, e, acc);
      }
      emitGroup(acc, p);
    }
    for (; p < count; ++p) {
      T s = T{0};
      for (std::size_t f = 0; f < d; ++f) {
        const T diff = leaf_soa[(f * count) + p] - qp[f];
        s += diff * diff;
      }
      if (s <= radius_sq) {
        emit(p);
      }
    }
    return;
  }
#endif
  for (std::size_t p = 0; p < count; ++p) {
    T s = T{0};
    for (std::size_t f = 0; f < d; ++f) {
      const T diff = leaf_soa[(f * count) + p] - qp[f];
      s += diff * diff;
    }
    if (s <= radius_sq) {
      emit(p);
    }
  }
}

/**
 * @brief Paired-source variant of @ref radiusScanSoa: two query points sweep the same leaf in
 *        one pass.
 *
 * The single-source loop is load bound on the feature columns; sharing each column load
 * between two sources nearly halves the memory operations per scanned pair and folds two
 * call setups into one. Emit callbacks stay per-source so each caller-side row keeps a single
 * writer.
 *
 * @tparam EmitFn0 `std::invocable<std::size_t>` callable for the first source.
 * @tparam EmitFn1 `std::invocable<std::size_t>` callable for the second source.
 */
template <class T, class EmitFn0, class EmitFn1>
inline void radiusScanSoaPair(const T *qp0, const T *qp1, const T *leaf_soa, std::size_t count,
                              std::size_t d, T radius_sq, EmitFn0 &&emit0,
                              EmitFn1 &&emit1) noexcept {
#ifdef CLUSTERING_USE_AVX2
  if constexpr (std::is_same_v<T, float>) {
    const __m256 rsq = _mm256_set1_ps(radius_sq);
    const auto emitGroup = [&rsq](__m256 acc, std::size_t outBase, auto &sink) noexcept {
      auto m = static_cast<unsigned>(_mm256_movemask_ps(_mm256_cmp_ps(acc, rsq, _CMP_LE_OS)));
      while (m != 0) {
        sink(outBase + static_cast<std::size_t>(__builtin_ctz(m)));
        m &= m - 1;
      }
    };
    std::size_t p = 0;
    // Two point-groups by two sources keeps four accumulator chains in flight, enough to
    // cover the multiply-add latency while each column load feeds both sources.
    for (; p + 16 <= count; p += 16) {
      __m256 a00 = _mm256_setzero_ps();
      __m256 a01 = _mm256_setzero_ps();
      __m256 a10 = _mm256_setzero_ps();
      __m256 a11 = _mm256_setzero_ps();
      for (std::size_t f = 0; f < d; ++f) {
        const __m256 q0 = _mm256_broadcast_ss(qp0 + f);
        const __m256 q1 = _mm256_broadcast_ss(qp1 + f);
        const float *col = leaf_soa + (f * count) + p;
        const __m256 c0 = _mm256_loadu_ps(col);
        const __m256 c1 = _mm256_loadu_ps(col + 8);
        const __m256 e00 = _mm256_sub_ps(c0, q0);
        const __m256 e01 = _mm256_sub_ps(c1, q0);
        const __m256 e10 = _mm256_sub_ps(c0, q1);
        const __m256 e11 = _mm256_sub_ps(c1, q1);
        a00 = _mm256_fmadd_ps(e00, e00, a00);
        a01 = _mm256_fmadd_ps(e01, e01, a01);
        a10 = _mm256_fmadd_ps(e10, e10, a10);
        a11 = _mm256_fmadd_ps(e11, e11, a11);
      }
      emitGroup(a00, p, emit0);
      emitGroup(a01, p + 8, emit0);
      emitGroup(a10, p, emit1);
      emitGroup(a11, p + 8, emit1);
    }
    for (; p + 8 <= count; p += 8) {
      __m256 a0 = _mm256_setzero_ps();
      __m256 a1 = _mm256_setzero_ps();
      for (std::size_t f = 0; f < d; ++f) {
        const __m256 c = _mm256_loadu_ps(leaf_soa + (f * count) + p);
        const __m256 e0 = _mm256_sub_ps(c, _mm256_broadcast_ss(qp0 + f));
        const __m256 e1 = _mm256_sub_ps(c, _mm256_broadcast_ss(qp1 + f));
        a0 = _mm256_fmadd_ps(e0, e0, a0);
        a1 = _mm256_fmadd_ps(e1, e1, a1);
      }
      emitGroup(a0, p, emit0);
      emitGroup(a1, p, emit1);
    }
    for (; p < count; ++p) {
      T s0 = T{0};
      T s1 = T{0};
      for (std::size_t f = 0; f < d; ++f) {
        const T v = leaf_soa[(f * count) + p];
        const T d0 = v - qp0[f];
        const T d1 = v - qp1[f];
        s0 += d0 * d0;
        s1 += d1 * d1;
      }
      if (s0 <= radius_sq) {
        emit0(p);
      }
      if (s1 <= radius_sq) {
        emit1(p);
      }
    }
    return;
  }
#endif
  radiusScanSoa(qp0, leaf_soa, count, d, radius_sq, emit0);
  radiusScanSoa(qp1, leaf_soa, count, d, radius_sq, emit1);
}

} // namespace clustering::math::detail
