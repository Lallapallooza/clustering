#pragma once

#include <array>
#include <cfloat>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>

#include "clustering/always_assert.h"
#include "clustering/math/detail/gemm_kernel_scalar.h"
#include "clustering/math/detail/gemm_outer.h"
#include "clustering/math/detail/gemm_pack.h"
#include "clustering/math/detail/matrix_desc.h"
#include "clustering/math/thread.h"
#include "clustering/ndarray.h"

#ifdef CLUSTERING_USE_AVX2
#include <immintrin.h>

#include "clustering/math/defaults.h"
#include "clustering/math/detail/gemm_kernel_avx2_f32_argmin.h"
#endif

namespace clustering::math::detail {

/**
 * @brief Pack the per-column squared norms of @c C into @c Nr -wide panels aligned with the
 *        @c packB panel layout.
 *
 * Panel @c p stores the norms for columns `[p*Nr, p*Nr + Nr)`. Positions beyond @p k are set
 * to `+inf`inity so padded columns never win the argmin contest.
 *
 * @tparam T Element type (@c float for the AVX2 path).
 * @param cSqNorms Row-1 array of length @p k with `||c_j||^2` per centroid.
 * @param k        Number of centroids (matches the @c nc passed to @c packB).
 * @param out      Destination buffer of capacity `ceil(k / Nr)` * Nr, 32-byte aligned.
 */
template <class T> inline void packCSqNorms(const T *cSqNorms, std::size_t k, T *out) noexcept {
  constexpr std::size_t kNr = kKernelNr<T>;
  const std::size_t panels = (k + kNr - 1) / kNr;
  for (std::size_t p = 0; p < panels; ++p) {
    const std::size_t cBase = p * kNr;
    const std::size_t cValid = (cBase + kNr <= k) ? kNr : (k - cBase);
    T *row = out + (p * kNr);
    for (std::size_t c = 0; c < cValid; ++c) {
      row[c] = cSqNorms[cBase + c];
    }
    for (std::size_t c = cValid; c < kNr; ++c) {
      row[c] = std::numeric_limits<T>::infinity();
    }
  }
}

#ifdef CLUSTERING_USE_AVX2

/**
 * @brief Process a single 8-row M-tile of the fused argmin-GEMM driver.
 *
 * Serial entry point used by callers that want to run their own outer fan-out (e.g. the kmeans
 * Lloyd loop, which fuses assignment with the per-cluster scatter to halve the per-iter
 * fork-join count).
 */
inline void argminFusedMTileF32(const NDArray<float, 2, Layout::Contig> &X, const float *bpacked,
                                const float *normsPacked, NDArray<std::int32_t, 1> &labels,
                                NDArray<float, 1> &outMinSq, std::size_t tileIdx, std::size_t n,
                                std::size_t k, std::size_t d) noexcept {
  constexpr std::size_t kMr = kKernelMr<float>;
  constexpr std::size_t kNr = kKernelNr<float>;
  const std::size_t nPanels = (k + kNr - 1) / kNr;
  const std::size_t cpanelSize = kNr * d;

  const std::size_t iBase = tileIdx * kMr;
  const std::size_t mc = (iBase + kMr <= n) ? kMr : (n - iBase);

  alignas(32) std::array<float, kMr * defaults::pairwiseArgminMaxD> apScratch{};
  CLUSTERING_ALWAYS_ASSERT(d <= defaults::pairwiseArgminMaxD);
  const auto xDesc = ::clustering::detail::describeMatrix(X);
  packA<float>(xDesc, iBase, mc, 0, d, apScratch.data());

  __m256 bestMin = _mm256_set1_ps(std::numeric_limits<float>::infinity());
  __m256i bestArg = _mm256_set1_epi32(-1);

  for (std::size_t p = 0; p < nPanels; ++p) {
    const auto jBase = static_cast<std::int32_t>(p * kNr);
    const float *bpPanel = bpacked + (p * cpanelSize);
    const float *normsPanel = normsPacked + (p * kNr);
    gemmKernel8x6Avx2F32FusedArgmin(apScratch.data(), bpPanel, normsPanel, d, jBase, bestMin,
                                    bestArg);
  }

  alignas(32) std::array<float, kMr> xNorms{};
  for (std::size_t r = 0; r < mc; ++r) {
    const float *row = X.data() + ((iBase + r) * d);
    float s = 0.0F;
    for (std::size_t t = 0; t < d; ++t) {
      s += row[t] * row[t];
    }
    xNorms[r] = s;
  }
  bestMin = _mm256_add_ps(bestMin, _mm256_load_ps(xNorms.data()));
  bestMin = _mm256_max_ps(bestMin, _mm256_setzero_ps());

  alignas(32) std::array<float, kMr> minBuf{};
  alignas(32) std::array<std::int32_t, kMr> argBuf{};
  _mm256_store_ps(minBuf.data(), bestMin);
  _mm256_store_si256(reinterpret_cast<__m256i *>(argBuf.data()), bestArg);

  std::memcpy(outMinSq.data() + iBase, minBuf.data(), mc * sizeof(float));
  std::memcpy(labels.data() + iBase, argBuf.data(), mc * sizeof(std::int32_t));
}

inline void argminFusedM16TileF32(const NDArray<float, 2, Layout::Contig> &X, const float *bpacked,
                                  const float *normsPacked, NDArray<std::int32_t, 1> &labels,
                                  NDArray<float, 1> &outMinSq, std::size_t tileIdx, std::size_t n,
                                  std::size_t k, std::size_t d) noexcept {
  constexpr std::size_t kMr = kKernelMr<float>;
  constexpr std::size_t kTileRows = 16;
  constexpr std::size_t kNr = kKernelNr<float>;
  const std::size_t nPanels = (k + kNr - 1) / kNr;
  const std::size_t cpanelSize = kNr * d;

  const std::size_t iBase = tileIdx * kTileRows;
  const std::size_t mc = (iBase + kTileRows <= n) ? kTileRows : (n - iBase);
  const std::size_t mc0 = (mc < kMr) ? mc : kMr;
  const std::size_t mc1 = mc - mc0;

  alignas(32) std::array<float, kTileRows * defaults::pairwiseArgminMaxD> apScratch{};
  CLUSTERING_ALWAYS_ASSERT(d <= defaults::pairwiseArgminMaxD);
  const auto xDesc = ::clustering::detail::describeMatrix(X);
  packA<float>(xDesc, iBase, mc, 0, d, apScratch.data());

  const float *ap0 = apScratch.data();
  const float *ap1 = apScratch.data() + (kMr * d);
  __m256 bestMin0 = _mm256_set1_ps(std::numeric_limits<float>::infinity());
  __m256 bestMin1 = _mm256_set1_ps(std::numeric_limits<float>::infinity());
  __m256i bestArg0 = _mm256_set1_epi32(-1);
  __m256i bestArg1 = _mm256_set1_epi32(-1);

  for (std::size_t p = 0; p < nPanels; ++p) {
    const auto jBase = static_cast<std::int32_t>(p * kNr);
    const float *bpPanel = bpacked + (p * cpanelSize);
    const float *normsPanel = normsPacked + (p * kNr);
    gemmKernel8x6Avx2F32FusedArgmin(ap0, bpPanel, normsPanel, d, jBase, bestMin0, bestArg0);
    if (mc1 > 0) {
      gemmKernel8x6Avx2F32FusedArgmin(ap1, bpPanel, normsPanel, d, jBase, bestMin1, bestArg1);
    }
  }

  alignas(32) std::array<float, kMr> xNorms0{};
  alignas(32) std::array<float, kMr> xNorms1{};
  for (std::size_t r = 0; r < mc0; ++r) {
    const float *row = X.data() + ((iBase + r) * d);
    float s = 0.0F;
    for (std::size_t t = 0; t < d; ++t) {
      s += row[t] * row[t];
    }
    xNorms0[r] = s;
  }
  for (std::size_t r = 0; r < mc1; ++r) {
    const float *row = X.data() + ((iBase + kMr + r) * d);
    float s = 0.0F;
    for (std::size_t t = 0; t < d; ++t) {
      s += row[t] * row[t];
    }
    xNorms1[r] = s;
  }
  bestMin0 = _mm256_add_ps(bestMin0, _mm256_load_ps(xNorms0.data()));
  bestMin0 = _mm256_max_ps(bestMin0, _mm256_setzero_ps());
  bestMin1 = _mm256_add_ps(bestMin1, _mm256_load_ps(xNorms1.data()));
  bestMin1 = _mm256_max_ps(bestMin1, _mm256_setzero_ps());

  alignas(32) std::array<float, kMr> minBuf0{};
  alignas(32) std::array<float, kMr> minBuf1{};
  alignas(32) std::array<std::int32_t, kMr> argBuf0{};
  alignas(32) std::array<std::int32_t, kMr> argBuf1{};
  _mm256_store_ps(minBuf0.data(), bestMin0);
  _mm256_store_ps(minBuf1.data(), bestMin1);
  _mm256_store_si256(reinterpret_cast<__m256i *>(argBuf0.data()), bestArg0);
  _mm256_store_si256(reinterpret_cast<__m256i *>(argBuf1.data()), bestArg1);

  std::memcpy(outMinSq.data() + iBase, minBuf0.data(), mc0 * sizeof(float));
  std::memcpy(labels.data() + iBase, argBuf0.data(), mc0 * sizeof(std::int32_t));
  if (mc1 > 0) {
    std::memcpy(outMinSq.data() + iBase + kMr, minBuf1.data(), mc1 * sizeof(float));
    std::memcpy(labels.data() + iBase + kMr, argBuf1.data(), mc1 * sizeof(std::int32_t));
  }
}

/**
 * @brief Process a single 8-row M-tile of the small-d direct argmin path.
 *
 * Serial entry point. Companion to @ref argminFusedMTileF32 for the @c d <= 8 regime where the
 * direct `||x - c||^2` formula beats the packed-GEMM path.
 */
inline void argminDirectMTileF32(const NDArray<float, 2, Layout::Contig> &X,
                                 const NDArray<float, 2, Layout::Contig> &C,
                                 NDArray<std::int32_t, 1> &labels, NDArray<float, 1> &outMinSq,
                                 std::size_t tileIdx, std::size_t n, std::size_t k,
                                 std::size_t d) noexcept {
  constexpr std::size_t kMr = 8;
  const float *xData = X.data();
  const float *cData = C.data();
  std::int32_t *labelsData = labels.data();
  float *minSqData = outMinSq.data();

  const std::size_t iBase = tileIdx * kMr;
  const std::size_t mc = (iBase + kMr <= n) ? kMr : (n - iBase);

  alignas(32) std::array<float, kMr * 8> xTile{};
  for (std::size_t r = 0; r < mc; ++r) {
    const float *src = xData + ((iBase + r) * d);
    for (std::size_t t = 0; t < d; ++t) {
      xTile[(t * kMr) + r] = src[t];
    }
  }

  __m256 bestMin = _mm256_set1_ps(std::numeric_limits<float>::infinity());
  __m256i bestArg = _mm256_set1_epi32(-1);

  for (std::size_t j = 0; j < k; ++j) {
    const float *cRow = cData + (j * d);
    __m256 acc = _mm256_setzero_ps();
    for (std::size_t t = 0; t < d; ++t) {
      const __m256 xv = _mm256_load_ps(xTile.data() + (t * kMr));
      const __m256 cv = _mm256_set1_ps(cRow[t]);
      const __m256 diff = _mm256_sub_ps(xv, cv);
      acc = _mm256_fmadd_ps(diff, diff, acc);
    }
    const __m256 mask = _mm256_cmp_ps(acc, bestMin, _CMP_LT_OQ);
    bestMin = _mm256_blendv_ps(bestMin, acc, mask);
    const __m256i jVec = _mm256_set1_epi32(static_cast<std::int32_t>(j));
    bestArg = _mm256_blendv_epi8(bestArg, jVec, _mm256_castps_si256(mask));
  }

  alignas(32) std::array<float, kMr> minBuf{};
  alignas(32) std::array<std::int32_t, kMr> argBuf{};
  _mm256_store_ps(minBuf.data(), bestMin);
  _mm256_store_si256(reinterpret_cast<__m256i *>(argBuf.data()), bestArg);
  std::memcpy(minSqData + iBase, minBuf.data(), mc * sizeof(float));
  std::memcpy(labelsData + iBase, argBuf.data(), mc * sizeof(std::int32_t));
}

inline void argminDirectM16TileF32(const NDArray<float, 2, Layout::Contig> &X,
                                   const NDArray<float, 2, Layout::Contig> &C,
                                   NDArray<std::int32_t, 1> &labels, NDArray<float, 1> &outMinSq,
                                   std::size_t tileIdx, std::size_t n, std::size_t k,
                                   std::size_t d) noexcept {
  constexpr std::size_t kMr = 8;
  constexpr std::size_t kTileRows = 16;
  const float *xData = X.data();
  const float *cData = C.data();
  std::int32_t *labelsData = labels.data();
  float *minSqData = outMinSq.data();

  const std::size_t iBase = tileIdx * kTileRows;
  const std::size_t mc = (iBase + kTileRows <= n) ? kTileRows : (n - iBase);
  const std::size_t mc0 = (mc < kMr) ? mc : kMr;
  const std::size_t mc1 = mc - mc0;

  alignas(32) std::array<float, kMr * 8> xTile0{};
  alignas(32) std::array<float, kMr * 8> xTile1{};
  for (std::size_t r = 0; r < mc0; ++r) {
    const float *src = xData + ((iBase + r) * d);
    for (std::size_t t = 0; t < d; ++t) {
      xTile0[(t * kMr) + r] = src[t];
    }
  }
  for (std::size_t r = 0; r < mc1; ++r) {
    const float *src = xData + ((iBase + kMr + r) * d);
    for (std::size_t t = 0; t < d; ++t) {
      xTile1[(t * kMr) + r] = src[t];
    }
  }

  __m256 bestMin0 = _mm256_set1_ps(std::numeric_limits<float>::infinity());
  __m256 bestMin1 = _mm256_set1_ps(std::numeric_limits<float>::infinity());
  __m256i bestArg0 = _mm256_set1_epi32(-1);
  __m256i bestArg1 = _mm256_set1_epi32(-1);

  for (std::size_t j = 0; j < k; ++j) {
    const float *cRow = cData + (j * d);
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    for (std::size_t t = 0; t < d; ++t) {
      const __m256 cv = _mm256_set1_ps(cRow[t]);
      const __m256 diff0 = _mm256_sub_ps(_mm256_load_ps(xTile0.data() + (t * kMr)), cv);
      const __m256 diff1 = _mm256_sub_ps(_mm256_load_ps(xTile1.data() + (t * kMr)), cv);
      acc0 = _mm256_fmadd_ps(diff0, diff0, acc0);
      acc1 = _mm256_fmadd_ps(diff1, diff1, acc1);
    }
    const __m256 mask0 = _mm256_cmp_ps(acc0, bestMin0, _CMP_LT_OQ);
    const __m256 mask1 = _mm256_cmp_ps(acc1, bestMin1, _CMP_LT_OQ);
    bestMin0 = _mm256_blendv_ps(bestMin0, acc0, mask0);
    bestMin1 = _mm256_blendv_ps(bestMin1, acc1, mask1);
    const __m256i jVec = _mm256_set1_epi32(static_cast<std::int32_t>(j));
    bestArg0 = _mm256_blendv_epi8(bestArg0, jVec, _mm256_castps_si256(mask0));
    bestArg1 = _mm256_blendv_epi8(bestArg1, jVec, _mm256_castps_si256(mask1));
  }

  alignas(32) std::array<float, kMr> minBuf0{};
  alignas(32) std::array<float, kMr> minBuf1{};
  alignas(32) std::array<std::int32_t, kMr> argBuf0{};
  alignas(32) std::array<std::int32_t, kMr> argBuf1{};
  _mm256_store_ps(minBuf0.data(), bestMin0);
  _mm256_store_ps(minBuf1.data(), bestMin1);
  _mm256_store_si256(reinterpret_cast<__m256i *>(argBuf0.data()), bestArg0);
  _mm256_store_si256(reinterpret_cast<__m256i *>(argBuf1.data()), bestArg1);
  std::memcpy(minSqData + iBase, minBuf0.data(), mc0 * sizeof(float));
  std::memcpy(labelsData + iBase, argBuf0.data(), mc0 * sizeof(std::int32_t));
  if (mc1 > 0) {
    std::memcpy(minSqData + iBase + kMr, minBuf1.data(), mc1 * sizeof(float));
    std::memcpy(labelsData + iBase + kMr, argBuf1.data(), mc1 * sizeof(std::int32_t));
  }
}

/**
 * @brief Pack the centroid matrix into the panel layout used by @ref argminFusedMTileF32.
 */
inline void packCentroidsForFusedArgminF32(const NDArray<float, 2, Layout::Contig> &C,
                                           std::size_t k, std::size_t d, float *bpacked) noexcept {
  constexpr std::size_t kNr = kKernelNr<float>;
  const std::size_t nPanels = (k + kNr - 1) / kNr;
  const std::size_t cpanelSize = kNr * d;
  const auto cTransposed = C.t();
  const auto cDesc = ::clustering::detail::describeMatrix(cTransposed);
  for (std::size_t p = 0; p < nPanels; ++p) {
    const std::size_t jBase = p * kNr;
    const std::size_t nc = (jBase + kNr <= k) ? kNr : (k - jBase);
    float *panelOut = bpacked + (p * cpanelSize);
    packB<float>(cDesc, 0, d, jBase, nc, panelOut);
  }
}

/**
 * @brief AVX2 f32 specialization of the fused argmin-GEMM outer driver.
 *
 * Iterates 8-row M-tiles. For each tile, packs an Mr x d A-strip via @c packA (full K range,
 * no K-blocking), then for every Nr-wide C-panel calls @c gemmKernel8x6Avx2F32FusedArgmin to
 * fold the per-column candidate distance into the tile's running `(bestMin, bestArg)`. At
 * the tile epilogue, adds `||x_i||^2` per row, clamps to zero, and writes out labels +
 * minimum distances.
 *
 * Per-tile scratch is stack-local (`alignas(32)` buffers sized for the worst-case @p d);
 * because the maximum @p d is bounded by @c defaults::pairwiseArgminMaxD at the public entry,
 * stack usage remains small and predictable.
 *
 * Precondition: @p X and @p C must expose 32-byte aligned contiguous rows (the public entry's
 * dispatch checks this); violating this precondition triggers undefined behavior on the
 * aligned @c _mm256_load_ps inside the kernel.
 *
 * @param X         Data matrix (n x d), contiguous, 32-byte aligned.
 * @param C         Centroid matrix (k x d), contiguous, 32-byte aligned.
 * @param cSqNorms  Per-centroid squared norms; length k.
 * @param labels    Output labels of length n.
 * @param outMinSq  Output minimum squared distances of length n.
 * @param pool      Parallelism injection; fans out over M-tiles when @c shouldParallelize.
 */
inline void pairwiseArgminOuterAvx2F32(const NDArray<float, 2, Layout::Contig> &X,
                                       const NDArray<float, 2, Layout::Contig> &C,
                                       const NDArray<float, 1> &cSqNorms,
                                       NDArray<std::int32_t, 1> &labels,
                                       NDArray<float, 1> &outMinSq, Pool pool) noexcept {
  constexpr std::size_t kMr = kKernelMr<float>;
  constexpr std::size_t kNr = kKernelNr<float>;

  const std::size_t n = X.dim(0);
  const std::size_t k = C.dim(0);
  const std::size_t d = X.dim(1);

  const std::size_t mTiles = (n + kMr - 1) / kMr;
  const std::size_t nPanels = (k + kNr - 1) / kNr;
  const std::size_t cpanelSize = kNr * d; // one packB panel: d rows by Nr cols
  const std::size_t normsPaddedSize = nPanels * kNr;
  const std::size_t bpackSize = nPanels * cpanelSize;

  // Pack the full centroid matrix once per call. Memory is ceil(k/Nr) * d * Nr floats; with the
  // fused path gated at `d <= defaults`::pairwiseArgminMaxD this is strictly bounded and fits
  // comfortably in L2 for the envelopes the fused driver is advantageous on.
  std::vector<float, ::clustering::detail::AlignedAllocator<float, 32>> bpackedStorage(bpackSize);
  std::vector<float, ::clustering::detail::AlignedAllocator<float, 32>> normsPackedStorage(
      normsPaddedSize);

  packCentroidsForFusedArgminF32(C, k, d, bpackedStorage.data());
  packCSqNorms<float>(cSqNorms.data(), k, normsPackedStorage.data());

  pool.parallelForBlocks(
      std::size_t{0}, mTiles, std::size_t{0}, [&](std::size_t lo, std::size_t hi) {
        for (std::size_t t = lo; t < hi; ++t) {
          argminFusedMTileF32(X, bpackedStorage.data(), normsPackedStorage.data(), labels, outMinSq,
                              t, n, k, d);
        }
      });
}

/**
 * @brief Minimum byte count of the packed-B scratch that
 *        @ref pairwiseArgminOuterAvx2F32WithScratch expects.
 *
 * Sized at `ceil(k / Nr)` * Nr * d floats. Callers that amortize assignment across many
 * iterations size this once at shape-change time so each assignment reuses the buffer.
 */
inline std::size_t packedBScratchSizeFloats(std::size_t k, std::size_t d) noexcept {
  constexpr std::size_t kNr = kKernelNr<float>;
  const std::size_t nPanels = (k + kNr - 1) / kNr;
  return nPanels * kNr * d;
}

/**
 * @brief Minimum element count of the packed centroid-norms scratch.
 */
inline std::size_t packedCSqNormsScratchSizeFloats(std::size_t k) noexcept {
  constexpr std::size_t kNr = kKernelNr<float>;
  const std::size_t nPanels = (k + kNr - 1) / kNr;
  return nPanels * kNr;
}

/**
 * @brief Total element count of a packed-B buffer laid out for @c gemmRunPrepacked.
 *
 * Sums @c k_dim * roundedNc over each @c jc block of width `kNc<T>`, where
 * @c roundedNc = ceil(nc / kNr<T>) * kNr<T>. The result is the exact buffer size
 * @c gemmRunPrepacked expects and matches @c GemmPlan's packing loop.
 */
template <class T>
inline std::size_t packedBScratchSizeFloatsTiled(std::size_t k, std::size_t d) noexcept {
  constexpr std::size_t kNr = kKernelNr<T>;
  constexpr std::size_t kNcVal = kNc<T>;
  if (k == 0 || d == 0) {
    return 0;
  }
  std::size_t total = 0;
  for (std::size_t jc = 0; jc < k; jc += kNcVal) {
    const std::size_t nc = (jc + kNcVal <= k) ? kNcVal : (k - jc);
    const std::size_t roundedNc = ((nc + kNr - 1) / kNr) * kNr;
    total += d * roundedNc;
  }
  return total;
}

/**
 * @brief AVX2 f32 fused argmin-GEMM outer driver that consumes caller-owned packed scratch.
 *
 * Logically identical to @ref pairwiseArgminOuterAvx2F32 but skips the per-call
 * @c std::vector allocation of the packed B panels and packed centroid norms. The caller
 * supplies pre-sized, 32-byte aligned buffers; this routine refreshes their contents (packB
 * + packCSqNorms) against the current @p C before running the tile loop. Used by
 * @c kmeans::detail::Solver to keep the Lloyd assignment step allocation-free across
 * iterations.
 *
 * @param X                 Data matrix (n x d), contiguous, 32-byte aligned.
 * @param C                 Centroid matrix (k x d), contiguous, 32-byte aligned.
 * @param cSqNorms          Per-centroid squared norms; length k.
 * @param labels            Output labels of length n.
 * @param outMinSq          Output minimum squared distances of length n.
 * @param bpackedScratch    Scratch of at least @ref packedBScratchSizeFloats elements.
 * @param normsPackedScratch Scratch of at least @ref packedCSqNormsScratchSizeFloats elements.
 * @param pool              Parallelism injection.
 */
inline void pairwiseArgminOuterAvx2F32WithScratch(const NDArray<float, 2, Layout::Contig> &X,
                                                  const NDArray<float, 2, Layout::Contig> &C,
                                                  const NDArray<float, 1> &cSqNorms,
                                                  NDArray<std::int32_t, 1> &labels,
                                                  NDArray<float, 1> &outMinSq,
                                                  float *bpackedScratch, float *normsPackedScratch,
                                                  Pool pool) noexcept {
  constexpr std::size_t kMr = kKernelMr<float>;

  const std::size_t n = X.dim(0);
  const std::size_t k = C.dim(0);
  const std::size_t d = X.dim(1);

  const std::size_t mTiles = (n + kMr - 1) / kMr;

  packCentroidsForFusedArgminF32(C, k, d, bpackedScratch);
  packCSqNorms<float>(cSqNorms.data(), k, normsPackedScratch);

  pool.parallelForBlocks(
      std::size_t{0}, mTiles, std::size_t{0}, [&](std::size_t lo, std::size_t hi) {
        for (std::size_t t = lo; t < hi; ++t) {
          argminFusedMTileF32(X, bpackedScratch, normsPackedScratch, labels, outMinSq, t, n, k, d);
        }
      });
}

/**
 * @brief Direct squared-distance argmin for the small-@c d hot path (no GEMM packing).
 *
 * At `d <= a` few lanes, the fused argmin-GEMM driver spends a disproportionate share of its
 * wall time in @c packA: packing an 8x2 A-panel into the microkernel layout costs nearly as
 * much as the 96-FMA kernel it feeds. This routine collapses the assignment to the direct
 * `||x_i - c_j||^2` formula, iterated 8 rows at a time with AVX2 accumulators and broadcast
 * centroid components. Writes the true squared distance (no `||x||^2` + `||c||^2` - 2 x.c
 * decomposition) so downstream callers consume numerically faithful inertia without a final
 * @c recomputeMinDistSqDirect pass.
 *
 * @param X        Data matrix (n x d), contiguous, 32-byte aligned.
 * @param C        Centroid matrix (k x d), contiguous.
 * @param labels   Output labels of length n.
 * @param outMinSq Output squared distances of length n.
 * @param pool     Parallelism injection; pooled calls fan out over 16-row M-tiles.
 */
inline void pairwiseArgminDirectSmallDF32(const NDArray<float, 2, Layout::Contig> &X,
                                          const NDArray<float, 2, Layout::Contig> &C,
                                          NDArray<std::int32_t, 1> &labels,
                                          NDArray<float, 1> &outMinSq, Pool pool) noexcept {
  constexpr std::size_t kMr8 = 8;
  constexpr std::size_t kMr16 = 16;
  constexpr std::size_t kMaxD = 8;
  const std::size_t n = X.dim(0);
  const std::size_t k = C.dim(0);
  const std::size_t d = X.dim(1);
  CLUSTERING_ALWAYS_ASSERT(d >= 1);
  CLUSTERING_ALWAYS_ASSERT(d <= kMaxD);

  const bool useWideTile = pool.workerCount() > 1;
  const std::size_t mTiles = useWideTile ? ((n + kMr16 - 1) / kMr16) : ((n + kMr8 - 1) / kMr8);
  pool.parallelForBlocks(std::size_t{0}, mTiles, std::size_t{0},
                         [&](std::size_t lo, std::size_t hi) {
                           for (std::size_t t = lo; t < hi; ++t) {
                             if (useWideTile) {
                               argminDirectM16TileF32(X, C, labels, outMinSq, t, n, k, d);
                             } else {
                               argminDirectMTileF32(X, C, labels, outMinSq, t, n, k, d);
                             }
                           }
                         });
}

#endif // CLUSTERING_USE_AVX2

} // namespace clustering::math::detail
