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
#include "clustering/math/detail/gemm_pack.h"
#include "clustering/math/detail/matrix_desc.h"
#include "clustering/math/thread.h"
#include "clustering/ndarray.h"

#ifdef CLUSTERING_USE_AVX2
#include <immintrin.h>

#include "clustering/math/detail/gemm_kernel_avx2_f32_argmin.h"
#endif

namespace clustering::math::detail {

/**
 * @brief Pack the per-column squared norms of @c C into @c Nr -wide panels aligned with the
 *        @c packB panel layout.
 *
 * Panel @c p stores the norms for columns @c [p*Nr, p*Nr + Nr). Positions beyond @p k are set
 * to @c +infinity so padded columns never win the argmin contest.
 *
 * @tparam T Element type (@c float for the AVX2 path).
 * @param cSqNorms Row-1 array of length @p k with @c ||c_j||^2 per centroid.
 * @param k        Number of centroids (matches the @c nc passed to @c packB).
 * @param out      Destination buffer of capacity @c ceil(k / Nr) * Nr, 32-byte aligned.
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

/**
 * @brief Scalar fallback of the fused argmin-GEMM outer driver.
 *
 * Correctness-oracle path for inputs that do not satisfy the AVX2 specialization's
 * preconditions (e.g. @c T != float, strided X or C, or @c d == 0). Produces identical labels
 * to the AVX2 driver within float reassociation tolerance; the strict-less-than tie-break is
 * mirrored so ties resolve to the earliest column in both paths.
 *
 * @tparam T  Element type; float or double per the NDArray invariant.
 * @tparam LX Layout tag of @p X.
 * @tparam LC Layout tag of @p C.
 * @param X         Data matrix (n x d).
 * @param C         Centroid matrix (k x d).
 * @param cSqNorms  Per-centroid squared norms; length k.
 * @param labels    Output labels of length n.
 * @param outMinSq  Output minimum squared distances of length n.
 */
template <class T, Layout LX, Layout LC>
void pairwiseArgminOuterScalar(const NDArray<T, 2, LX> &X, const NDArray<T, 2, LC> &C,
                               const NDArray<T, 1> &cSqNorms, NDArray<std::int32_t, 1> &labels,
                               NDArray<T, 1> &outMinSq, Pool pool) noexcept {
  const std::size_t n = X.dim(0);
  const std::size_t k = C.dim(0);
  const std::size_t d = X.dim(1);

  auto runRowRange = [&](std::size_t lo, std::size_t hi) noexcept {
    for (std::size_t i = lo; i < hi; ++i) {
      T xNormSq = T{0};
      for (std::size_t t = 0; t < d; ++t) {
        const T xit = X(i, t);
        xNormSq += xit * xit;
      }
      T bestMin = std::numeric_limits<T>::infinity();
      std::int32_t bestArg = -1;
      for (std::size_t j = 0; j < k; ++j) {
        // -2 * dot + ||c_j||^2 — same per-column residual the AVX2 path tracks; add xNormSq
        // at the end to recover the full squared distance.
        T dot = T{0};
        for (std::size_t t = 0; t < d; ++t) {
          dot += X(i, t) * C(j, t);
        }
        const T cand = cSqNorms(j) - (T{2} * dot);
        if (cand < bestMin) {
          bestMin = cand;
          bestArg = static_cast<std::int32_t>(j);
        }
      }
      T dist = bestMin + xNormSq;
      if (dist < T{0}) {
        dist = T{0};
      }
      outMinSq(i) = dist;
      labels(i) = bestArg;
    }
  };

  if (pool.shouldParallelize(n, 4, 2) && pool.pool != nullptr) {
    pool.pool
        ->submit_blocks(std::size_t{0}, n,
                        [&](std::size_t lo, std::size_t hi) { runRowRange(lo, hi); })
        .wait();
  } else {
    runRowRange(0, n);
  }
}

#ifdef CLUSTERING_USE_AVX2

/**
 * @brief AVX2 f32 specialization of the fused argmin-GEMM outer driver.
 *
 * Iterates 8-row M-tiles. For each tile, packs an Mr x d A-strip via @c packA (full K range,
 * no K-blocking), then for every Nr-wide C-panel calls @c gemmKernel8x6Avx2F32FusedArgmin to
 * fold the per-column candidate distance into the tile's running @c (bestMin, bestArg). At
 * the tile epilogue, adds @c ||x_i||^2 per row, clamps to zero, and writes out labels +
 * minimum distances.
 *
 * Per-tile scratch is stack-local (@c alignas(32) buffers sized for the worst-case @p d);
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

  // Pack the full centroid matrix once per call. Memory is ceil(k/Nr) * d * Nr floats; for the
  // V1 envelope (k <= 1024, d <= 128) this is bounded at ~772 KiB — large but strictly better
  // than re-packing per M-tile.
  std::vector<float, ::clustering::detail::AlignedAllocator<float, 32>> bpackedStorage(bpackSize);
  std::vector<float, ::clustering::detail::AlignedAllocator<float, 32>> normsPackedStorage(
      normsPaddedSize);

  // packB expects B in @c (K x N) orientation, i.e. features-by-centroids. Our @p C is
  // @c (centroids x features), so take its transpose view before handing the descriptor to
  // packB. @c C.t() is a borrowed MaybeStrided view; describeMatrix preserves the strides so
  // packB's scalar element access picks up the right C[j][k_iter] per packed position.
  const auto cTransposed = C.t();
  const auto cDesc = ::clustering::detail::describeMatrix(cTransposed);
  for (std::size_t p = 0; p < nPanels; ++p) {
    const std::size_t jBase = p * kNr;
    const std::size_t nc = (jBase + kNr <= k) ? kNr : (k - jBase);
    float *panelOut = bpackedStorage.data() + (p * cpanelSize);
    packB<float>(cDesc, 0, d, jBase, nc, panelOut);
  }
  packCSqNorms<float>(cSqNorms.data(), k, normsPackedStorage.data());

  auto runOneMTile = [&](std::size_t tileIdx) noexcept {
    const std::size_t iBase = tileIdx * kMr;
    const std::size_t mc = (iBase + kMr <= n) ? kMr : (n - iBase);

    // Pack the M-strip via packA at column 0, full K. ceil(mc/Mr) * Mr * d floats; for mc <= 8
    // that is exactly Mr * d = 8 * d <= 8 * 256 = 2048 floats (8 KiB) per tile.
    alignas(32) std::array<float, kMr * 256> apScratch{};
    CLUSTERING_ALWAYS_ASSERT(d <= 256);
    const auto xDesc = ::clustering::detail::describeMatrix(X);
    packA<float>(xDesc, iBase, mc, 0, d, apScratch.data());

    __m256 bestMin = _mm256_set1_ps(std::numeric_limits<float>::infinity());
    __m256i bestArg = _mm256_set1_epi32(-1);

    for (std::size_t p = 0; p < nPanels; ++p) {
      const auto jBase = static_cast<std::int32_t>(p * kNr);
      const float *bpPanel = bpackedStorage.data() + (p * cpanelSize);
      const float *normsPanel = normsPackedStorage.data() + (p * kNr);
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
    // Padding lanes for the last (partial) tile carry bogus norms; their best* entries are
    // discarded by the write-back below, so setting xNorms past mc to zero is a consistency
    // hygiene rather than a correctness requirement.

    bestMin = _mm256_add_ps(bestMin, _mm256_load_ps(xNorms.data()));
    bestMin = _mm256_max_ps(bestMin, _mm256_setzero_ps());

    alignas(32) std::array<float, kMr> minBuf{};
    alignas(32) std::array<std::int32_t, kMr> argBuf{};
    _mm256_store_ps(minBuf.data(), bestMin);
    _mm256_store_si256(reinterpret_cast<__m256i *>(argBuf.data()), bestArg);

    std::memcpy(outMinSq.data() + iBase, minBuf.data(), mc * sizeof(float));
    std::memcpy(labels.data() + iBase, argBuf.data(), mc * sizeof(std::int32_t));
  };

  if (pool.shouldParallelize(mTiles, 1, 2) && pool.pool != nullptr) {
    pool.pool
        ->submit_blocks(std::size_t{0}, mTiles,
                        [&](std::size_t lo, std::size_t hi) {
                          for (std::size_t t = lo; t < hi; ++t) {
                            runOneMTile(t);
                          }
                        })
        .wait();
  } else {
    for (std::size_t t = 0; t < mTiles; ++t) {
      runOneMTile(t);
    }
  }
}

#endif // CLUSTERING_USE_AVX2

} // namespace clustering::math::detail
