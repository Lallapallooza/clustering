#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

#include "clustering/always_assert.h"
#include "clustering/math/detail/gemm_kernel_scalar.h"
#include "clustering/math/detail/gemm_pack.h"
#include "clustering/math/detail/matrix_desc.h"
#include "clustering/math/thread.h"
#include "clustering/ndarray.h"

#ifdef CLUSTERING_USE_AVX2
#include <immintrin.h>

#include "clustering/math/detail/gemm_kernel_avx2_f32_threshold.h"
#endif

namespace clustering::math::detail {

/**
 * @brief Pack per-column squared norms of @c Y into @c Nr -wide panels aligned with the
 *        @c packB panel layout.
 *
 * Panel @c p stores the norms for columns @c [p*Nr, p*Nr + Nr). Positions beyond @p m are
 * set to @c +infinity so padded columns can never pass a finite @c radiusSq comparison.
 *
 * @tparam T Element type (@c float for the AVX2 path).
 * @param yRowNormsSq Row-1 array of length @p m with @c ||y_j||^2 per point.
 * @param m           Number of points in @c Y.
 * @param out         Destination buffer of capacity @c ceil(m / Nr) * Nr, 32-byte aligned.
 */
template <class T>
inline void packYColNormsSq(const T *yRowNormsSq, std::size_t m, T *out) noexcept {
  constexpr std::size_t kNr = kKernelNr<T>;
  const std::size_t panels = (m + kNr - 1) / kNr;
  for (std::size_t p = 0; p < panels; ++p) {
    const std::size_t cBase = p * kNr;
    const std::size_t cValid = (cBase + kNr <= m) ? kNr : (m - cBase);
    T *row = out + (p * kNr);
    for (std::size_t c = 0; c < cValid; ++c) {
      row[c] = yRowNormsSq[cBase + c];
    }
    for (std::size_t c = cValid; c < kNr; ++c) {
      row[c] = std::numeric_limits<T>::infinity();
    }
  }
}

#ifdef CLUSTERING_USE_AVX2

/**
 * @brief Upper bound on @c d handled by the fused threshold outer driver's per-tile stack
 *        scratch for packA.
 *
 * Raising this past 256 pushes the packA buffer past 8 KiB per tile, at which point
 * worker-stack pressure becomes non-trivial under nested parallelism.
 */
inline constexpr std::size_t kThresholdMaxD = 256;

/**
 * @brief Row-chunk height used by the fused threshold outer driver when striping over @c n.
 *
 * Mirrors @ref pairwiseArgminChunkRows = 256 so the packA arena stays L2-resident across the
 * chunk's tile sweep on targets with >= 256 KiB L2.
 */
inline constexpr std::size_t kThresholdChunkRows = 256;

/**
 * @brief Number of @c Nr-wide B panels processed as a group so their packed bytes stay
 *        L1-resident while every M-tile in the current chunk sweeps through them.
 *
 * A group of 32 panels at @c Nr=6 and @c d=64 is 48 KiB -- roughly one Zen L1d.
 */
inline constexpr std::size_t kThresholdPanelGroup = 32;

/**
 * @brief AVX2 f32 specialization of the fused threshold outer driver.
 *
 * Iterates @c kThresholdChunkRows -sized row chunks of @p X. Within a chunk, each 8-row
 * M-tile packs an @c Mr x d A-strip via @c packA (full K range, no K-blocking), then for
 * every Nr-wide B-panel of @p Y calls @c gemmKernel8x6Avx2F32Threshold. The per-column
 * @c Y-norms are pre-packed once per call; @c X-row-norms are computed inline per M-tile and
 * staged in @c alignas(32) stack scratch so the kernel can issue aligned loads.
 *
 * Pool fan-out is over row chunks. Each chunk's emit calls are serialized through @p emit
 * from the worker that owns the chunk; emit implementations that need thread safety across
 * chunks are the caller's responsibility, the kernel itself touches no shared state beyond
 * the read-only packed Y and Y-norms.
 *
 * Precondition: @p X and @p Y must expose 32-byte aligned contiguous rows and @p d must not
 * exceed @c kThresholdMaxD; the public entry's dispatch checks both.
 *
 * @tparam Emit @c std::invocable<std::size_t, std::size_t> callable.
 * @param X        Data matrix (n x d), contiguous, 32-byte aligned.
 * @param Y        Point matrix (m x d), contiguous, 32-byte aligned.
 * @param xRowNormsSq Row-1 array of @c ||x_i||^2, length n.
 * @param yRowNormsSq Row-1 array of @c ||y_j||^2, length m.
 * @param radiusSq Non-negative squared radius.
 * @param pool     Parallelism injection; fans out over row chunks when @c shouldParallelize.
 * @param emit     Callback for each surviving @c (row, col) pair.
 */
template <class Emit>
inline void pairwiseThresholdOuterAvx2F32(const NDArray<float, 2, Layout::Contig> &X,
                                          const NDArray<float, 2, Layout::Contig> &Y,
                                          const NDArray<float, 1> &xRowNormsSq,
                                          const NDArray<float, 1> &yRowNormsSq, float radiusSq,
                                          Pool pool, Emit &&emit) {
  constexpr std::size_t kMr = kKernelMr<float>;
  constexpr std::size_t kNr = kKernelNr<float>;

  const std::size_t n = X.dim(0);
  const std::size_t m = Y.dim(0);
  const std::size_t d = X.dim(1);

  if (n == 0 || m == 0) {
    return;
  }

  CLUSTERING_ALWAYS_ASSERT(d <= kThresholdMaxD);

  const std::size_t nPanels = (m + kNr - 1) / kNr;
  const std::size_t bpanelSize = kNr * d;
  const std::size_t bpackSize = nPanels * bpanelSize;
  const std::size_t yNormsPaddedSize = nPanels * kNr;

  std::vector<float, ::clustering::detail::AlignedAllocator<float, 32>> bpackedStorage(bpackSize);
  std::vector<float, ::clustering::detail::AlignedAllocator<float, 32>> yNormsPackedStorage(
      yNormsPaddedSize);

  // packB wants B in K x N orientation (features x points); @p Y is (points x features), so
  // take its transpose view before describing the source. @c Y.t() is a borrowed MaybeStrided
  // view; describeMatrix preserves the strides so packB's scalar element access sees the right
  // Y[j][k_iter] per packed position.
  const auto yTransposed = Y.t();
  const auto yDesc = ::clustering::detail::describeMatrix(yTransposed);
  for (std::size_t p = 0; p < nPanels; ++p) {
    const std::size_t jBase = p * kNr;
    const std::size_t nc = (jBase + kNr <= m) ? kNr : (m - jBase);
    float *panelOut = bpackedStorage.data() + (p * bpanelSize);
    packB<float>(yDesc, 0, d, jBase, nc, panelOut);
  }
  packYColNormsSq<float>(yRowNormsSq.data(), m, yNormsPackedStorage.data());

  const std::size_t chunkCount = (n + kThresholdChunkRows - 1) / kThresholdChunkRows;

  auto runOneChunk = [&](std::size_t chunkIdx) {
    const std::size_t iChunkBase = chunkIdx * kThresholdChunkRows;
    const std::size_t chunkRows =
        (iChunkBase + kThresholdChunkRows <= n) ? kThresholdChunkRows : (n - iChunkBase);
    const std::size_t mTilesInChunk = (chunkRows + kMr - 1) / kMr;

    const auto xDesc = ::clustering::detail::describeMatrix(X);

    // Pre-pack every A-tile in the chunk and its row-norms once, then swap the (M-tile, panel)
    // loop nest so panels walk the inner axis while every M-tile sweeps the same panel group.
    // A panel group ~= 48 KiB stays L1-resident across the M-tile inner sweep, so the B bytes
    // we re-read go to L1 instead of L3 / cross-CCD fabric on the 9950X3D.
    alignas(32) std::array<float, kThresholdChunkRows * kThresholdMaxD> apPacked;
    alignas(32) std::array<float, kThresholdChunkRows> xNormsPacked;
    for (std::size_t tileIdx = 0; tileIdx < mTilesInChunk; ++tileIdx) {
      const std::size_t iBase = iChunkBase + (tileIdx * kMr);
      const std::size_t mc =
          (iBase + kMr <= iChunkBase + chunkRows) ? kMr : (iChunkBase + chunkRows - iBase);
      float *tileSlot = apPacked.data() + (tileIdx * kMr * d);
      packA<float>(xDesc, iBase, mc, 0, d, tileSlot);
      for (std::size_t r = 0; r < mc; ++r) {
        xNormsPacked[(tileIdx * kMr) + r] = xRowNormsSq(iBase + r);
      }
      for (std::size_t r = mc; r < kMr; ++r) {
        xNormsPacked[(tileIdx * kMr) + r] = 0.0F;
      }
    }

    for (std::size_t panelBase = 0; panelBase < nPanels; panelBase += kThresholdPanelGroup) {
      const std::size_t panelEnd = std::min(panelBase + kThresholdPanelGroup, nPanels);
      for (std::size_t tileIdx = 0; tileIdx < mTilesInChunk; ++tileIdx) {
        const std::size_t iBase = iChunkBase + (tileIdx * kMr);
        const std::size_t mc =
            (iBase + kMr <= iChunkBase + chunkRows) ? kMr : (iChunkBase + chunkRows - iBase);
        const float *tileA = apPacked.data() + (tileIdx * kMr * d);
        const float *tileNorms = xNormsPacked.data() + (tileIdx * kMr);

        for (std::size_t p = panelBase; p < panelEnd; ++p) {
          const std::size_t jBase = p * kNr;
          const std::size_t nc = (jBase + kNr <= m) ? kNr : (m - jBase);
          const float *bpPanel = bpackedStorage.data() + (p * bpanelSize);
          const float *normsPanel = yNormsPackedStorage.data() + (p * kNr);
          gemmKernel8x6Avx2F32Threshold(tileA, bpPanel, d, tileNorms, normsPanel, iBase, jBase, mc,
                                        nc, radiusSq, emit);
        }
      }
    }
  };

  if (pool.shouldParallelize(n * m, 64, 2) && pool.pool != nullptr) {
    pool.pool
        ->submit_blocks(std::size_t{0}, chunkCount,
                        [&](std::size_t lo, std::size_t hi) {
                          for (std::size_t c = lo; c < hi; ++c) {
                            runOneChunk(c);
                          }
                        })
        .wait();
  } else {
    for (std::size_t c = 0; c < chunkCount; ++c) {
      runOneChunk(c);
    }
  }
}

#endif // CLUSTERING_USE_AVX2

} // namespace clustering::math::detail
