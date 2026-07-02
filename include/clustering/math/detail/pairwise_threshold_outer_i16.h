#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "clustering/always_assert.h"
#include "clustering/math/detail/pairwise_threshold_outer.h"
#include "clustering/math/thread.h"
#include "clustering/ndarray.h"

#ifdef CLUSTERING_USE_AVX2
#include <immintrin.h>

#include "clustering/math/detail/gemm_kernel_avx2_i16_threshold.h"
#endif

namespace clustering::math::detail {

#ifdef CLUSTERING_USE_AVX2

/// Fraction of the squared radius the quantization slack may reach before the int16 filter
/// stops paying: past it the candidate band swallows the pruning the filter exists to do.
inline constexpr double kI16FilterBoundCeiling = 0.25;

/// Diff-form squared distance over unaligned contiguous rows; the recheck's fast verdict.
[[gnu::always_inline]] inline float sqDistRowsF32(const float *a, const float *b,
                                                  std::size_t d) noexcept {
  __m256 acc = _mm256_setzero_ps();
  std::size_t k = 0;
  for (; k + 8 <= d; k += 8) {
    const __m256 diff = _mm256_sub_ps(_mm256_loadu_ps(a + k), _mm256_loadu_ps(b + k));
    acc = _mm256_fmadd_ps(diff, diff, acc);
  }
  __m128 lo = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
  lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
  lo = _mm_add_ss(lo, _mm_shuffle_ps(lo, lo, 1));
  float sum = _mm_cvtss_f32(lo);
  for (; k < d; ++k) {
    const float diff = a[k] - b[k];
    sum += diff * diff;
  }
  return sum;
}

/**
 * @brief Symmetric eps-graph sweep through an int16 filter with exact f32 recheck.
 *
 * Quantizes @p X once to int16 with a scale that keeps every `vpmaddwd` accumulation inside
 * int32 by construction, sweeps the upper triangle with the quantized kernel against a
 * threshold widened by the quantization and float-fold slack, and re-checks each surviving
 * candidate against the original rows before forwarding it to @p emit. The candidate set is
 * a superset of the true eps-neighbours, so the recheck makes the emitted set exact; pairs
 * whose diff-form distance lands inside a narrow rounding band of the radius are resolved
 * with the same fused-fold arithmetic the f32 kernel lane applies, keeping boundary verdicts
 * consistent with the unfiltered path.
 *
 * Returns @c false without emitting when the quantization slack would exceed
 * @c kI16FilterBoundCeiling of the squared radius (degenerate scales on wide-range data);
 * the caller then runs the f32 driver instead.
 */
template <class Emit>
inline bool
pairwiseThresholdOuterAvx2I16FilteredSymmetric(const NDArray<float, 2, Layout::Contig> &X,
                                               const NDArray<float, 1> &xRowNormsSq, float radiusSq,
                                               Pool pool, Emit &&emit) {
  constexpr std::size_t kNr = 6;
  constexpr std::size_t kTileRows = 16;
  constexpr std::size_t kSymmetricChunkRows = 64;

  const std::size_t n = X.dim(0);
  const std::size_t d = X.dim(1);
  if (n == 0) {
    return true;
  }
  CLUSTERING_ALWAYS_ASSERT(d <= kThresholdMaxD);

  // ---- Scale selection: coordinate range for the int16 lanes, row norm for the int32
  // accumulator. The norm constant sits under sqrt(2^31) with room for the per-coordinate
  // rounding that pushes a quantized row norm past S * ||x||.
  const float *xData = X.data();
  double maxRowNormSq = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    maxRowNormSq = std::max(maxRowNormSq, static_cast<double>(xRowNormsSq(i)));
  }
  std::vector<float> workerMaxAbs(pool.workerCount(), 0.0F);
  pool.parallelForBlocks(std::size_t{0}, n, std::size_t{0}, [&](std::size_t lo, std::size_t hi) {
    float local = 0.0F;
    for (std::size_t i = lo * d; i < hi * d; ++i) {
      local = std::max(local, std::fabs(xData[i]));
    }
    float &slot = workerMaxAbs[Pool::workerIndex()];
    slot = std::max(slot, local);
  });
  float maxAbs = 0.0F;
  for (const float m : workerMaxAbs) {
    maxAbs = std::max(maxAbs, m);
  }
  if (!(maxAbs > 0.0F) || !(maxRowNormSq > 0.0)) {
    return false;
  }
  const double scale =
      std::min(32767.0 / static_cast<double>(maxAbs), 46000.0 / std::sqrt(maxRowNormSq));
  const double eps = std::sqrt(static_cast<double>(radiusSq));
  const double quantSlack = (2.0 * std::sqrt(static_cast<double>(d)) * eps) / scale +
                            static_cast<double>(d) / (scale * scale);
  if (quantSlack > kI16FilterBoundCeiling * static_cast<double>(radiusSq)) {
    return false;
  }
  const auto scaleF = static_cast<float>(scale);
  const float qThresholdSq =
      static_cast<float>((static_cast<double>(radiusSq) + quantSlack) * scale * scale + 8192.0);

  // ---- Quantize rows and take exact quantized norms. ----
  const std::size_t kcPairs = (d + 1) / 2;
  const std::size_t kcEven = kcPairs * 2;
  std::vector<std::int16_t, ::clustering::detail::AlignedAllocator<std::int16_t, 32>> quantized(
      n * kcEven);
  std::vector<float, ::clustering::detail::AlignedAllocator<float, 32>> quantNorms(n);
  const auto quantizeRows = [&](std::size_t lo, std::size_t hi) {
    for (std::size_t i = lo; i < hi; ++i) {
      std::int16_t *q = quantized.data() + (i * kcEven);
      const float *src = xData + (i * d);
      std::int64_t norm = 0;
      for (std::size_t k = 0; k < d; ++k) {
        const auto v = static_cast<std::int32_t>(std::lrintf(src[k] * scaleF));
        q[k] = static_cast<std::int16_t>(v);
        norm += static_cast<std::int64_t>(v) * static_cast<std::int64_t>(v);
      }
      if (d < kcEven) {
        q[d] = 0;
      }
      quantNorms[i] = static_cast<float>(norm);
    }
  };
  pool.parallelForBlocks(std::size_t{0}, n, std::size_t{0}, quantizeRows);

  // ---- Pack B panels: per K pair, six interleaved column pairs so one broadcast feeds the
  // kernel; padded columns carry +inf norms so they can never pass the threshold.
  const std::size_t nPanels = (n + kNr - 1) / kNr;
  const std::size_t bpanelSize = kNr * kcEven;
  std::vector<std::int16_t, ::clustering::detail::AlignedAllocator<std::int16_t, 32>>
      bpackedStorage(nPanels * bpanelSize, 0);
  std::vector<float, ::clustering::detail::AlignedAllocator<float, 32>> bNormsPacked(
      nPanels * kNr, std::numeric_limits<float>::infinity());
  const auto packPanels = [&](std::size_t lo, std::size_t hi) {
    for (std::size_t p = lo; p < hi; ++p) {
      std::int16_t *panelOut = bpackedStorage.data() + (p * bpanelSize);
      const std::size_t jBase = p * kNr;
      const std::size_t nc = std::min(kNr, n - jBase);
      for (std::size_t c = 0; c < nc; ++c) {
        const std::int16_t *q = quantized.data() + ((jBase + c) * kcEven);
        bNormsPacked[(p * kNr) + c] = quantNorms[jBase + c];
        for (std::size_t pr = 0; pr < kcPairs; ++pr) {
          panelOut[(pr * 2 * kNr) + (c * 2) + 0] = q[pr * 2];
          panelOut[(pr * 2 * kNr) + (c * 2) + 1] = q[(pr * 2) + 1];
        }
      }
    }
  };
  pool.parallelForBlocks(std::size_t{0}, nPanels, std::size_t{0}, packPanels);

  // ---- Recheck: exact f32 verdict per candidate. Ambiguous pairs inside the rounding band
  // are resolved with the fused-fold lane arithmetic so boundary decisions match the f32
  // kernel's; everything else the diff form settles outright.
  auto recheckEmit = [&](std::size_t row, std::size_t col) {
    const float *a = xData + (row * d);
    const float *b = xData + (col * d);
    const float ds = sqDistRowsF32(a, b, d);
    const float xn = xRowNormsSq(row);
    const float yn = xRowNormsSq(col);
    const float band = (xn + yn + radiusSq) * (static_cast<float>(d / 2 + 8) * 1.2e-7F);
    if (ds <= radiusSq - band) {
      emit(row, col);
      return;
    }
    if (ds > radiusSq + band) {
      return;
    }
    float dot = 0.0F;
    for (std::size_t k = 0; k < d; ++k) {
      dot = std::fmaf(a[k], b[k], dot);
    }
    if (std::fmaf(dot, -2.0F, xn + yn) <= radiusSq) {
      emit(row, col);
    }
  };

  // ---- Chunked triangular sweep; mirrors the f32 symmetric driver's fan-out shape. ----
  const std::size_t chunkCount = (n + kSymmetricChunkRows - 1) / kSymmetricChunkRows;
  const std::size_t panelGroup = (d >= 128) ? std::size_t{16} : kThresholdPanelGroup;

  auto runOneChunk = [&](std::size_t chunkIdx) {
    const std::size_t iChunkBase = chunkIdx * kSymmetricChunkRows;
    const std::size_t chunkRows = std::min(kSymmetricChunkRows, n - iChunkBase);
    const std::size_t tilesInChunk = (chunkRows + kTileRows - 1) / kTileRows;

    // Pack the chunk's A tiles in kernel layout; padded rows stay zero and their norms stay
    // zero, so the row mask alone keeps them out of the emitted set.
    alignas(32) std::array<std::int16_t, kSymmetricChunkRows * kThresholdMaxD> apackTile{};
    alignas(32) std::array<float, kSymmetricChunkRows> aNorms{};
    for (std::size_t t = 0; t < tilesInChunk; ++t) {
      std::int16_t *tileOut = apackTile.data() + (t * kTileRows * kcEven);
      for (std::size_t r = 0; r < kTileRows; ++r) {
        const std::size_t row = iChunkBase + (t * kTileRows) + r;
        if (row >= n) {
          break;
        }
        aNorms[(t * kTileRows) + r] = quantNorms[row];
        const std::int16_t *q = quantized.data() + (row * kcEven);
        const std::size_t half = r / 8;
        const std::size_t rr = r % 8;
        for (std::size_t pr = 0; pr < kcPairs; ++pr) {
          tileOut[(pr * 2 * kTileRows) + (half * kTileRows) + (rr * 2) + 0] = q[pr * 2];
          tileOut[(pr * 2 * kTileRows) + (half * kTileRows) + (rr * 2) + 1] = q[(pr * 2) + 1];
        }
      }
    }

    for (std::size_t panelBase = 0; panelBase < nPanels; panelBase += panelGroup) {
      const std::size_t panelEnd = std::min(panelBase + panelGroup, nPanels);
      for (std::size_t t = 0; t < tilesInChunk; ++t) {
        const std::size_t iBase = iChunkBase + (t * kTileRows);
        const std::size_t mc = std::min(kTileRows, iChunkBase + chunkRows - iBase);
        const std::int16_t *tileA = apackTile.data() + (t * kTileRows * kcEven);
        const float *tileNorms = aNorms.data() + (t * kTileRows);

        const std::size_t pSkip = iBase / kNr;
        const std::size_t pStart = std::max(panelBase, pSkip);
        if (pStart >= panelEnd) {
          continue;
        }
        const std::size_t pStrictUpper = (iBase + mc + kNr - 1) / kNr;

        auto filteredEmit = [&](std::size_t row, std::size_t col) {
          if (row <= col) {
            recheckEmit(row, col);
          }
        };

        const std::size_t pDiagEnd = std::clamp(pStrictUpper, pStart, panelEnd);
        for (std::size_t p = pStart; p < pDiagEnd; ++p) {
          const std::size_t jBase = p * kNr;
          const std::size_t nc = std::min(kNr, n - jBase);
          gemmKernel16x6Avx2I16Threshold(tileA, bpackedStorage.data() + (p * bpanelSize), kcPairs,
                                         tileNorms, bNormsPacked.data() + (p * kNr), iBase, jBase,
                                         mc, nc, qThresholdSq, filteredEmit);
        }
        for (std::size_t p = pDiagEnd; p < panelEnd; ++p) {
          const std::size_t jBase = p * kNr;
          const std::size_t nc = std::min(kNr, n - jBase);
          gemmKernel16x6Avx2I16Threshold(tileA, bpackedStorage.data() + (p * bpanelSize), kcPairs,
                                         tileNorms, bNormsPacked.data() + (p * kNr), iBase, jBase,
                                         mc, nc, qThresholdSq, recheckEmit);
        }
      }
    }
  };

  if (pool.shouldParallelize(n * n / 2, 64, 2)) {
    pool.parallelForChunks<ThresholdChunkHints>(chunkCount, [&](std::size_t c) { runOneChunk(c); });
  } else {
    for (std::size_t c = 0; c < chunkCount; ++c) {
      runOneChunk(c);
    }
  }
  return true;
}

#endif // CLUSTERING_USE_AVX2

} // namespace clustering::math::detail
