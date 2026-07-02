#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#ifdef CLUSTERING_USE_AVX2
#include <immintrin.h>
#endif

namespace clustering::math::detail {

#ifdef CLUSTERING_USE_AVX2

/**
 * @brief Fused 16x6 AVX2 int16 threshold filter microkernel over quantized panels.
 *
 * Filter-stage sibling of @c gemmKernel16x6Avx2F32Threshold: accumulates quantized dot
 * products with `vpmaddwd`, which retires sixteen multiply-accumulates per instruction
 * against the float kernel's eight, folds the quantized norms in float after a lane-wise
 * `vcvtdq2ps`, and emits every cell whose scaled distance passes @p qThresholdSq. The
 * threshold carries the quantization and conversion slack, so the surviving set is a
 * superset of the exact eps-neighbours and the caller re-checks each candidate in f32.
 *
 * Panel layout pairs adjacent K steps to feed `vpmaddwd`:
 *   - @p apPanel: per K pair @c p, 32 values `[r0k0 r0k1 .. r7k0 r7k1]` then rows 8-15,
 *     i.e. element `(r, k)` of the tile sits at `ap[p*32 + (r/8)*16 + (r%8)*2 + (k%2)]`.
 *   - @p bpPanel: per K pair @c p, 12 values `[c0k0 c0k1 c1k0 c1k1 ..]`, so one
 *     `vpbroadcastd` splats a column's K pair across all lanes.
 *
 * @param apPanel     Packed quantized A tile (32-byte aligned), 16 rows.
 * @param bpPanel     Packed quantized B panel (32-byte aligned), 6 columns.
 * @param kcPairs     Count of K pairs (quantized dimension rounded up to even, halved).
 * @param aRowNormsQ  Scaled squared norms of the 16 tile rows as float, 32-byte aligned.
 * @param bColNormsQ  Scaled squared norms of the 6 panel columns as float.
 * @param qThresholdSq Scaled squared radius plus quantization and float-fold slack.
 * @param emit        `void(std::size_t, std::size_t)` candidate callback.
 */
template <class Emit>
[[gnu::always_inline]] inline void gemmKernel16x6Avx2I16Threshold(
    const std::int16_t *apPanel, const std::int16_t *bpPanel, std::size_t kcPairs,
    const float *aRowNormsQ, const float *bColNormsQ, std::size_t rowBase, std::size_t colBase,
    std::size_t validRows, std::size_t validCols, float qThresholdSq, Emit &&emit) {
  constexpr std::size_t kMr = 8;
  constexpr std::size_t kNr = 6;
  constexpr std::size_t kRows16 = 16;

  const std::int16_t *__restrict__ apLocal = apPanel;
  const std::int16_t *__restrict__ bpLocal = bpPanel;
  const std::size_t pairsLocal = kcPairs;

  __m256i c0lo = _mm256_setzero_si256();
  __m256i c0hi = _mm256_setzero_si256();
  __m256i c1lo = _mm256_setzero_si256();
  __m256i c1hi = _mm256_setzero_si256();
  __m256i c2lo = _mm256_setzero_si256();
  __m256i c2hi = _mm256_setzero_si256();
  __m256i c3lo = _mm256_setzero_si256();
  __m256i c3hi = _mm256_setzero_si256();
  __m256i c4lo = _mm256_setzero_si256();
  __m256i c4hi = _mm256_setzero_si256();
  __m256i c5lo = _mm256_setzero_si256();
  __m256i c5hi = _mm256_setzero_si256();

  for (std::size_t p = 0; p < pairsLocal; ++p) {
    const __m256i aLo =
        _mm256_load_si256(reinterpret_cast<const __m256i *>(apLocal + (p * 2 * kRows16)));
    const __m256i aHi =
        _mm256_load_si256(reinterpret_cast<const __m256i *>(apLocal + (p * 2 * kRows16) + 16));
    const auto *bRow = reinterpret_cast<const std::int32_t *>(bpLocal + (p * 2 * kNr));
    const __m256i b0 = _mm256_set1_epi32(bRow[0]);
    c0lo = _mm256_add_epi32(c0lo, _mm256_madd_epi16(aLo, b0));
    c0hi = _mm256_add_epi32(c0hi, _mm256_madd_epi16(aHi, b0));
    const __m256i b1 = _mm256_set1_epi32(bRow[1]);
    c1lo = _mm256_add_epi32(c1lo, _mm256_madd_epi16(aLo, b1));
    c1hi = _mm256_add_epi32(c1hi, _mm256_madd_epi16(aHi, b1));
    const __m256i b2 = _mm256_set1_epi32(bRow[2]);
    c2lo = _mm256_add_epi32(c2lo, _mm256_madd_epi16(aLo, b2));
    c2hi = _mm256_add_epi32(c2hi, _mm256_madd_epi16(aHi, b2));
    const __m256i b3 = _mm256_set1_epi32(bRow[3]);
    c3lo = _mm256_add_epi32(c3lo, _mm256_madd_epi16(aLo, b3));
    c3hi = _mm256_add_epi32(c3hi, _mm256_madd_epi16(aHi, b3));
    const __m256i b4 = _mm256_set1_epi32(bRow[4]);
    c4lo = _mm256_add_epi32(c4lo, _mm256_madd_epi16(aLo, b4));
    c4hi = _mm256_add_epi32(c4hi, _mm256_madd_epi16(aHi, b4));
    const __m256i b5 = _mm256_set1_epi32(bRow[5]);
    c5lo = _mm256_add_epi32(c5lo, _mm256_madd_epi16(aLo, b5));
    c5hi = _mm256_add_epi32(c5hi, _mm256_madd_epi16(aHi, b5));
  }

  const __m256 xNormsLo = _mm256_load_ps(aRowNormsQ);
  const __m256 xNormsHi = _mm256_load_ps(aRowNormsQ + kMr);
  const __m256 neg2 = _mm256_set1_ps(-2.0F);
  const __m256 thresholdVec = _mm256_set1_ps(qThresholdSq);

  auto foldAndMask16 = [&](__m256i accLo, __m256i accHi,
                           std::size_t colOffset) noexcept -> std::uint16_t {
    const __m256 yNorm = _mm256_set1_ps(bColNormsQ[colOffset]);
    const __m256 dotLo = _mm256_cvtepi32_ps(accLo);
    const __m256 dotHi = _mm256_cvtepi32_ps(accHi);
    const __m256 distLo = _mm256_fmadd_ps(dotLo, neg2, _mm256_add_ps(xNormsLo, yNorm));
    const __m256 distHi = _mm256_fmadd_ps(dotHi, neg2, _mm256_add_ps(xNormsHi, yNorm));
    const auto maskLo =
        static_cast<unsigned>(_mm256_movemask_ps(_mm256_cmp_ps(distLo, thresholdVec, _CMP_LE_OQ)));
    const auto maskHi =
        static_cast<unsigned>(_mm256_movemask_ps(_mm256_cmp_ps(distHi, thresholdVec, _CMP_LE_OQ)));
    return static_cast<std::uint16_t>(maskLo | (maskHi << kMr));
  };

  std::array<std::uint16_t, kNr> masks{};
  if (validCols > 0) {
    masks[0] = foldAndMask16(c0lo, c0hi, 0);
  }
  if (validCols > 1) {
    masks[1] = foldAndMask16(c1lo, c1hi, 1);
  }
  if (validCols > 2) {
    masks[2] = foldAndMask16(c2lo, c2hi, 2);
  }
  if (validCols > 3) {
    masks[3] = foldAndMask16(c3lo, c3hi, 3);
  }
  if (validCols > 4) {
    masks[4] = foldAndMask16(c4lo, c4hi, 4);
  }
  if (validCols > 5) {
    masks[5] = foldAndMask16(c5lo, c5hi, 5);
  }

  const auto rowMask = (validRows >= kRows16) ? std::uint16_t{0xFFFF}
                                              : static_cast<std::uint16_t>((1U << validRows) - 1U);
  const auto anyMask =
      static_cast<std::uint16_t>(masks[0] | masks[1] | masks[2] | masks[3] | masks[4] | masks[5]);
  if ((anyMask & rowMask) == 0) {
    return;
  }
  for (std::size_t c = 0; c < validCols; ++c) {
    auto m = static_cast<std::uint16_t>(masks[c] & rowMask);
    while (m != 0) {
      const int bit = __builtin_ctz(static_cast<unsigned>(m));
      emit(rowBase + static_cast<std::size_t>(bit), colBase + c);
      m = static_cast<std::uint16_t>(m & (m - 1));
    }
  }
}

#endif // CLUSTERING_USE_AVX2

} // namespace clustering::math::detail
