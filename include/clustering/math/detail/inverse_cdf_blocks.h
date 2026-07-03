#pragma once

#include <cstddef>
#include <type_traits>

#include "clustering/math/detail/avx2_helpers.h"
#include "clustering/math/detail/avx2_reductions.h"

#ifdef CLUSTERING_USE_AVX2
#include <immintrin.h>
#endif

namespace clustering::math::detail {

/**
 * @brief Bank per-block sums of @p weights: `blockSums[b]` sums block @c b's elements.
 *
 * Blocks are @p blockElems wide with a short tail block. Each block reduces through
 * @ref sumReduceAvx2's fixed lane tree, so the banked totals depend only on the block
 * layout and never on which caller or worker touched a block.
 */
template <class T>
inline void bankWeightBlockSums(const T *weights, std::size_t n, std::size_t blockElems,
                                T *blockSums) noexcept {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                "bankWeightBlockSums: T must be float or double.");
  std::size_t b = 0;
  for (std::size_t lo = 0; lo < n; lo += blockElems, ++b) {
    const std::size_t len = (n - lo < blockElems) ? (n - lo) : blockElems;
    blockSums[b] = sumReduceAvx2(weights + lo, len);
  }
}

#ifdef CLUSTERING_USE_AVX2
/**
 * @brief Lane index of the first element in the 8-float block at @p w whose inclusive
 *        in-register prefix exceeds @p rem.
 *
 * The prefix builds with two per-lane shifted adds plus one cross-lane carry, so the walk
 * has no loop-carried dependency and a draw batch resolves at load throughput. Falls back
 * to the last lane when rounding pushes @p rem past the block's mass, mirroring
 * @ref inverseCdfPickInRange's end clamp.
 */
[[nodiscard]] inline std::size_t inverseCdfPickInBlock8F32(const float *w, float rem) noexcept {
  const __m256 v = _mm256_loadu_ps(w);
  const __m256 s1 =
      _mm256_add_ps(v, _mm256_castsi256_ps(_mm256_slli_si256(_mm256_castps_si256(v), 4)));
  const __m256 s2 =
      _mm256_add_ps(s1, _mm256_castsi256_ps(_mm256_slli_si256(_mm256_castps_si256(s1), 8)));
  // Carry the low lane's total into the high lane to finish the inclusive prefix.
  const __m256 lowDup = _mm256_permute2f128_ps(s2, s2, 0x00);
  const __m256 lowTot = _mm256_shuffle_ps(lowDup, lowDup, 0xFF);
  const __m256 prefix = _mm256_add_ps(s2, _mm256_blend_ps(_mm256_setzero_ps(), lowTot, 0xF0));
  const int mask = _mm256_movemask_ps(_mm256_cmp_ps(prefix, _mm256_set1_ps(rem), _CMP_GT_OQ));
  return (mask != 0) ? static_cast<std::size_t>(__builtin_ctz(static_cast<unsigned>(mask)))
                     : std::size_t{7};
}
#endif

/**
 * @brief Index of the first element in `[lo, hi)` whose inclusive prefix over @p weights
 *        exceeds @p rem.
 *
 * The 8-lane stride hops whole chunks while their mass stays at or below the draw, then the
 * scalar walk finishes inside the crossing chunk. Falls back to the last index when rounding
 * pushes the draw past the range's mass, mirroring the end-iterator clamp of an
 * `upper_bound` over a materialized prefix array.
 */
template <class T>
[[nodiscard]] inline std::size_t inverseCdfPickInRange(const T *weights, std::size_t lo,
                                                       std::size_t hi, T rem) noexcept {
  std::size_t i = lo;
  T run = T{0};
#ifdef CLUSTERING_USE_AVX2
  if constexpr (std::is_same_v<T, float>) {
    for (; i + 8 <= hi; i += 8) {
      const float chunk = horizontalSumAvx2(_mm256_loadu_ps(weights + i));
      if (!(run + chunk <= rem)) {
        break;
      }
      run += chunk;
    }
  }
#endif
  for (; i < hi; ++i) {
    run += weights[i];
    if (run > rem) {
      return i;
    }
  }
  return hi - 1;
}

} // namespace clustering::math::detail
