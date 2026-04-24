#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

#include "clustering/ndarray.h"

#ifndef __SIZEOF_INT128__
#error "clustering::math::rng requires a compiler with __uint128_t support (GCC/Clang)."
#endif

namespace clustering::math {

/**
 * @brief 128-bit state for the PCG-XSL-RR 64-bit output generator (Melissa O'Neill).
 *
 * State and stream are held as @c __uint128_t. Seed with `seed(seed, stream)`; advance via
 * the free function `advanceState(rng)`. Reproducibility is load-bearing: the multiplier and
 * output function match pcg-cpp's @c pcg64 so identical seed + stream produce identical u64
 * streams across platforms that have @c __uint128_t.
 */
struct pcg64 {
  /// 128-bit generator state; advanced by every @c advanceState call.
  __uint128_t m_state = 0;
  /// Stream-encoded odd increment mixed into the LCG step.
  __uint128_t m_inc = 0;

  /**
   * @brief Initialize the generator per PCG's canonical seeding procedure.
   *
   * Matches Melissa O'Neill's @c pcg_basic reference: zero the state, set inc to
   * `(stream << 1)` | 1, advance once, add the user seed, advance once more.
   *
   * @param seedValue User-supplied seed mixed into the state after the stream selector.
   * @param stream Stream identifier; two generators with identical seed but different streams
   *        produce uncorrelated output sequences.
   */
  void seed(std::uint64_t seedValue, std::uint64_t stream = 0) noexcept {
    static constexpr __uint128_t kMultHi =
        (static_cast<__uint128_t>(2549297995355413924ULL) << 64) | 4865540595714422341ULL;
    m_state = 0;
    m_inc = (static_cast<__uint128_t>(stream) << 1U) | 1U;
    m_state = (m_state * kMultHi) + m_inc;
    m_state += seedValue;
    m_state = (m_state * kMultHi) + m_inc;
  }
};

/**
 * @brief Advance a @ref pcg64 one step and return the 64-bit XSL-RR output.
 *
 * @param rng In-out generator state; mutated by the call.
 * @return Next 64-bit output word.
 */
inline std::uint64_t advanceState(pcg64 &rng) noexcept {
  static constexpr __uint128_t kMult =
      (static_cast<__uint128_t>(2549297995355413924ULL) << 64) | 4865540595714422341ULL;
  const __uint128_t old = rng.m_state;
  rng.m_state = (old * kMult) + rng.m_inc;
  const auto rot = static_cast<std::uint64_t>(old >> 122);
  const auto xorshifted = static_cast<std::uint64_t>(old ^ (old >> 64));
  return (xorshifted >> rot) | (xorshifted << ((-rot) & 63U));
}

/**
 * @brief 256-bit state for Vigna & Blackman's xoshiro256** generator.
 *
 * Seed with `seed(seedValue)` (SplitMix64-diffused); advance via `advanceState(rng)`. The
 * canonical s={0,0,0,0} state is a fixed point, so @c seed always routes through SplitMix64
 * even for `seedValue == 0`.
 */
struct xoshiro256ss {
  /// Four 64-bit state words; SplitMix64-diffused at @c seed time.
  std::array<std::uint64_t, 4> m_s{0, 0, 0, 0};

  /**
   * @brief Initialize the four state words via SplitMix64 diffusion of a single 64-bit seed.
   *
   * SplitMix64's avalanche ensures every state word is well-mixed even when the user supplies
   * a small-integer seed like 0 or 1. Reference: https://prng.di.unimi.it/splitmix64.c.
   *
   * @param seedValue The 64-bit seed word.
   */
  void seed(std::uint64_t seedValue) noexcept {
    std::uint64_t z = seedValue;
    for (auto &word : m_s) {
      z += 0x9E3779B97F4A7C15ULL;
      std::uint64_t x = z;
      x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
      x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
      x = x ^ (x >> 31);
      word = x;
    }
  }
};

/**
 * @brief Advance a @ref xoshiro256ss one step and return the 64-bit output.
 *
 * @param rng In-out generator state; mutated by the call.
 * @return Next 64-bit output word.
 */
inline std::uint64_t advanceState(xoshiro256ss &rng) noexcept {
  const auto rotl = [](std::uint64_t x, int k) -> std::uint64_t {
    return (x << k) | (x >> (64 - k));
  };
  const std::uint64_t result = rotl(rng.m_s[1] * 5U, 7) * 9U;
  const std::uint64_t t = rng.m_s[1] << 17U;
  rng.m_s[2] ^= rng.m_s[0];
  rng.m_s[3] ^= rng.m_s[1];
  rng.m_s[1] ^= rng.m_s[2];
  rng.m_s[0] ^= rng.m_s[3];
  rng.m_s[2] ^= t;
  rng.m_s[3] = rotl(rng.m_s[3], 45);
  return result;
}

/**
 * @brief Draw a 32-bit unsigned integer uniformly at random from the full u32 range.
 *
 * Returns the top 32 bits of the 64-bit output; in generators like PCG64 these are
 * higher-quality than the low bits.
 */
template <class Rng> inline std::uint32_t randUniformU32(Rng &rng) noexcept {
  return static_cast<std::uint32_t>(advanceState(rng) >> 32U);
}

/**
 * @brief Draw a 64-bit unsigned integer uniformly at random from the full u64 range.
 */
template <class Rng> inline std::uint64_t randUniformU64(Rng &rng) noexcept {
  return advanceState(rng);
}

/**
 * @brief Draw a uniform variate in the half-open unit interval `[0, 1)`.
 *
 * For @c double, returns `(u >> 11) * 2^-53`, the canonical bias-free form that selects one
 * of `2^53` representable doubles in `[0, 1)`. For @c float, returns `(u >> 40) * 2^-24`
 * over `2^24` representable floats. Both forms round-trip without the @c 1.0 endpoint.
 *
 * @tparam T Either @c float or @c double.
 */
template <class T, class Rng> inline T randUnit(Rng &rng) noexcept {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                "randUnit<T> requires T to be float or double");
  if constexpr (std::is_same_v<T, double>) {
    return static_cast<double>(advanceState(rng) >> 11U) * 0x1.0p-53;
  } else {
    return static_cast<float>(advanceState(rng) >> 40U) * 0x1.0p-24F;
  }
}

/**
 * @brief Sample one category index proportionally to non-negative weights.
 *
 * Implements the k-means++ seeding primitive. Draws @c u in `[0, 1)`, scales to the weight
 * total, and walks the cumulative sum returning the first index whose prefix-sum is strictly
 * greater than @c u*total. The strict comparison matches `numpy.searchsorted(side='right')`
 * and avoids the classical `<= vs` `< off`-by-one that biases toward index 0 when a weight
 * equals the threshold exactly.
 *
 * @tparam T Element type of the weight array (@c float or @c double).
 * @tparam L Layout tag of the weight array; accepts both contiguous and strided.
 * @tparam Rng Generator type accepted by @ref advanceState.
 * @param weights Non-empty rank-1 array of non-negative weights with at least one positive entry.
 * @param rng In-out generator state.
 * @return Index in `[0, weights.dim(0))`.
 */
template <class T, Layout L, class Rng>
inline std::size_t weightedCategorical(const NDArray<T, 1, L> &weights, Rng &rng) noexcept {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                "weightedCategorical<T> requires T to be float or double");
  const std::size_t n = weights.dim(0);
  assert(n > 0 && "weightedCategorical requires at least one weight");

  T total = T{0};
  for (std::size_t i = 0; i < n; ++i) {
    const T w = weights(i);
    assert(w >= T{0} && "weightedCategorical requires non-negative weights");
    total += w;
  }
  assert(total > T{0} && "weightedCategorical requires at least one positive weight");

  const T u = randUnit<T>(rng) * total;
  T cumulative = T{0};
  std::size_t lastPositive = 0;
  for (std::size_t i = 0; i < n; ++i) {
    const T w = weights(i);
    cumulative += w;
    if (cumulative > u) {
      return i;
    }
    if (w > T{0}) {
      lastPositive = i;
    }
  }
  // Guard against floating-point drift pushing the final cumulative just below u*total: fall
  // back to the last index that actually contributed mass so we never return a zero-weight slot.
  return lastPositive;
}

/**
 * @brief Efraimidis-Spirakis weighted reservoir sampling (A-Exp variant, log-key form).
 *
 * For each item @c i draws `u_i` uniformly in `(0, 1)` and computes the key
 * `key_i = log(u_i) / w_i`; the @c k items with the largest keys are selected. The log form
 * is algebraically identical to the paper's `u_i^(1/w_i)` but avoids silent underflow when
 * `w_i` is small (the naive @c pow form collapses to @c 0 once `1/w_i` exceeds `~1075`,
 * biasing the selection).
 *
 * Reference: Efraimidis & Spirakis, "Weighted random sampling with a reservoir," IPL 97 (2006),
 * https://arxiv.org/pdf/1012.0256.
 *
 * @tparam T Element type of the weight array (@c float or @c double).
 * @tparam L Layout tag of the weight array; accepts both contiguous and strided.
 * @tparam Rng Generator type accepted by @ref advanceState.
 * @param weights Rank-1 array of strictly positive weights.
 * @param k Number of indices to select; must satisfy `k <= weights`.dim(0).
 * @param rng In-out generator state.
 * @param outIdx Output buffer of exactly @c k positions; filled with the sampled indices in
 *        unspecified order.
 */
template <class T, Layout L, class Rng>
inline void aExpjReservoir(const NDArray<T, 1, L> &weights, std::size_t k, Rng &rng,
                           std::span<std::size_t> outIdx) noexcept {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                "aExpjReservoir<T> requires T to be float or double");
  const std::size_t n = weights.dim(0);
  assert(outIdx.size() == k && "aExpjReservoir requires outIdx.size() == k");
  assert(k <= n && "aExpjReservoir requires k <= weights.dim(0)");
  if (k == 0) {
    return;
  }

  // Generate all keys, partial-sort by key descending, emit the top-k indices. O(n log n) in the
  // straight path; a size-k min-heap variant would trim this to O(n log k) when k << n.
  std::vector<std::pair<T, std::size_t>> keyed;
  keyed.reserve(n);
  for (std::size_t i = 0; i < n; ++i) {
    const T w = weights(i);
    assert(w > T{0} && "aExpjReservoir requires strictly positive weights");
    // randUnit draws from [0, 1); nudge zero away from the log singularity by resampling. In
    // double precision a single redraw is sufficient with probability 1 - 2^-53.
    T u = randUnit<T>(rng);
    while (u <= T{0}) {
      u = randUnit<T>(rng);
    }
    const T key = std::log(u) / w;
    keyed.emplace_back(key, i);
  }

  // Partial sort by key descending: the k largest keys bubble to the front.
  const auto cmp = [](const std::pair<T, std::size_t> &a,
                      const std::pair<T, std::size_t> &b) noexcept { return a.first > b.first; };
  std::partial_sort(keyed.begin(), keyed.begin() + static_cast<std::ptrdiff_t>(k), keyed.end(),
                    cmp);

  for (std::size_t i = 0; i < k; ++i) {
    outIdx[i] = keyed[i].second;
  }
}

} // namespace clustering::math
