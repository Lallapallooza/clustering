#include <gtest/gtest.h>

#include <cstddef>
#include <vector>

#include "clustering/math/detail/inverse_cdf_blocks.h"
#include "clustering/math/rng.h"

using clustering::math::detail::bankWeightBlockSums;
using clustering::math::detail::inverseCdfPickInRange;

namespace {

// Integer-valued weights keep every partial sum exactly representable, so the banked
// two-level pick and a scalar reference walk must agree index-for-index.
std::vector<float> integerWeights(std::size_t n, std::uint64_t seed) {
  clustering::math::pcg64 rng;
  rng.seed(seed);
  std::vector<float> w(n);
  for (float &v : w) {
    v = static_cast<float>(1 + (clustering::math::randUniformU64(rng) % 7));
  }
  return w;
}

std::size_t referencePick(const std::vector<float> &w, float u) {
  float run = 0.0F;
  for (std::size_t i = 0; i < w.size(); ++i) {
    run += w[i];
    if (run > u) {
      return i;
    }
  }
  return w.size() - 1;
}

} // namespace

TEST(BankWeightBlockSums, MatchesPerBlockTotalsIncludingShortTail) {
  const std::size_t n = 173;
  const std::size_t blockElems = 32;
  const auto w = integerWeights(n, 7U);
  const std::size_t blocks = (n + blockElems - 1) / blockElems;
  std::vector<float> sums(blocks, -1.0F);
  bankWeightBlockSums(w.data(), n, blockElems, sums.data());
  for (std::size_t b = 0; b < blocks; ++b) {
    float expected = 0.0F;
    const std::size_t lo = b * blockElems;
    const std::size_t hi = (n - lo < blockElems) ? n : (lo + blockElems);
    for (std::size_t i = lo; i < hi; ++i) {
      expected += w[i];
    }
    EXPECT_EQ(sums[b], expected) << "block " << b;
  }
}

TEST(InverseCdfPickInRange, MatchesScalarReferenceOverFullRange) {
  const std::size_t n = 200;
  const auto w = integerWeights(n, 11U);
  float total = 0.0F;
  for (const float v : w) {
    total += v;
  }
  for (float u = 0.0F; u < total; u += 0.37F) {
    EXPECT_EQ(inverseCdfPickInRange(w.data(), std::size_t{0}, n, u), referencePick(w, u))
        << "u=" << u;
  }
}

TEST(InverseCdfPickInRange, ClampsDrawBeyondRangeMass) {
  const std::vector<float> w{1.0F, 2.0F, 3.0F};
  EXPECT_EQ(inverseCdfPickInRange(w.data(), std::size_t{0}, w.size(), 100.0F), w.size() - 1);
}

TEST(InverseCdfPickInRange, RespectsSubrangeBounds) {
  const std::size_t n = 96;
  const auto w = integerWeights(n, 13U);
  const std::size_t lo = 32;
  const std::size_t hi = 64;
  float mass = 0.0F;
  for (std::size_t i = lo; i < hi; ++i) {
    mass += w[i];
  }
  for (float rem = 0.0F; rem < mass; rem += 0.51F) {
    const std::size_t pick = inverseCdfPickInRange(w.data(), lo, hi, rem);
    EXPECT_GE(pick, lo);
    EXPECT_LT(pick, hi);
    float run = 0.0F;
    std::size_t expected = hi - 1;
    for (std::size_t i = lo; i < hi; ++i) {
      run += w[i];
      if (run > rem) {
        expected = i;
        break;
      }
    }
    EXPECT_EQ(pick, expected) << "rem=" << rem;
  }
}

#ifdef CLUSTERING_USE_AVX2
TEST(InverseCdfPickInBlock8, MatchesScalarWalkOnIntegerWeights) {
  const auto w = integerWeights(8, 23U);
  float mass = 0.0F;
  for (const float v : w) {
    mass += v;
  }
  for (float rem = 0.0F; rem < mass; rem += 0.29F) {
    float run = 0.0F;
    std::size_t expected = 7;
    for (std::size_t i = 0; i < 8; ++i) {
      run += w[i];
      if (run > rem) {
        expected = i;
        break;
      }
    }
    EXPECT_EQ(clustering::math::detail::inverseCdfPickInBlock8F32(w.data(), rem), expected)
        << "rem=" << rem;
  }
}

TEST(InverseCdfPickInBlock8, ClampsDrawBeyondBlockMass) {
  const auto w = integerWeights(8, 29U);
  EXPECT_EQ(clustering::math::detail::inverseCdfPickInBlock8F32(w.data(), 1e9F), 7U);
}
#endif

// Composition check for the seeder's two-level path: upper_bound over the banked block
// prefix plus an in-block walk lands on the same index as one scalar walk over the raw
// weights, for draws spanning block interiors and exact block boundaries.
TEST(InverseCdfBlocks, TwoLevelPickMatchesFlatReference) {
  const std::size_t n = 517;
  const std::size_t blockElems = 64;
  const auto w = integerWeights(n, 17U);
  const std::size_t blocks = (n + blockElems - 1) / blockElems;
  std::vector<float> prefix(blocks);
  bankWeightBlockSums(w.data(), n, blockElems, prefix.data());
  float running = 0.0F;
  for (std::size_t b = 0; b < blocks; ++b) {
    running += prefix[b];
    prefix[b] = running;
  }
  const float total = prefix[blocks - 1];
  for (float u = 0.0F; u < total; u += 0.83F) {
    std::size_t b = 0;
    std::size_t len = blocks + 1;
    while (len > 1) {
      const std::size_t half = len / 2;
      b += (prefix[b + half - 1] <= u) ? half : 0;
      len -= half;
    }
    if (b >= blocks) {
      b = blocks - 1;
    }
    const float rem = u - ((b > 0) ? prefix[b - 1] : 0.0F);
    const std::size_t lo = b * blockElems;
    const std::size_t hi = (n - lo < blockElems) ? n : (lo + blockElems);
    EXPECT_EQ(inverseCdfPickInRange(w.data(), lo, hi, rem), referencePick(w, u)) << "u=" << u;
  }
}
