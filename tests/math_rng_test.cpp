#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <set>
#include <span>

#include "clustering/math/rng.h"
#include "clustering/ndarray.h"

using clustering::NDArray;
using clustering::math::advanceState;
using clustering::math::aExpjReservoir;
using clustering::math::pcg64;
using clustering::math::randUniformU32;
using clustering::math::randUniformU64;
using clustering::math::randUnit;
using clustering::math::weightedCategorical;
using clustering::math::xoshiro256ss;

TEST(PCG64, AdvanceStateIsDeterministic) {
  pcg64 a;
  pcg64 b;
  a.seed(12345ULL, 0ULL);
  b.seed(12345ULL, 0ULL);
  for (int i = 0; i < 10; ++i) {
    EXPECT_EQ(advanceState(a), advanceState(b)) << "mismatch at step " << i;
  }
}

TEST(PCG64, DifferentSeedsProduceDifferentStreams) {
  pcg64 a;
  pcg64 b;
  a.seed(1ULL, 0ULL);
  b.seed(2ULL, 0ULL);
  bool anyDiffers = false;
  for (int i = 0; i < 10; ++i) {
    if (advanceState(a) != advanceState(b)) {
      anyDiffers = true;
    }
  }
  EXPECT_TRUE(anyDiffers);
}

TEST(PCG64, MatchesReferenceVector) {
  // Reference vector generated locally from the canonical PCG128-XSL-RR algorithm
  // (Melissa O'Neill, pcg-cpp), seed = 42, stream = 0.
  const std::array<std::uint64_t, 5> expected = {0x63b4a3a813ce70faULL, 0x3f042f649083f6aaULL,
                                                 0x649af5df021045f2ULL, 0x1b7f129837b93984ULL,
                                                 0x8306f9f6d118d044ULL};
  pcg64 rng;
  rng.seed(42ULL, 0ULL);
  for (std::size_t i = 0; i < expected.size(); ++i) {
    EXPECT_EQ(advanceState(rng), expected[i]) << "mismatch at index " << i;
  }
}

TEST(Xoshiro256ss, AdvanceStateIsDeterministic) {
  xoshiro256ss a;
  xoshiro256ss b;
  a.seed(0xDEADBEEFCAFEBABEULL);
  b.seed(0xDEADBEEFCAFEBABEULL);
  for (int i = 0; i < 10; ++i) {
    EXPECT_EQ(advanceState(a), advanceState(b));
  }
}

TEST(Xoshiro256ss, DifferentSeedsProduceDifferentStreams) {
  xoshiro256ss a;
  xoshiro256ss b;
  a.seed(1ULL);
  b.seed(2ULL);
  bool anyDiffers = false;
  for (int i = 0; i < 10; ++i) {
    if (advanceState(a) != advanceState(b)) {
      anyDiffers = true;
    }
  }
  EXPECT_TRUE(anyDiffers);
}

TEST(Xoshiro256ss, MatchesPublishedReferenceVector) {
  // Reference vector generated locally from xoshiro256starstar.c (Vigna & Blackman,
  // https://prng.di.unimi.it/xoshiro256starstar.c), state seeded via SplitMix64
  // from seed = 0x123456789ABCDEFULL.
  const std::array<std::uint64_t, 5> expected = {0xa2c2a42038d4ec3dULL, 0x05fc25d0738e7b0fULL,
                                                 0x625e7bff938e701eULL, 0x1ba4ddc6fe2b5726ULL,
                                                 0xdf0a2482ac9254cfULL};
  xoshiro256ss rng;
  rng.seed(0x123456789ABCDEFULL);
  for (std::size_t i = 0; i < expected.size(); ++i) {
    EXPECT_EQ(advanceState(rng), expected[i]) << "mismatch at index " << i;
  }
}

TEST(RandUniform, U32TopBitsAreNotStuck) {
  pcg64 rng;
  rng.seed(7ULL, 0ULL);
  std::uint32_t accumulator = 0;
  for (int i = 0; i < 1000; ++i) {
    accumulator |= randUniformU32(rng);
  }
  // Across 1000 draws every bit of the top half should flip at least once.
  EXPECT_EQ(accumulator, 0xFFFFFFFFU);
}

TEST(RandUniform, U64IsWellSpread) {
  pcg64 rng;
  rng.seed(7ULL, 0ULL);
  std::uint64_t accumulator = 0;
  for (int i = 0; i < 1000; ++i) {
    accumulator |= randUniformU64(rng);
  }
  EXPECT_EQ(accumulator, 0xFFFFFFFFFFFFFFFFULL);
}

TEST(RandUnit, RangeF32IsHalfOpenUnit) {
  xoshiro256ss rng;
  rng.seed(101ULL);
  float minSeen = 1.0F;
  float maxSeen = 0.0F;
  for (int i = 0; i < 10000; ++i) {
    const auto u = randUnit<float>(rng);
    EXPECT_GE(u, 0.0F);
    EXPECT_LT(u, 1.0F);
    minSeen = std::min(minSeen, u);
    maxSeen = std::max(maxSeen, u);
  }
  EXPECT_LT(minSeen, 0.01F);
  EXPECT_GT(maxSeen, 0.99F);
}

TEST(RandUnit, RangeF64IsHalfOpenUnit) {
  xoshiro256ss rng;
  rng.seed(202ULL);
  double minSeen = 1.0;
  double maxSeen = 0.0;
  for (int i = 0; i < 10000; ++i) {
    const auto u = randUnit<double>(rng);
    EXPECT_GE(u, 0.0);
    EXPECT_LT(u, 1.0);
    minSeen = std::min(minSeen, u);
    maxSeen = std::max(maxSeen, u);
  }
  EXPECT_LT(minSeen, 0.01);
  EXPECT_GT(maxSeen, 0.99);
}

TEST(WeightedCategorical, UniformWeightsGiveNearUniformFrequencies) {
  constexpr std::size_t kCategories = 10;
  constexpr std::size_t kSamples = 100000;
  NDArray<double, 1> weights({kCategories});
  for (std::size_t i = 0; i < kCategories; ++i) {
    weights(i) = 1.0;
  }
  pcg64 rng;
  rng.seed(42ULL, 0ULL);
  std::array<std::size_t, kCategories> counts{};
  for (std::size_t s = 0; s < kSamples; ++s) {
    const std::size_t idx = weightedCategorical(weights, rng);
    ASSERT_LT(idx, kCategories);
    counts[idx]++;
  }
  // Expected ~10,000 per bucket; 3 sigma of a binomial(N=100k, p=0.1) is ~285.
  const double expected = static_cast<double>(kSamples) / static_cast<double>(kCategories);
  for (std::size_t i = 0; i < kCategories; ++i) {
    const double dev = std::fabs(static_cast<double>(counts[i]) - expected);
    EXPECT_LT(dev, expected * 0.05) << "bucket " << i << " saw " << counts[i];
  }
}

TEST(WeightedCategorical, HeavilyBiasedPicksHighWeightIndex) {
  NDArray<double, 1> weights({4});
  weights(0) = 0.001;
  weights(1) = 0.001;
  weights(2) = 1.0;
  weights(3) = 0.001;
  pcg64 rng;
  rng.seed(17ULL, 0ULL);
  constexpr std::size_t kSamples = 1000;
  std::size_t hits = 0;
  for (std::size_t s = 0; s < kSamples; ++s) {
    if (weightedCategorical(weights, rng) == 2U) {
      ++hits;
    }
  }
  EXPECT_GT(hits, static_cast<std::size_t>(kSamples * 0.80));
}

TEST(WeightedCategorical, SingleNonzeroWeightAlwaysPicksIt) {
  NDArray<double, 1> weights({4});
  weights(0) = 0.0;
  weights(1) = 0.0;
  weights(2) = 5.0;
  weights(3) = 0.0;
  pcg64 rng;
  rng.seed(99ULL, 0ULL);
  for (int s = 0; s < 100; ++s) {
    EXPECT_EQ(weightedCategorical(weights, rng), 2U);
  }
}

TEST(WeightedCategorical, StridedInputsWork) {
  // Build a 4x3 matrix and take column 1 as a strided rank-1 view.
  NDArray<double, 2> table({4, 3});
  const std::array<double, 4> columnValues = {0.0, 0.0, 5.0, 0.0};
  for (std::size_t i = 0; i < 4; ++i) {
    for (std::size_t j = 0; j < 3; ++j) {
      table[i][j] = (j == 1U) ? columnValues[i] : 1000.0;
    }
  }
  const auto col = table.col(1);
  EXPECT_EQ(col.dim(0), 4U);
  pcg64 rng;
  rng.seed(5ULL, 0ULL);
  for (int s = 0; s < 50; ++s) {
    EXPECT_EQ(weightedCategorical(col, rng), 2U);
  }
}

TEST(AExpjReservoir, UniformWeightsUniformFrequencies) {
  constexpr std::size_t kN = 100;
  constexpr std::size_t kK = 10;
  constexpr std::size_t kTrials = 1000;
  NDArray<double, 1> weights({kN});
  for (std::size_t i = 0; i < kN; ++i) {
    weights(i) = 1.0;
  }
  pcg64 rng;
  rng.seed(2024ULL, 0ULL);
  std::array<std::size_t, kN> counts{};
  std::array<std::size_t, kK> outBuf{};
  for (std::size_t t = 0; t < kTrials; ++t) {
    aExpjReservoir(weights, kK, rng, std::span<std::size_t>(outBuf.data(), kK));
    for (const std::size_t idx : outBuf) {
      ASSERT_LT(idx, kN);
      counts[idx]++;
    }
  }
  // Each index's expected count = kTrials * kK / kN = 100. Hypergeometric variance per trial
  // caps below the binomial bound B(kTrials, kK/kN); 4 sigma ~= 38. Use a permissive 50 to
  // avoid flakes while still being tight enough to catch a ~2x bias from the broken form.
  const double expected = static_cast<double>(kTrials * kK) / static_cast<double>(kN);
  for (std::size_t i = 0; i < kN; ++i) {
    const double dev = std::fabs(static_cast<double>(counts[i]) - expected);
    EXPECT_LT(dev, 50.0) << "index " << i << " count=" << counts[i];
  }
}

TEST(AExpjReservoir, WeightOneHeavilyDominates) {
  NDArray<double, 1> weights({5});
  weights(0) = 0.01;
  weights(1) = 0.01;
  weights(2) = 0.01;
  weights(3) = 0.01;
  weights(4) = 100.0;
  pcg64 rng;
  rng.seed(7777ULL, 0ULL);
  constexpr std::size_t kTrials = 1000;
  std::array<std::size_t, 1> outBuf{};
  std::size_t hits = 0;
  for (std::size_t t = 0; t < kTrials; ++t) {
    aExpjReservoir(weights, 1U, rng, std::span<std::size_t>(outBuf.data(), 1U));
    if (outBuf[0] == 4U) {
      ++hits;
    }
  }
  EXPECT_GT(hits, static_cast<std::size_t>(kTrials * 0.95));
}

TEST(AExpjReservoir, KEqualsN) {
  constexpr std::size_t kN = 8;
  NDArray<double, 1> weights({kN});
  for (std::size_t i = 0; i < kN; ++i) {
    weights(i) = 1.0 + static_cast<double>(i);
  }
  pcg64 rng;
  rng.seed(321ULL, 0ULL);
  std::array<std::size_t, kN> outBuf{};
  aExpjReservoir(weights, kN, rng, std::span<std::size_t>(outBuf.data(), kN));
  const std::set<std::size_t> unique(outBuf.begin(), outBuf.end());
  EXPECT_EQ(unique.size(), kN);
  for (std::size_t i = 0; i < kN; ++i) {
    EXPECT_TRUE(unique.count(i) == 1U) << "missing index " << i;
  }
}

TEST(AExpjReservoir, SmallWeightsDoNotUnderflow) {
  constexpr std::size_t kN = 100;
  constexpr std::size_t kK = 10;
  NDArray<double, 1> weights({kN});
  for (std::size_t i = 0; i < kN; ++i) {
    weights(i) = 1e-10;
  }
  pcg64 rng;
  rng.seed(8675309ULL, 0ULL);
  std::array<std::size_t, kK> outBuf{};
  aExpjReservoir(weights, kK, rng, std::span<std::size_t>(outBuf.data(), kK));
  // With log-key form all keys are finite negative numbers; the output must be k distinct
  // valid indices. With the broken u^(1/w) form every key is zero, outBuf would repeat 0..9.
  const std::set<std::size_t> unique(outBuf.begin(), outBuf.end());
  EXPECT_EQ(unique.size(), kK);
  for (const std::size_t idx : outBuf) {
    EXPECT_LT(idx, kN);
  }
}
