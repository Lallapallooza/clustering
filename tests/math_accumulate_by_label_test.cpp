#include <gtest/gtest.h>

#include <BS_thread_pool.hpp>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <utility>
#include <vector>

#include "clustering/math/accumulate_by_label.h"
#include "clustering/math/thread.h"
#include "clustering/ndarray.h"

using clustering::NDArray;
using clustering::math::accumulateByLabel;
using clustering::math::accumulateByLabelKahan;
using clustering::math::Pool;

namespace {

template <class T> void fillRandom(NDArray<T, 2> &a, std::uint32_t seed) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<T> dist(static_cast<T>(-1.0), static_cast<T>(1.0));
  const std::size_t n = a.dim(0);
  const std::size_t d = a.dim(1);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < d; ++j) {
      a[i][j] = dist(gen);
    }
  }
}

void fillLabelsUniform(NDArray<std::int32_t, 1> &labels, std::int32_t k, std::uint32_t seed) {
  std::mt19937 gen(seed);
  std::uniform_int_distribution<std::int32_t> dist(0, k - 1);
  for (std::size_t i = 0; i < labels.dim(0); ++i) {
    labels(i) = dist(gen);
  }
}

template <class T>
void scalarReference(const NDArray<T, 2> &X, const NDArray<std::int32_t, 1> &labels, std::size_t k,
                     NDArray<T, 2> &refSums, NDArray<std::int32_t, 1> &refCounts) {
  const std::size_t n = X.dim(0);
  const std::size_t d = X.dim(1);
  for (std::size_t c = 0; c < k; ++c) {
    refCounts(c) = 0;
    for (std::size_t t = 0; t < d; ++t) {
      refSums(c, t) = T{0};
    }
  }
  for (std::size_t i = 0; i < n; ++i) {
    const std::int32_t lbl = labels(i);
    if (lbl < 0 || std::cmp_greater_equal(lbl, k)) {
      continue;
    }
    refCounts(lbl) += 1;
    for (std::size_t t = 0; t < d; ++t) {
      refSums(lbl, t) += X(i, t);
    }
  }
}

} // namespace

TEST(AccumulateByLabel, SerialMatchesScalarReference) {
  constexpr std::size_t n = 256;
  constexpr std::size_t d = 16;
  constexpr std::size_t k = 8;
  NDArray<float, 2> X({n, d});
  NDArray<std::int32_t, 1> labels({n});
  fillRandom(X, 101U);
  fillLabelsUniform(labels, static_cast<std::int32_t>(k), 102U);

  NDArray<float, 2> sums({k, d});
  NDArray<std::int32_t, 1> counts({k});
  accumulateByLabel(X, labels, k, sums, counts, Pool{nullptr});

  NDArray<float, 2> refSums({k, d});
  NDArray<std::int32_t, 1> refCounts({k});
  scalarReference(X, labels, k, refSums, refCounts);

  for (std::size_t c = 0; c < k; ++c) {
    EXPECT_EQ(counts(c), refCounts(c)) << "count at c=" << c;
    for (std::size_t t = 0; t < d; ++t) {
      EXPECT_FLOAT_EQ(sums(c, t), refSums(c, t))
          << "sum mismatch at (c=" << c << ", t=" << t << ")";
    }
  }
}

TEST(AccumulateByLabel, ThreadedBitIdenticalAcrossRunsForFixedWorkerCount) {
  // Fixed workerCount -> identical fold produces identical output across repeated runs.
  constexpr std::size_t n = 10000;
  constexpr std::size_t d = 16;
  constexpr std::size_t k = 32;
  NDArray<float, 2> X({n, d});
  NDArray<std::int32_t, 1> labels({n});
  fillRandom(X, 201U);
  fillLabelsUniform(labels, static_cast<std::int32_t>(k), 202U);

  BS::light_thread_pool tp(4);
  NDArray<float, 2> sumsA({k, d});
  NDArray<std::int32_t, 1> countsA({k});
  accumulateByLabel(X, labels, k, sumsA, countsA, Pool{&tp});

  NDArray<float, 2> sumsB({k, d});
  NDArray<std::int32_t, 1> countsB({k});
  accumulateByLabel(X, labels, k, sumsB, countsB, Pool{&tp});

  for (std::size_t c = 0; c < k; ++c) {
    EXPECT_EQ(countsA(c), countsB(c));
    for (std::size_t t = 0; t < d; ++t) {
      // Bit-identity across repeated runs at the same workerCount.
      EXPECT_EQ(std::bit_cast<std::uint32_t>(sumsA(c, t)),
                std::bit_cast<std::uint32_t>(sumsB(c, t)));
    }
  }
}

TEST(AccumulateByLabel, MultipleWorkerCountsConverge) {
  // Sums at workerCount=1/2/4/8/16 should all land within a small relative tolerance; they are
  // NOT required to be bit-identical across different workerCounts. This test pins that the
  // reduction stays accurate even as partial-sum locality fragments.
  constexpr std::size_t n = 20000;
  constexpr std::size_t d = 8;
  constexpr std::size_t k = 16;
  NDArray<float, 2> X({n, d});
  NDArray<std::int32_t, 1> labels({n});
  fillRandom(X, 301U);
  fillLabelsUniform(labels, static_cast<std::int32_t>(k), 302U);

  NDArray<float, 2> sumsSerial({k, d});
  NDArray<std::int32_t, 1> countsSerial({k});
  accumulateByLabel(X, labels, k, sumsSerial, countsSerial, Pool{nullptr});

  for (const std::size_t workers :
       {std::size_t{2}, std::size_t{4}, std::size_t{8}, std::size_t{16}}) {
    BS::light_thread_pool tp(workers);
    NDArray<float, 2> sumsPar({k, d});
    NDArray<std::int32_t, 1> countsPar({k});
    accumulateByLabel(X, labels, k, sumsPar, countsPar, Pool{&tp});

    for (std::size_t c = 0; c < k; ++c) {
      EXPECT_EQ(countsSerial(c), countsPar(c)) << "workers=" << workers << " c=" << c;
      for (std::size_t t = 0; t < d; ++t) {
        // Cross-workerCount reassociation permits accumulated float drift proportional to
        // the number of partial sums folded; 1e-3 tolerates tree-depth differences at 20k
        // points across up to 16 blocks without masking a real regression.
        const float denom = std::max(std::abs(sumsSerial(c, t)), 1e-3F);
        EXPECT_LE(std::abs(sumsSerial(c, t) - sumsPar(c, t)) / denom, 1e-3F)
            << "workers=" << workers << " c=" << c << " t=" << t;
      }
    }
  }
}

TEST(AccumulateByLabel, EmptyClusterHasZeroSumsAndCount) {
  // Labels skip cluster 1 entirely.
  constexpr std::size_t n = 20;
  constexpr std::size_t d = 4;
  constexpr std::size_t k = 3;
  NDArray<float, 2> X({n, d});
  NDArray<std::int32_t, 1> labels({n});
  fillRandom(X, 401U);
  for (std::size_t i = 0; i < n; ++i) {
    labels(i) = (i % 2 == 0) ? 0 : 2; // never 1
  }

  NDArray<float, 2> sums({k, d});
  NDArray<std::int32_t, 1> counts({k});
  accumulateByLabel(X, labels, k, sums, counts, Pool{nullptr});

  EXPECT_EQ(counts(1), 0);
  for (std::size_t t = 0; t < d; ++t) {
    EXPECT_FLOAT_EQ(sums(1, t), 0.0F);
  }
  EXPECT_EQ(counts(0) + counts(2), static_cast<std::int32_t>(n));
}

TEST(AccumulateByLabelKahan, MatchesScalarKahanReferenceOnAdversarialData) {
  // Mix a dominant per-cluster large offset with many small per-point perturbations; the plain
  // accumulator should visibly lose the small contributions, while Kahan recovers them. We do
  // not compare plain vs Kahan here -- that would enforce plain is worse, which is not a
  // stable property; instead compare Kahan to a scalar-Kahan reference at full precision.
  constexpr std::size_t n = 1000000;
  constexpr std::size_t d = 16;
  constexpr std::size_t k = 100;
  NDArray<float, 2> X({n, d});
  NDArray<std::int32_t, 1> labels({n});
  std::mt19937 gen(501U);
  std::uniform_real_distribution<float> small(-1e-3F, 1e-3F);
  std::uniform_int_distribution<std::int32_t> lbl(0, static_cast<std::int32_t>(k) - 1);
  for (std::size_t i = 0; i < n; ++i) {
    labels(i) = lbl(gen);
    for (std::size_t t = 0; t < d; ++t) {
      X(i, t) = 1e6F + small(gen);
    }
  }

  NDArray<float, 2> sums({k, d});
  NDArray<std::int32_t, 1> counts({k});
  accumulateByLabelKahan(X, labels, k, sums, counts, Pool{nullptr});

  // Scalar Kahan reference.
  NDArray<float, 2> refSums({k, d});
  NDArray<std::int32_t, 1> refCounts({k});
  std::vector<float> refComp(k * d, 0.0F);
  for (std::size_t c = 0; c < k; ++c) {
    refCounts(c) = 0;
    for (std::size_t t = 0; t < d; ++t) {
      refSums(c, t) = 0.0F;
    }
  }
  for (std::size_t i = 0; i < n; ++i) {
    const std::int32_t li = labels(i);
    refCounts(li) += 1;
    for (std::size_t t = 0; t < d; ++t) {
      const float y = X(i, t) - refComp[(li * d) + t];
      const float tVal = refSums(li, t) + y;
      refComp[(li * d) + t] = (tVal - refSums(li, t)) - y;
      refSums(li, t) = tVal;
    }
  }

  for (std::size_t c = 0; c < k; ++c) {
    EXPECT_EQ(counts(c), refCounts(c)) << "c=" << c;
    for (std::size_t t = 0; t < d; ++t) {
      const float denom = std::max(std::abs(refSums(c, t)), 1e-3F);
      EXPECT_LE(std::abs(sums(c, t) - refSums(c, t)) / denom, 1e-7F)
          << "c=" << c << " t=" << t << " kahan=" << sums(c, t) << " ref=" << refSums(c, t);
    }
  }
}

TEST(AccumulateByLabelKahan, PlainAndKahanAgreeOnBenignInputs) {
  // On values without adversarial magnitude spread, plain and Kahan should agree tightly.
  constexpr std::size_t n = 5000;
  constexpr std::size_t d = 8;
  constexpr std::size_t k = 16;
  NDArray<float, 2> X({n, d});
  NDArray<std::int32_t, 1> labels({n});
  fillRandom(X, 601U);
  fillLabelsUniform(labels, static_cast<std::int32_t>(k), 602U);

  NDArray<float, 2> plainSums({k, d});
  NDArray<std::int32_t, 1> plainCounts({k});
  accumulateByLabel(X, labels, k, plainSums, plainCounts, Pool{nullptr});

  NDArray<float, 2> kSums({k, d});
  NDArray<std::int32_t, 1> kCounts({k});
  accumulateByLabelKahan(X, labels, k, kSums, kCounts, Pool{nullptr});

  for (std::size_t c = 0; c < k; ++c) {
    EXPECT_EQ(plainCounts(c), kCounts(c));
    for (std::size_t t = 0; t < d; ++t) {
      const float denom = std::max(std::abs(plainSums(c, t)), 1e-6F);
      EXPECT_LE(std::abs(plainSums(c, t) - kSums(c, t)) / denom, 1e-4F) << "c=" << c << " t=" << t;
    }
  }
}

TEST(AccumulateByLabel, NegativeLabelsSkippedAsNoise) {
  // Negative labels represent noise per DBSCAN's convention; accumulateByLabel must ignore them.
  constexpr std::size_t n = 12;
  constexpr std::size_t d = 3;
  constexpr std::size_t k = 2;
  NDArray<float, 2> X({n, d});
  NDArray<std::int32_t, 1> labels({n});
  fillRandom(X, 701U);
  for (std::size_t i = 0; i < n; ++i) {
    labels(i) = (i % 3 == 0) ? -1 : static_cast<std::int32_t>(i % k);
  }

  NDArray<float, 2> sums({k, d});
  NDArray<std::int32_t, 1> counts({k});
  accumulateByLabel(X, labels, k, sums, counts, Pool{nullptr});

  std::int32_t expectedCount = 0;
  for (std::size_t i = 0; i < n; ++i) {
    if (labels(i) >= 0) {
      ++expectedCount;
    }
  }
  EXPECT_EQ(counts(0) + counts(1), expectedCount);
}
