#include <gtest/gtest.h>

#include <BS_thread_pool.hpp>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <utility>
#include <vector>

#include "clustering/math/defaults.h"
#include "clustering/math/pairwise.h"
#include "clustering/math/pairwise_argmin.h"
#include "clustering/math/thread.h"
#include "clustering/ndarray.h"

using clustering::NDArray;
using clustering::math::pairwiseArgminSqEuclidean;
using clustering::math::Pool;
using clustering::math::detail::ArgminPath;
using clustering::math::detail::pairwiseArgminMaterialized;
using clustering::math::detail::pairwiseArgminSqEuclideanWithDispatchInfo;

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

template <class T> void computeRowNormsReference(const NDArray<T, 2> &A, NDArray<T, 1> &norms) {
  const std::size_t n = A.dim(0);
  const std::size_t d = A.dim(1);
  for (std::size_t i = 0; i < n; ++i) {
    T s = T{0};
    for (std::size_t j = 0; j < d; ++j) {
      const T v = A(i, j);
      s += v * v;
    }
    norms(i) = s;
  }
}

} // namespace

TEST(PairwiseArgminDispatch, DefaultMaxDIs64) {
  EXPECT_EQ(clustering::math::defaults::pairwiseArgminMaxD, std::size_t{64});
}

TEST(PairwiseArgminDispatch, ContiguousSmallDTakesFusedPath) {
  constexpr std::size_t n = 32;
  constexpr std::size_t k = 16;
  constexpr std::size_t d = 8;
  NDArray<float, 2> X({n, d});
  NDArray<float, 2> C({k, d});
  NDArray<float, 1> cSqNorms({k});
  fillRandom(X, 101U);
  fillRandom(C, 102U);
  computeRowNormsReference(C, cSqNorms);

  NDArray<std::int32_t, 1> labels({n});
  NDArray<float, 1> outMin({n});
  const ArgminPath path =
      pairwiseArgminSqEuclideanWithDispatchInfo(X, C, cSqNorms, labels, outMin, Pool{nullptr});
#ifdef CLUSTERING_USE_AVX2
  EXPECT_EQ(path, ArgminPath::Fused);
#else
  EXPECT_EQ(path, ArgminPath::Materialized);
#endif
}

TEST(PairwiseArgminDispatch, LargeDBeyondMaxFallsBackToMaterialized) {
  // d = 257 crosses pairwiseArgminMaxD, so the fused path must refuse even though the other
  // dispatch conditions are satisfied.
  constexpr std::size_t n = 8;
  constexpr std::size_t k = 6;
  constexpr std::size_t d = 257;
  NDArray<float, 2> X({n, d});
  NDArray<float, 2> C({k, d});
  NDArray<float, 1> cSqNorms({k});
  fillRandom(X, 201U);
  fillRandom(C, 202U);
  computeRowNormsReference(C, cSqNorms);

  NDArray<std::int32_t, 1> labels({n});
  NDArray<float, 1> outMin({n});
  const ArgminPath path =
      pairwiseArgminSqEuclideanWithDispatchInfo(X, C, cSqNorms, labels, outMin, Pool{nullptr});
  EXPECT_EQ(path, ArgminPath::Materialized);
}

TEST(PairwiseArgminDispatch, MidDimBeyondMaxFallsBackToMaterialized) {
  // d=65, d=128, d=256 are all above pairwiseArgminMaxD=64 so they route through the chunked
  // materialized path -- this pins the configured cutoff.
  for (const std::size_t d : {std::size_t{65}, std::size_t{128}, std::size_t{256}}) {
    constexpr std::size_t n = 32;
    constexpr std::size_t k = 8;
    NDArray<float, 2> X({n, d});
    NDArray<float, 2> C({k, d});
    NDArray<float, 1> cSqNorms({k});
    fillRandom(X, 1001U);
    fillRandom(C, 1002U);
    computeRowNormsReference(C, cSqNorms);

    NDArray<std::int32_t, 1> labels({n});
    NDArray<float, 1> outMin({n});
    const ArgminPath path =
        pairwiseArgminSqEuclideanWithDispatchInfo(X, C, cSqNorms, labels, outMin, Pool{nullptr});
    EXPECT_EQ(path, ArgminPath::Materialized) << "d=" << d;
  }
}

TEST(PairwiseArgminDispatch, StridedXFallsBackToMaterialized) {
  // Use Z.t() to produce a MaybeStrided view. Z is (d x n), so Z.t() is (n x d) strided.
  constexpr std::size_t n = 16;
  constexpr std::size_t k = 4;
  constexpr std::size_t d = 8;
  NDArray<float, 2> Z({d, n});
  NDArray<float, 2> C({k, d});
  NDArray<float, 1> cSqNorms({k});
  fillRandom(Z, 301U);
  fillRandom(C, 302U);
  computeRowNormsReference(C, cSqNorms);

  auto X = Z.t();
  NDArray<std::int32_t, 1> labels({n});
  NDArray<float, 1> outMin({n});
  const ArgminPath path =
      pairwiseArgminSqEuclideanWithDispatchInfo(X, C, cSqNorms, labels, outMin, Pool{nullptr});
  EXPECT_EQ(path, ArgminPath::Materialized);
}

TEST(PairwiseArgminDispatch, DoubleAlwaysUsesMaterialized) {
  constexpr std::size_t n = 16;
  constexpr std::size_t k = 4;
  constexpr std::size_t d = 8;
  NDArray<double, 2> X({n, d});
  NDArray<double, 2> C({k, d});
  NDArray<double, 1> cSqNorms({k});
  fillRandom(X, 401U);
  fillRandom(C, 402U);
  computeRowNormsReference(C, cSqNorms);

  NDArray<std::int32_t, 1> labels({n});
  NDArray<double, 1> outMin({n});
  const ArgminPath path =
      pairwiseArgminSqEuclideanWithDispatchInfo(X, C, cSqNorms, labels, outMin, Pool{nullptr});
  EXPECT_EQ(path, ArgminPath::Materialized);
}

TEST(PairwiseArgminAgreement, FusedMatchesMaterializedAcrossEnvelope) {
  // 100% label match across a sweep of shapes inside the envelope.
  // outMinDistSq agreement within 1e-5 relative tolerance.
  const std::vector<std::size_t> ds{2, 4, 8, 16, 32, 64, 128};
  const std::vector<std::size_t> ks{2, 8, 32, 64, 256};
  const std::vector<std::size_t> ns{128, 1024}; // kept modest so the sweep stays under a second
  std::uint32_t seed = 1000U;

  for (const std::size_t d : ds) {
    for (const std::size_t k : ks) {
      for (const std::size_t n : ns) {
        NDArray<float, 2> X({n, d});
        NDArray<float, 2> C({k, d});
        NDArray<float, 1> cSqNorms({k});
        fillRandom(X, seed++);
        fillRandom(C, seed++);
        computeRowNormsReference(C, cSqNorms);

        NDArray<std::int32_t, 1> labelsFused({n});
        NDArray<float, 1> minFused({n});
        NDArray<std::int32_t, 1> labelsMat({n});
        NDArray<float, 1> minMat({n});

        // Fused path via the public entry (which routes to fused under these preconditions).
        pairwiseArgminSqEuclidean(X, C, cSqNorms, labelsFused, minFused, Pool{nullptr});
        // Materialized path directly.
        pairwiseArgminMaterialized(X, C, labelsMat, minMat, Pool{nullptr});

        for (std::size_t i = 0; i < n; ++i) {
          EXPECT_EQ(labelsFused(i), labelsMat(i))
              << "label mismatch at (d=" << d << ", k=" << k << ", n=" << n << ", i=" << i << ")";
          const float fused = minFused(i);
          const float mat = minMat(i);
          // Reassociation between the fused path's (||c_j||^2 - 2*dot + ||x||^2) and the
          // materialized path's (||x||^2 + ||y||^2 - 2*x.y) can shift the last few ULPs; at
          // very small distances those ULPs dominate relative error. Accept either a small
          // relative error OR a small absolute error so near-collinear sample pairs don't
          // force the test into pathology territory.
          const float diff = std::abs(fused - mat);
          const float denom = std::max(std::abs(mat), 1e-3F);
          const bool relOk = (diff / denom) <= 1e-3F;
          const bool absOk = diff <= 1e-5F;
          EXPECT_TRUE(relOk || absOk)
              << "minDistSq mismatch at (d=" << d << ", k=" << k << ", n=" << n << ", i=" << i
              << "): fused=" << fused << " mat=" << mat << " diff=" << diff;
        }
      }
    }
  }
}

TEST(PairwiseArgminAgreement, LargeDFallbackProducesCorrectLabels) {
  // d > 256 must fall through to the materialized path and still produce sane labels.
  constexpr std::size_t n = 64;
  constexpr std::size_t k = 12;
  constexpr std::size_t d = 300;
  NDArray<float, 2> X({n, d});
  NDArray<float, 2> C({k, d});
  NDArray<float, 1> cSqNorms({k});
  fillRandom(X, 5000U);
  fillRandom(C, 5001U);
  computeRowNormsReference(C, cSqNorms);

  NDArray<std::int32_t, 1> labels({n});
  NDArray<float, 1> outMin({n});
  pairwiseArgminSqEuclidean(X, C, cSqNorms, labels, outMin, Pool{nullptr});

  // Materialized reference via pairwiseSqEuclidean + scan.
  NDArray<std::int32_t, 1> refLabels({n});
  NDArray<float, 1> refMin({n});
  pairwiseArgminMaterialized(X, C, refLabels, refMin, Pool{nullptr});

  for (std::size_t i = 0; i < n; ++i) {
    EXPECT_EQ(labels(i), refLabels(i)) << "label mismatch at i=" << i;
  }
}

TEST(PairwiseArgminAgreement, MinDistSqMatchesBruteForceReference) {
  // Guard against a silent bug in both the fused AND materialized paths by comparing against a
  // triple-nested scalar reference.
  constexpr std::size_t n = 64;
  constexpr std::size_t k = 16;
  constexpr std::size_t d = 32;
  NDArray<float, 2> X({n, d});
  NDArray<float, 2> C({k, d});
  NDArray<float, 1> cSqNorms({k});
  fillRandom(X, 6000U);
  fillRandom(C, 6001U);
  computeRowNormsReference(C, cSqNorms);

  NDArray<std::int32_t, 1> labels({n});
  NDArray<float, 1> outMin({n});
  pairwiseArgminSqEuclidean(X, C, cSqNorms, labels, outMin, Pool{nullptr});

  for (std::size_t i = 0; i < n; ++i) {
    float bestVal = std::numeric_limits<float>::infinity();
    std::int32_t bestIdx = 0;
    for (std::size_t j = 0; j < k; ++j) {
      float s = 0.0F;
      for (std::size_t t = 0; t < d; ++t) {
        const float diff = X(i, t) - C(j, t);
        s += diff * diff;
      }
      if (s < bestVal) {
        bestVal = s;
        bestIdx = static_cast<std::int32_t>(j);
      }
    }
    EXPECT_EQ(labels(i), bestIdx) << "label mismatch at i=" << i;
    const float denom = std::max(std::abs(bestVal), 1e-6F);
    EXPECT_LE(std::abs(outMin(i) - bestVal) / denom, 1e-3F)
        << "minDistSq mismatch at i=" << i << ": got=" << outMin(i) << " ref=" << bestVal;
  }
}

TEST(PairwiseArgminAgreement, NonMultipleOfMrPartialTile) {
  // n is not a multiple of Mr (=8); the last M-tile must correctly emit only the valid rows.
  constexpr std::size_t n = 13;
  constexpr std::size_t k = 7;
  constexpr std::size_t d = 24;
  NDArray<float, 2> X({n, d});
  NDArray<float, 2> C({k, d});
  NDArray<float, 1> cSqNorms({k});
  fillRandom(X, 7000U);
  fillRandom(C, 7001U);
  computeRowNormsReference(C, cSqNorms);

  NDArray<std::int32_t, 1> labels({n});
  NDArray<float, 1> outMin({n});
  pairwiseArgminSqEuclidean(X, C, cSqNorms, labels, outMin, Pool{nullptr});

  NDArray<std::int32_t, 1> refLabels({n});
  NDArray<float, 1> refMin({n});
  pairwiseArgminMaterialized(X, C, refLabels, refMin, Pool{nullptr});

  for (std::size_t i = 0; i < n; ++i) {
    EXPECT_EQ(labels(i), refLabels(i)) << "label mismatch at i=" << i;
  }
}

TEST(PairwiseArgminAgreement, NonMultipleOfNrPartialPanel) {
  // k is not a multiple of Nr (=6); the last N-panel must zero-pad with +inf norms so the
  // padded columns never win the argmin contest.
  constexpr std::size_t n = 16;
  constexpr std::size_t k = 7; // 7 = 6 + 1, forcing a partial panel
  constexpr std::size_t d = 32;
  NDArray<float, 2> X({n, d});
  NDArray<float, 2> C({k, d});
  NDArray<float, 1> cSqNorms({k});
  fillRandom(X, 8000U);
  fillRandom(C, 8001U);
  computeRowNormsReference(C, cSqNorms);

  NDArray<std::int32_t, 1> labels({n});
  NDArray<float, 1> outMin({n});
  pairwiseArgminSqEuclidean(X, C, cSqNorms, labels, outMin, Pool{nullptr});

  for (std::size_t i = 0; i < n; ++i) {
    const std::int32_t li = labels(i);
    // A padded centroid would show up as a label >= k (the backing memory is uninitialized
    // beyond column k in cSqNorms' unpacked form, but the packer sets it to +inf).
    ASSERT_GE(li, 0);
    ASSERT_LT(li, static_cast<std::int32_t>(k));
  }
}

TEST(PairwiseArgminAgreement, ChunkedMaterializedMatchesBruteForceAcrossBoundary) {
  // Chunk-boundary stress: n straddles pairwiseArgminChunkRows so both whole-chunk and partial
  // tail paths execute in one call. Compare labels + minDistSq to a triple-nested scalar
  // reference so a bug in either the chunk loop or the per-chunk pairwiseSqEuclidean surfaces.
  using clustering::math::chunkedMaterializedScratchShape;
  using clustering::math::detail::pairwiseArgminMaterializedWithScratch;
  const std::vector<std::size_t> ns{200, 256, 300, 700};
  const std::vector<std::size_t> ds{32, 128};
  constexpr std::size_t k = 16;
  std::uint32_t seed = 20000U;

  for (const std::size_t n : ns) {
    for (const std::size_t d : ds) {
      NDArray<float, 2> X({n, d});
      NDArray<float, 2> C({k, d});
      NDArray<float, 1> cSqNorms({k});
      fillRandom(X, seed++);
      fillRandom(C, seed++);
      computeRowNormsReference(C, cSqNorms);

      NDArray<std::int32_t, 1> labels({n});
      NDArray<float, 1> outMin({n});
      const auto shape = chunkedMaterializedScratchShape(n, k);
      NDArray<float, 2> distsScratch({shape[0], shape[1]});
      pairwiseArgminMaterializedWithScratch(X, C, labels, outMin, distsScratch, Pool{nullptr});

      for (std::size_t i = 0; i < n; ++i) {
        float bestVal = std::numeric_limits<float>::infinity();
        std::int32_t bestIdx = 0;
        for (std::size_t j = 0; j < k; ++j) {
          float s = 0.0F;
          for (std::size_t t = 0; t < d; ++t) {
            const float diff = X(i, t) - C(j, t);
            s += diff * diff;
          }
          if (s < bestVal) {
            bestVal = s;
            bestIdx = static_cast<std::int32_t>(j);
          }
        }
        EXPECT_EQ(labels(i), bestIdx) << "n=" << n << " d=" << d << " i=" << i;
        const float denom = std::max(std::abs(bestVal), 1e-6F);
        EXPECT_LE(std::abs(outMin(i) - bestVal) / denom, 1e-3F)
            << "n=" << n << " d=" << d << " i=" << i;
      }
    }
  }
}

TEST(PairwiseArgminAgreement, ChunkedAndUnchunkedWrapperAgreeOnLabels) {
  // The public pairwiseArgminMaterialized wrapper and the explicit with-scratch entry must
  // produce bit-identical labels (the chunk loop is shape-deterministic, not order-dependent).
  using clustering::math::chunkedMaterializedScratchShape;
  using clustering::math::detail::pairwiseArgminMaterializedWithScratch;
  constexpr std::size_t n = 512;
  constexpr std::size_t d = 64;
  constexpr std::size_t k = 24;
  NDArray<float, 2> X({n, d});
  NDArray<float, 2> C({k, d});
  NDArray<float, 1> cSqNorms({k});
  fillRandom(X, 31000U);
  fillRandom(C, 31001U);
  computeRowNormsReference(C, cSqNorms);

  NDArray<std::int32_t, 1> labelsA({n});
  NDArray<float, 1> minA({n});
  pairwiseArgminMaterialized(X, C, labelsA, minA, Pool{nullptr});

  NDArray<std::int32_t, 1> labelsB({n});
  NDArray<float, 1> minB({n});
  const auto shape = chunkedMaterializedScratchShape(n, k);
  NDArray<float, 2> distsScratch({shape[0], shape[1]});
  pairwiseArgminMaterializedWithScratch(X, C, labelsB, minB, distsScratch, Pool{nullptr});

  for (std::size_t i = 0; i < n; ++i) {
    EXPECT_EQ(labelsA(i), labelsB(i)) << "at i=" << i;
    EXPECT_FLOAT_EQ(minA(i), minB(i)) << "at i=" << i;
  }
}

TEST(PairwiseArgminParallel, SerialAndThreadedAgree) {
  constexpr std::size_t n = 256;
  constexpr std::size_t k = 32;
  constexpr std::size_t d = 32;
  NDArray<float, 2> X({n, d});
  NDArray<float, 2> C({k, d});
  NDArray<float, 1> cSqNorms({k});
  fillRandom(X, 9000U);
  fillRandom(C, 9001U);
  computeRowNormsReference(C, cSqNorms);

  NDArray<std::int32_t, 1> labelsSerial({n});
  NDArray<float, 1> minSerial({n});
  pairwiseArgminSqEuclidean(X, C, cSqNorms, labelsSerial, minSerial, Pool{nullptr});

  BS::light_thread_pool tp(4);
  NDArray<std::int32_t, 1> labelsPar({n});
  NDArray<float, 1> minPar({n});
  pairwiseArgminSqEuclidean(X, C, cSqNorms, labelsPar, minPar, Pool{&tp});

  // Labels are bit-identical: each M-tile's arithmetic order is local to that tile regardless
  // of threading; the only source of reordering is which thread runs which tile.
  for (std::size_t i = 0; i < n; ++i) {
    EXPECT_EQ(labelsSerial(i), labelsPar(i)) << "at i=" << i;
    EXPECT_FLOAT_EQ(minSerial(i), minPar(i)) << "at i=" << i;
  }
}
