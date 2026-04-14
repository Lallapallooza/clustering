#include <gtest/gtest.h>

#include <BS_thread_pool.hpp>
#include <array>
#include <cstddef>
#include <cstdint>
#include <random>
#include <type_traits>

#include "clustering/math/equality.h"
#include "clustering/math/pairwise.h"
#include "clustering/math/thread.h"
#include "clustering/ndarray.h"

using clustering::Layout;
using clustering::NDArray;
using clustering::math::allClose;
using clustering::math::arrayEqual;
using clustering::math::pairwiseSqEuclidean;
using clustering::math::Pool;

namespace {

template <class T> void fillRandom(NDArray<T, 2> &a, std::uint32_t seed) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<T> dist(static_cast<T>(-1.0), static_cast<T>(1.0));
  const std::size_t n = a.dim(0);
  const std::size_t d = a.dim(1);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t k = 0; k < d; ++k) {
      a[i][k] = dist(gen);
    }
  }
}

template <class T> void fillConst(NDArray<T, 2> &a, T value) {
  const std::size_t r = a.dim(0);
  const std::size_t c = a.dim(1);
  for (std::size_t i = 0; i < r; ++i) {
    for (std::size_t j = 0; j < c; ++j) {
      a[i][j] = value;
    }
  }
}

template <class T, Layout LX, Layout LY>
void referencePairwise(const NDArray<T, 2, LX> &X, const NDArray<T, 2, LY> &Y, NDArray<T, 2> &out) {
  const std::size_t n = X.dim(0);
  const std::size_t m = Y.dim(0);
  const std::size_t d = X.dim(1);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < m; ++j) {
      T sum = T{0};
      for (std::size_t k = 0; k < d; ++k) {
        const T diff = X(i, k) - Y(j, k);
        sum += diff * diff;
      }
      out(i, j) = sum;
    }
  }
}

} // namespace

TEST(PairwiseSqEuclideanF32, MatchesReferenceForSeveralFeatureDims) {
  constexpr std::size_t n = 9;
  constexpr std::size_t m = 7;
  for (const std::size_t d :
       {std::size_t{1}, std::size_t{7}, std::size_t{8}, std::size_t{16}, std::size_t{128}}) {
    NDArray<float, 2> X({n, d});
    NDArray<float, 2> Y({m, d});
    const auto seedBase = static_cast<std::uint32_t>(d) * 7U;
    fillRandom(X, seedBase + 1U);
    fillRandom(Y, seedBase + 2U);

    NDArray<float, 2> got({n, m});
    NDArray<float, 2> expect({n, m});
    fillConst(got, 0.0F);
    fillConst(expect, 0.0F);

    pairwiseSqEuclidean(X, Y, got, Pool{nullptr});
    referencePairwise<float, Layout::Contig, Layout::Contig>(X, Y, expect);

    EXPECT_TRUE(allClose(expect, got, 1e-5F, 1e-5F)) << "d=" << d;
  }
}

TEST(PairwiseSqEuclideanF64, MatchesReferenceForSeveralFeatureDims) {
  constexpr std::size_t n = 9;
  constexpr std::size_t m = 7;
  for (const std::size_t d :
       {std::size_t{1}, std::size_t{7}, std::size_t{8}, std::size_t{16}, std::size_t{128}}) {
    NDArray<double, 2> X({n, d});
    NDArray<double, 2> Y({m, d});
    const auto seedBase = static_cast<std::uint32_t>(d) * 11U;
    fillRandom(X, seedBase + 1U);
    fillRandom(Y, seedBase + 2U);

    NDArray<double, 2> got({n, m});
    NDArray<double, 2> expect({n, m});
    fillConst(got, 0.0);
    fillConst(expect, 0.0);

    pairwiseSqEuclidean(X, Y, got, Pool{nullptr});
    referencePairwise<double, Layout::Contig, Layout::Contig>(X, Y, expect);

    EXPECT_TRUE(allClose(expect, got, 1e-12, 1e-12)) << "d=" << d;
  }
}

TEST(PairwiseSqEuclideanF32, ScalarFallbackAtD7MatchesReference) {
  constexpr std::size_t n = 5;
  constexpr std::size_t m = 6;
  constexpr std::size_t d = 7;

  NDArray<float, 2> X({n, d});
  NDArray<float, 2> Y({m, d});
  fillRandom(X, 101U);
  fillRandom(Y, 102U);

  NDArray<float, 2> got({n, m});
  NDArray<float, 2> expect({n, m});
  fillConst(got, 0.0F);
  fillConst(expect, 0.0F);

  pairwiseSqEuclidean(X, Y, got, Pool{nullptr});
  referencePairwise<float, Layout::Contig, Layout::Contig>(X, Y, expect);

  EXPECT_TRUE(allClose(expect, got, 1e-5F, 1e-5F));
}

TEST(PairwiseSqEuclideanF32, PureAvx2AtD16MatchesReference) {
  constexpr std::size_t n = 5;
  constexpr std::size_t m = 6;
  constexpr std::size_t d = 16;

  NDArray<float, 2> X({n, d});
  NDArray<float, 2> Y({m, d});
  fillRandom(X, 201U);
  fillRandom(Y, 202U);

  NDArray<float, 2> got({n, m});
  NDArray<float, 2> expect({n, m});
  fillConst(got, 0.0F);
  fillConst(expect, 0.0F);

  pairwiseSqEuclidean(X, Y, got, Pool{nullptr});
  referencePairwise<float, Layout::Contig, Layout::Contig>(X, Y, expect);

  EXPECT_TRUE(allClose(expect, got, 1e-5F, 1e-5F));
}

TEST(PairwiseSqEuclideanF32, EmptyNLeavesOutputUntouched) {
  constexpr std::size_t d = 8;
  const NDArray<float, 2> X({0, d});
  const NDArray<float, 2> Y({5, d});
  NDArray<float, 2> out({0, 5});

  // Output already has zero elements: a call must simply return without touching storage or
  // misbehaving on the zero-sized shape.
  pairwiseSqEuclidean(X, Y, out, Pool{nullptr});
  EXPECT_EQ(out.dim(0), 0U);
  EXPECT_EQ(out.dim(1), 5U);
}

TEST(PairwiseSqEuclideanF32, EmptyMPreservesSentinelBytes) {
  constexpr std::size_t d = 8;
  const NDArray<float, 2> X({4, d});
  const NDArray<float, 2> Y({0, d});
  NDArray<float, 2> out({4, 0});

  // out has shape (4, 0): iterating either axis is a no-op and no store can happen. Still verify
  // the call returns and preserves shape metadata.
  pairwiseSqEuclidean(X, Y, out, Pool{nullptr});
  EXPECT_EQ(out.dim(0), 4U);
  EXPECT_EQ(out.dim(1), 0U);
}

TEST(PairwiseSqEuclideanF32, ThreadedMatchesSerialBitExact) {
  // n=512 with workerCount=4, minChunk=4 -> 512/4 = 128 >= 8 -> parallel fires.
  constexpr std::size_t n = 512;
  constexpr std::size_t m = 32;
  constexpr std::size_t d = 16;

  NDArray<float, 2> X({n, d});
  NDArray<float, 2> Y({m, d});
  fillRandom(X, 301U);
  fillRandom(Y, 302U);

  NDArray<float, 2> serial({n, m});
  NDArray<float, 2> threaded({n, m});
  fillConst(serial, 0.0F);
  fillConst(threaded, 0.0F);

  pairwiseSqEuclidean(X, Y, serial, Pool{nullptr});

  BS::light_thread_pool pool(4);
  pairwiseSqEuclidean(X, Y, threaded, Pool{&pool});

  // Threading fans out over rows of X; per-cell arithmetic order is untouched so results must be
  // bit-identical between the serial and threaded paths.
  EXPECT_TRUE(arrayEqual(serial, threaded));
}

TEST(PairwiseSqEuclideanF32, TransposedSourceYieldsCorrectResult) {
  // Z is stored as (d x n); Z.t() is (n x d) with MaybeStrided layout. This exercises the scalar
  // fallback path on a genuinely strided view and checks numerical correctness.
  constexpr std::size_t n = 6;
  constexpr std::size_t m = 5;
  constexpr std::size_t d = 16;

  NDArray<float, 2> Z({d, n});
  NDArray<float, 2> Y({m, d});
  fillRandom(Z, 401U);
  fillRandom(Y, 402U);

  auto Xstrided = Z.t();
  using TransposedX = decltype(Xstrided);
  static_assert(std::is_same_v<TransposedX, NDArray<float, 2, Layout::MaybeStrided>>,
                "Z.t() must yield Layout::MaybeStrided");

  NDArray<float, 2> got({n, m});
  NDArray<float, 2> expect({n, m});
  fillConst(got, 0.0F);
  fillConst(expect, 0.0F);

  pairwiseSqEuclidean(Xstrided, Y, got, Pool{nullptr});
  referencePairwise<float, Layout::MaybeStrided, Layout::Contig>(Xstrided, Y, expect);

  EXPECT_TRUE(allClose(expect, got, 1e-5F, 1e-5F));
}

TEST(PairwiseSqEuclideanF32Death, FeatureDimMismatchAborts) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  const NDArray<float, 2> X({4, 8});
  const NDArray<float, 2> Y({5, 7});
  NDArray<float, 2> out({4, 5});
  EXPECT_DEATH(pairwiseSqEuclidean(X, Y, out, Pool{nullptr}),
               "always-assert failed: X\\.dim\\(1\\) == Y\\.dim\\(1\\)");
}

TEST(PairwiseSqEuclideanF32Death, OutputRowMismatchAborts) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  const NDArray<float, 2> X({4, 8});
  const NDArray<float, 2> Y({5, 8});
  NDArray<float, 2> out({3, 5});
  EXPECT_DEATH(pairwiseSqEuclidean(X, Y, out, Pool{nullptr}),
               "always-assert failed: out\\.dim\\(0\\) == X\\.dim\\(0\\)");
}

TEST(PairwiseSqEuclideanF32Death, OutputColMismatchAborts) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  const NDArray<float, 2> X({4, 8});
  const NDArray<float, 2> Y({5, 8});
  NDArray<float, 2> out({4, 6});
  EXPECT_DEATH(pairwiseSqEuclidean(X, Y, out, Pool{nullptr}),
               "always-assert failed: out\\.dim\\(1\\) == Y\\.dim\\(0\\)");
}

TEST(PairwiseSqEuclideanF32Death, ConstBorrowedOutputAbortsAtPublicEntry) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  alignas(32) std::array<float, 32> xData{};
  alignas(32) std::array<float, 40> yData{};
  alignas(32) std::array<float, 20> outData{};
  xData.fill(1.0F);
  yData.fill(1.0F);
  outData.fill(0.0F);

  auto X = NDArray<float, 2>::borrow(xData.data(), {4, 8});
  auto Y = NDArray<float, 2>::borrow(yData.data(), {5, 8});
  auto out = NDArray<float, 2>::borrow(static_cast<const float *>(outData.data()), {4, 5});
  ASSERT_FALSE(out.isMutable());

  EXPECT_DEATH(pairwiseSqEuclidean(X, Y, out, Pool{nullptr}),
               "always-assert failed: out\\.isMutable\\(\\)");
}
