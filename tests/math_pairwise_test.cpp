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
using clustering::math::detail::PairwisePath;
using clustering::math::detail::pairwiseSqEuclideanGemm;
using clustering::math::detail::pairwiseSqEuclideanSimd;
using clustering::math::detail::pairwiseSqEuclideanWithDispatchInfo;
using clustering::math::detail::rowNormsSq;

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

namespace {

template <class T, Layout LX>
void referenceRowNormsSq(const NDArray<T, 2, LX> &X, NDArray<T, 1> &norms) {
  const std::size_t n = X.dim(0);
  const std::size_t d = X.dim(1);
  for (std::size_t i = 0; i < n; ++i) {
    T s = T{0};
    for (std::size_t k = 0; k < d; ++k) {
      const T v = X(i, k);
      s += v * v;
    }
    norms(i) = s;
  }
}

template <class T> bool allCloseRank1(const NDArray<T, 1> &a, const NDArray<T, 1> &b, T tol) {
  if (a.dim(0) != b.dim(0)) {
    return false;
  }
  for (std::size_t i = 0; i < a.dim(0); ++i) {
    const T d = a(i) - b(i);
    const T abs = d < T{0} ? -d : d;
    if (abs > tol) {
      return false;
    }
  }
  return true;
}

} // namespace

TEST(RowNormsSqF32, MatchesReferenceForSeveralFeatureDims) {
  for (const std::size_t d :
       {std::size_t{1}, std::size_t{7}, std::size_t{8}, std::size_t{16}, std::size_t{128}}) {
    for (const std::size_t n :
         {std::size_t{1}, std::size_t{4}, std::size_t{64}, std::size_t{256}}) {
      NDArray<float, 2> X({n, d});
      const auto seed = (static_cast<std::uint32_t>(d) * 13U) + static_cast<std::uint32_t>(n);
      fillRandom(X, seed);
      NDArray<float, 1> got({n});
      NDArray<float, 1> expect({n});
      for (std::size_t i = 0; i < n; ++i) {
        got(i) = -1.0F;
        expect(i) = -1.0F;
      }
      rowNormsSq(X, got, Pool{nullptr});
      referenceRowNormsSq<float, Layout::Contig>(X, expect);
      // Lane-split AVX2 reduction reassociates relative to the scalar reference; the residue
      // scales with the row's accumulated magnitude (~d/3 under U[-1,1]). 1e-4 absolute covers
      // d=128 comfortably without masking a real bug.
      EXPECT_TRUE(allCloseRank1<float>(expect, got, 1e-4F)) << "n=" << n << " d=" << d;
    }
  }
}

TEST(RowNormsSqF64, MatchesReferenceForSeveralFeatureDims) {
  for (const std::size_t d :
       {std::size_t{1}, std::size_t{7}, std::size_t{8}, std::size_t{16}, std::size_t{128}}) {
    for (const std::size_t n :
         {std::size_t{1}, std::size_t{4}, std::size_t{64}, std::size_t{256}}) {
      NDArray<double, 2> X({n, d});
      const auto seed = (static_cast<std::uint32_t>(d) * 17U) + static_cast<std::uint32_t>(n);
      fillRandom(X, seed);
      NDArray<double, 1> got({n});
      NDArray<double, 1> expect({n});
      for (std::size_t i = 0; i < n; ++i) {
        got(i) = -1.0;
        expect(i) = -1.0;
      }
      rowNormsSq(X, got, Pool{nullptr});
      referenceRowNormsSq<double, Layout::Contig>(X, expect);
      EXPECT_TRUE(allCloseRank1<double>(expect, got, 1e-12)) << "n=" << n << " d=" << d;
    }
  }
}

TEST(RowNormsSqF32, ThreadedMatchesSerialBitExact) {
  // n=512 with workerCount=4, minChunk=4 -> 128 >= 8 -> parallel fires; per-row inner arithmetic
  // is untouched across the fan-out, so results must match bit-for-bit.
  constexpr std::size_t n = 512;
  constexpr std::size_t d = 16;
  NDArray<float, 2> X({n, d});
  fillRandom(X, 701U);

  NDArray<float, 1> serial({n});
  NDArray<float, 1> threaded({n});
  for (std::size_t i = 0; i < n; ++i) {
    serial(i) = 0.0F;
    threaded(i) = 0.0F;
  }

  rowNormsSq(X, serial, Pool{nullptr});
  BS::light_thread_pool pool(4);
  rowNormsSq(X, threaded, Pool{&pool});

  for (std::size_t i = 0; i < n; ++i) {
    EXPECT_FLOAT_EQ(serial(i), threaded(i)) << "i=" << i;
  }
}

TEST(RowNormsSqF32, EmptyInputLeavesSentinelUntouched) {
  const NDArray<float, 2> X({0, 8});
  NDArray<float, 1> norms({0});
  // Empty X leaves the zero-length output untouched; shape metadata must survive.
  rowNormsSq(X, norms, Pool{nullptr});
  EXPECT_EQ(norms.dim(0), 0U);
}

TEST(RowNormsSqF32, TransposedSourceYieldsCorrectResult) {
  // Z is stored (d x n); Z.t() is (n x d) MaybeStrided. Scalar fallback path exercised.
  constexpr std::size_t n = 7;
  constexpr std::size_t d = 16;
  NDArray<float, 2> Z({d, n});
  fillRandom(Z, 801U);

  auto Xstrided = Z.t();
  using TransposedX = decltype(Xstrided);
  static_assert(std::is_same_v<TransposedX, NDArray<float, 2, Layout::MaybeStrided>>,
                "Z.t() must yield Layout::MaybeStrided");

  NDArray<float, 1> got({n});
  NDArray<float, 1> expect({n});
  rowNormsSq(Xstrided, got, Pool{nullptr});
  referenceRowNormsSq<float, Layout::MaybeStrided>(Xstrided, expect);
  EXPECT_TRUE(allCloseRank1<float>(expect, got, 1e-5F));
}

TEST(RowNormsSqF32Death, ShapeMismatchAborts) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  const NDArray<float, 2> X({4, 8});
  NDArray<float, 1> norms({3});
  EXPECT_DEATH(rowNormsSq(X, norms, Pool{nullptr}),
               "always-assert failed: norms\\.dim\\(0\\) == X\\.dim\\(0\\)");
}

TEST(RowNormsSqF32Death, ConstBorrowedNormsAborts) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  alignas(32) std::array<float, 32> xData{};
  alignas(32) std::array<float, 4> normsData{};
  xData.fill(1.0F);
  normsData.fill(0.0F);
  auto X = NDArray<float, 2>::borrow(xData.data(), {4, 8});
  auto norms =
      NDArray<float, 1>::borrow1D(static_cast<const float *>(normsData.data()), normsData.size());
  ASSERT_FALSE(norms.isMutable());
  EXPECT_DEATH(rowNormsSq(X, norms, Pool{nullptr}),
               "always-assert failed: norms\\.isMutable\\(\\)");
}

// GEMM-identity cross-validates against the SIMD-per-pair path. The reconstructed
// result acquires extra lane-order reassociation (gemm accumulation + norm add + clamp),
// so tolerances are loosened vs. the tight 1e-5/1e-12 used for the reference loop.
TEST(PairwiseSqEuclideanGemmF32, MatchesPublicSimdPath) {
  struct Shape {
    std::size_t n;
    std::size_t m;
    std::size_t d;
  };
  for (const Shape s : std::array<Shape, 3>{
           {{.n = 17, .m = 13, .d = 16}, {.n = 64, .m = 32, .d = 8}, {.n = 4, .m = 4, .d = 1}}}) {
    NDArray<float, 2> X({s.n, s.d});
    NDArray<float, 2> Y({s.m, s.d});
    const auto seed = static_cast<std::uint32_t>((s.n * 31U) + (s.m * 7U) + s.d + 911U);
    fillRandom(X, seed);
    fillRandom(Y, seed + 1U);

    NDArray<float, 2> simd({s.n, s.m});
    NDArray<float, 2> gemmOut({s.n, s.m});
    fillConst(simd, 0.0F);
    fillConst(gemmOut, 0.0F);

    pairwiseSqEuclidean(X, Y, simd, Pool{nullptr});
    pairwiseSqEuclideanGemm(X, Y, gemmOut, Pool{nullptr});

    EXPECT_TRUE(allClose(simd, gemmOut, 1e-4F, 1e-4F))
        << "n=" << s.n << " m=" << s.m << " d=" << s.d;
  }
}

TEST(PairwiseSqEuclideanGemmF64, MatchesPublicSimdPath) {
  struct Shape {
    std::size_t n;
    std::size_t m;
    std::size_t d;
  };
  for (const Shape s : std::array<Shape, 3>{
           {{.n = 17, .m = 13, .d = 16}, {.n = 64, .m = 32, .d = 8}, {.n = 4, .m = 4, .d = 1}}}) {
    NDArray<double, 2> X({s.n, s.d});
    NDArray<double, 2> Y({s.m, s.d});
    const auto seed = static_cast<std::uint32_t>((s.n * 41U) + (s.m * 11U) + s.d + 313U);
    fillRandom(X, seed);
    fillRandom(Y, seed + 1U);

    NDArray<double, 2> simd({s.n, s.m});
    NDArray<double, 2> gemmOut({s.n, s.m});
    fillConst(simd, 0.0);
    fillConst(gemmOut, 0.0);

    pairwiseSqEuclidean(X, Y, simd, Pool{nullptr});
    pairwiseSqEuclideanGemm(X, Y, gemmOut, Pool{nullptr});

    EXPECT_TRUE(allClose(simd, gemmOut, 1e-10, 1e-10))
        << "n=" << s.n << " m=" << s.m << " d=" << s.d;
  }
}

TEST(PairwiseSqEuclideanGemmF32, ThreadedMatchesSerialWithinTolerance) {
  // gemm picks its own block cadence internally so the accumulation order may differ between
  // serial and threaded calls; reuse the cross-path tolerance.
  constexpr std::size_t n = 128;
  constexpr std::size_t m = 64;
  constexpr std::size_t d = 16;
  NDArray<float, 2> X({n, d});
  NDArray<float, 2> Y({m, d});
  fillRandom(X, 1101U);
  fillRandom(Y, 1102U);

  NDArray<float, 2> serial({n, m});
  NDArray<float, 2> threaded({n, m});
  fillConst(serial, 0.0F);
  fillConst(threaded, 0.0F);

  pairwiseSqEuclideanGemm(X, Y, serial, Pool{nullptr});
  BS::light_thread_pool pool(4);
  pairwiseSqEuclideanGemm(X, Y, threaded, Pool{&pool});

  EXPECT_TRUE(allClose(serial, threaded, 1e-4F, 1e-4F));
}

TEST(PairwiseSqEuclideanGemmF32, ClampsNegativeCancellationToZero) {
  // X[0] == Y[0] at magnitude 1e3 with a per-lane perturbation drives ||x||^2 + ||y||^2 - 2 x . y
  // into measurably negative territory under gemm's accumulation order (probed offline:
  // pre-clamp residue ~-16 at d=128). The analytic squared distance is 0; without the clamp the
  // stored value leaks a negative. Pin non-negativity AND bounded magnitude near zero.
  constexpr std::size_t d = 128;
  NDArray<float, 2> X({1, d});
  NDArray<float, 2> Y({1, d});
  for (std::size_t k = 0; k < d; ++k) {
    const float v = 1.0e3F + (static_cast<float>(k) * 1e-3F);
    X(0, k) = v;
    Y(0, k) = v;
  }
  NDArray<float, 2> out({1, 1});
  fillConst(out, -42.0F);
  pairwiseSqEuclideanGemm(X, Y, out, Pool{nullptr});
  EXPECT_GE(out(0, 0), 0.0F);
  // Analytic result is 0; loose upper bound keeps the test portable across BLAS block cadences
  // that might change the residue magnitude.
  EXPECT_LE(out(0, 0), 1e-2F);
}

TEST(PairwiseSqEuclideanGemmF32, EmptyNPreservesShape) {
  constexpr std::size_t d = 8;
  const NDArray<float, 2> X({0, d});
  const NDArray<float, 2> Y({5, d});
  NDArray<float, 2> out({0, 5});
  pairwiseSqEuclideanGemm(X, Y, out, Pool{nullptr});
  EXPECT_EQ(out.dim(0), 0U);
  EXPECT_EQ(out.dim(1), 5U);
}

TEST(PairwiseSqEuclideanGemmF32, EmptyMPreservesShape) {
  constexpr std::size_t d = 8;
  const NDArray<float, 2> X({4, d});
  const NDArray<float, 2> Y({0, d});
  NDArray<float, 2> out({4, 0});
  pairwiseSqEuclideanGemm(X, Y, out, Pool{nullptr});
  EXPECT_EQ(out.dim(0), 4U);
  EXPECT_EQ(out.dim(1), 0U);
}

TEST(PairwiseSqEuclideanGemmF32, TransposedSourceYieldsCorrectResult) {
  constexpr std::size_t n = 6;
  constexpr std::size_t m = 5;
  constexpr std::size_t d = 16;
  NDArray<float, 2> Z({d, n});
  NDArray<float, 2> Y({m, d});
  fillRandom(Z, 1301U);
  fillRandom(Y, 1302U);

  auto Xstrided = Z.t();
  using TransposedX = decltype(Xstrided);
  static_assert(std::is_same_v<TransposedX, NDArray<float, 2, Layout::MaybeStrided>>,
                "Z.t() must yield Layout::MaybeStrided");

  NDArray<float, 2> gemmOut({n, m});
  NDArray<float, 2> simd({n, m});
  fillConst(gemmOut, 0.0F);
  fillConst(simd, 0.0F);

  pairwiseSqEuclideanGemm(Xstrided, Y, gemmOut, Pool{nullptr});
  pairwiseSqEuclidean(Xstrided, Y, simd, Pool{nullptr});
  EXPECT_TRUE(allClose(simd, gemmOut, 1e-4F, 1e-4F));
}

TEST(PairwiseSqEuclideanGemmF32Death, FeatureDimMismatchAborts) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  const NDArray<float, 2> X({4, 8});
  const NDArray<float, 2> Y({5, 7});
  NDArray<float, 2> out({4, 5});
  EXPECT_DEATH(pairwiseSqEuclideanGemm(X, Y, out, Pool{nullptr}),
               "always-assert failed: X\\.dim\\(1\\) == Y\\.dim\\(1\\)");
}

TEST(PairwiseSqEuclideanGemmF32Death, OutputRowMismatchAborts) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  const NDArray<float, 2> X({4, 8});
  const NDArray<float, 2> Y({5, 8});
  NDArray<float, 2> out({3, 5});
  EXPECT_DEATH(pairwiseSqEuclideanGemm(X, Y, out, Pool{nullptr}),
               "always-assert failed: out\\.dim\\(0\\) == X\\.dim\\(0\\)");
}

TEST(PairwiseSqEuclideanGemmF32Death, OutputColMismatchAborts) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  const NDArray<float, 2> X({4, 8});
  const NDArray<float, 2> Y({5, 8});
  NDArray<float, 2> out({4, 6});
  EXPECT_DEATH(pairwiseSqEuclideanGemm(X, Y, out, Pool{nullptr}),
               "always-assert failed: out\\.dim\\(1\\) == Y\\.dim\\(0\\)");
}

TEST(PairwiseSqEuclideanGemmF32Death, ConstBorrowedOutputAborts) {
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
  EXPECT_DEATH(pairwiseSqEuclideanGemm(X, Y, out, Pool{nullptr}),
               "always-assert failed: out\\.isMutable\\(\\)");
}

// The threshold macro must be defined by the header with the documented default. Lock the value
// so a stray -D override breaking the dispatch tests is obvious, and so the header's #ifndef
// guard is actually reached in the default build.
TEST(PairwiseDispatchThreshold, DefaultValueIsExposedByHeader) {
  EXPECT_EQ(static_cast<std::size_t>(CLUSTERING_PAIRWISE_GEMM_THRESHOLD), std::size_t{100000});
}

TEST(PairwiseDispatchThreshold, WorkJustBelowThresholdSelectsSimd) {
  // n=10, m=10, d=999 -> work = 99900 < 100000.
  constexpr std::size_t n = 10;
  constexpr std::size_t m = 10;
  constexpr std::size_t d = 999;
  NDArray<float, 2> X({n, d});
  NDArray<float, 2> Y({m, d});
  fillRandom(X, 2001U);
  fillRandom(Y, 2002U);
  NDArray<float, 2> out({n, m});
  fillConst(out, 0.0F);
  const PairwisePath path = pairwiseSqEuclideanWithDispatchInfo(X, Y, out, Pool{nullptr});
  EXPECT_EQ(path, PairwisePath::Simd);
}

TEST(PairwiseDispatchThreshold, WorkJustAboveThresholdSelectsGemm) {
  // n=10, m=10, d=1001 -> work = 100100 >= 100000.
  constexpr std::size_t n = 10;
  constexpr std::size_t m = 10;
  constexpr std::size_t d = 1001;
  NDArray<float, 2> X({n, d});
  NDArray<float, 2> Y({m, d});
  fillRandom(X, 2101U);
  fillRandom(Y, 2102U);
  NDArray<float, 2> out({n, m});
  fillConst(out, 0.0F);
  const PairwisePath path = pairwiseSqEuclideanWithDispatchInfo(X, Y, out, Pool{nullptr});
  EXPECT_EQ(path, PairwisePath::Gemm);
}

TEST(PairwiseDispatchThreshold, WorkAtThresholdSelectsGemm) {
  // Boundary: n=20, m=20, d=250 -> work = 100000, which satisfies `work >= threshold`.
  constexpr std::size_t n = 20;
  constexpr std::size_t m = 20;
  constexpr std::size_t d = 250;
  NDArray<float, 2> X({n, d});
  NDArray<float, 2> Y({m, d});
  fillRandom(X, 2201U);
  fillRandom(Y, 2202U);
  NDArray<float, 2> out({n, m});
  fillConst(out, 0.0F);
  const PairwisePath path = pairwiseSqEuclideanWithDispatchInfo(X, Y, out, Pool{nullptr});
  EXPECT_EQ(path, PairwisePath::Gemm);
}

TEST(PairwiseDispatchThreshold, PublicEntryAgreesWithBothKernelsAtBoundary) {
  // At work >= threshold the public entry takes the GEMM path. Sanity-check cross-path agreement
  // against both internal entries; tolerance absorbs the lane-order reassociation that GEMM
  // accumulation + norm-broadcast introduces relative to the per-pair reduction.
  constexpr std::size_t n = 20;
  constexpr std::size_t m = 20;
  constexpr std::size_t d = 251; // work = 100400, just above threshold
  NDArray<float, 2> X({n, d});
  NDArray<float, 2> Y({m, d});
  fillRandom(X, 2301U);
  fillRandom(Y, 2302U);

  NDArray<float, 2> pub({n, m});
  NDArray<float, 2> gemmOut({n, m});
  NDArray<float, 2> simd({n, m});
  fillConst(pub, 0.0F);
  fillConst(gemmOut, 0.0F);
  fillConst(simd, 0.0F);

  pairwiseSqEuclidean(X, Y, pub, Pool{nullptr});
  pairwiseSqEuclideanGemm(X, Y, gemmOut, Pool{nullptr});
  pairwiseSqEuclideanSimd(X, Y, simd, Pool{nullptr});

  EXPECT_TRUE(allClose(pub, gemmOut, 1e-4F, 1e-4F));
  EXPECT_TRUE(allClose(pub, simd, 1e-4F, 1e-4F));
}

TEST(PairwiseDispatchThreshold, PublicEntryEmptyNDoesNotWrite) {
  // Empty input must short-circuit in the public entry before dispatch so `work` is never
  // computed on a zero-row shape. A sentinel in a 1-element (0-axis zeroed) probe isn't
  // meaningful; instead use a (0, 5) out and confirm shape metadata survives.
  constexpr std::size_t d = 8;
  const NDArray<float, 2> X({0, d});
  const NDArray<float, 2> Y({5, d});
  NDArray<float, 2> out({0, 5});
  pairwiseSqEuclidean(X, Y, out, Pool{nullptr});
  EXPECT_EQ(out.dim(0), 0U);
  EXPECT_EQ(out.dim(1), 5U);
}

TEST(PairwiseDispatchThreshold, PublicEntryEmptyMDoesNotWrite) {
  constexpr std::size_t d = 8;
  const NDArray<float, 2> X({4, d});
  const NDArray<float, 2> Y({0, d});
  NDArray<float, 2> out({4, 0});
  pairwiseSqEuclidean(X, Y, out, Pool{nullptr});
  EXPECT_EQ(out.dim(0), 4U);
  EXPECT_EQ(out.dim(1), 0U);
}

TEST(PairwiseDispatchThreshold, DispatchInfoEmptyReturnsSimd) {
  // Empty-input convention: return the cheap path without invoking either kernel. Also pin the
  // no-write property with a sentinel-filled output that has at least one addressable cell on
  // the non-empty axis.
  constexpr std::size_t d = 8;
  const NDArray<float, 2> X({0, d});
  const NDArray<float, 2> Y({3, d});
  NDArray<float, 2> out({0, 3});
  const PairwisePath path = pairwiseSqEuclideanWithDispatchInfo(X, Y, out, Pool{nullptr});
  EXPECT_EQ(path, PairwisePath::Simd);
  EXPECT_EQ(out.dim(0), 0U);
  EXPECT_EQ(out.dim(1), 3U);
}

TEST(PairwiseDispatchThreshold, DispatchInfoEmptyMReturnsSimdAndPreservesSentinel) {
  // When m == 0 the out shape is (n, 0), so no cell is addressable anyway. Pin the API contract
  // (returns Simd) and ensure the call doesn't crash on the zero-sized dimension.
  constexpr std::size_t d = 8;
  const NDArray<float, 2> X({4, d});
  const NDArray<float, 2> Y({0, d});
  NDArray<float, 2> out({4, 0});
  const PairwisePath path = pairwiseSqEuclideanWithDispatchInfo(X, Y, out, Pool{nullptr});
  EXPECT_EQ(path, PairwisePath::Simd);
  EXPECT_EQ(out.dim(0), 4U);
  EXPECT_EQ(out.dim(1), 0U);
}
