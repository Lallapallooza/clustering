#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <random>
#include <tuple>
#include <vector>

#include "clustering/math/detail/gemm_outer.h"
#include "clustering/math/detail/matrix_desc.h"
#include "clustering/math/equality.h"
#include "clustering/math/thread.h"
#include "clustering/ndarray.h"

using clustering::Layout;
using clustering::NDArray;
using clustering::detail::describeMatrix;
using clustering::detail::describeMatrixMut;
using clustering::math::Pool;
using clustering::math::detail::gemmRunReference;
using clustering::math::detail::kKc;
using clustering::math::detail::kMc;
using clustering::math::detail::kNc;

namespace {

template <class T>
void naiveGemm(const NDArray<T, 2> &A, const NDArray<T, 2> &B, NDArray<T, 2> &C, T alpha, T beta) {
  const std::size_t M = C.dim(0);
  const std::size_t N = C.dim(1);
  const std::size_t K = A.dim(1);
  for (std::size_t i = 0; i < M; ++i) {
    for (std::size_t j = 0; j < N; ++j) {
      T acc = T{0};
      for (std::size_t k = 0; k < K; ++k) {
        acc += A(i, k) * B(k, j);
      }
      const T prior = C(i, j);
      C(i, j) = (alpha * acc) + (beta * prior);
    }
  }
}

// Same shape as naiveGemm but takes B via the variadic accessor so callers can pass a transposed
// view (Layout::MaybeStrided) without needing two overloads.
template <class T, Layout LB>
void naiveGemmStrided(const NDArray<T, 2> &A, const NDArray<T, 2, LB> &B, NDArray<T, 2> &C, T alpha,
                      T beta) {
  const std::size_t M = C.dim(0);
  const std::size_t N = C.dim(1);
  const std::size_t K = A.dim(1);
  for (std::size_t i = 0; i < M; ++i) {
    for (std::size_t j = 0; j < N; ++j) {
      T acc = T{0};
      for (std::size_t k = 0; k < K; ++k) {
        acc += A(i, k) * B(k, j);
      }
      const T prior = C(i, j);
      C(i, j) = (alpha * acc) + (beta * prior);
    }
  }
}

// Thin wrapper that materializes the per-call Bp/Ap arenas and dispatches to the detail API.
// Tests drive the outer loop directly; the public gemm() entry is not yet exposed.
template <class T, Layout LA, Layout LB>
void runGemm(const NDArray<T, 2, LA> &A, const NDArray<T, 2, LB> &B, NDArray<T, 2> &C, T alpha,
             T beta) {
  std::vector<T> apArena(kMc<T> * kKc<T>, T{0});
  std::vector<T> bpArena(kKc<T> * kNc<T>, T{0});
  auto Ad = describeMatrix(A);
  auto Bd = describeMatrix(B);
  auto Cd = describeMatrixMut(C);
  gemmRunReference<T>(Ad, Bd, Cd, alpha, beta, apArena.data(), bpArena.data(), Pool{nullptr});
}

template <class T> void fillRandom(NDArray<T, 2> &a, std::uint32_t seed) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<T> dist(static_cast<T>(-1.0), static_cast<T>(1.0));
  const std::size_t M = a.dim(0);
  const std::size_t N = a.dim(1);
  for (std::size_t i = 0; i < M; ++i) {
    for (std::size_t j = 0; j < N; ++j) {
      a[i][j] = dist(gen);
    }
  }
}

template <class T> void fillConst(NDArray<T, 2> &a, T value) {
  const std::size_t M = a.dim(0);
  const std::size_t N = a.dim(1);
  for (std::size_t i = 0; i < M; ++i) {
    for (std::size_t j = 0; j < N; ++j) {
      a[i][j] = value;
    }
  }
}

template <class T> NDArray<T, 2> cloneArray(const NDArray<T, 2> &src) {
  NDArray<T, 2> out({src.dim(0), src.dim(1)});
  for (std::size_t i = 0; i < src.dim(0); ++i) {
    for (std::size_t j = 0; j < src.dim(1); ++j) {
      out[i][j] = src[i][j];
    }
  }
  return out;
}

} // namespace

class GemmSquareSweepFixture
    : public ::testing::TestWithParam<std::tuple<std::size_t, std::size_t, std::size_t>> {};

TEST_P(GemmSquareSweepFixture, SquareAllclosesNaive) {
  const auto [M, K, N] = GetParam();
  NDArray<float, 2> A({M, K});
  NDArray<float, 2> B({K, N});
  NDArray<float, 2> Cnaive({M, N});
  NDArray<float, 2> Cgemm({M, N});

  const auto seed = static_cast<std::uint32_t>((M * 9301U) + (K * 49297U) + (N * 233280U) + 1U);
  fillRandom(A, seed);
  fillRandom(B, seed + 1U);
  fillConst(Cnaive, 0.0F);
  fillConst(Cgemm, 0.0F);

  naiveGemm<float>(A, B, Cnaive, 1.0F, 0.0F);
  runGemm<float, Layout::Contig, Layout::Contig>(A, B, Cgemm, 1.0F, 0.0F);

  EXPECT_TRUE(clustering::math::allClose(Cnaive, Cgemm, 1e-5F, 1e-5F))
      << "M=" << M << " K=" << K << " N=" << N;
}

INSTANTIATE_TEST_SUITE_P(Shapes, GemmSquareSweepFixture,
                         ::testing::Combine(::testing::Values<std::size_t>(1, 7, 8, 9, 63, 64, 65,
                                                                           97, 1000),
                                            ::testing::Values<std::size_t>(1, 5, 6, 7, 16, 64),
                                            ::testing::Values<std::size_t>(1, 5, 6, 7, 10, 100)));

class GemmAlphaBetaFixture : public ::testing::TestWithParam<std::tuple<float, float>> {};

TEST_P(GemmAlphaBetaFixture, AlphaBetaSweep) {
  const auto [alpha, beta] = GetParam();
  // A modest shape sweep: covers exact-multiple, fractional-tail, and tail-on-both axes.
  const std::array<std::tuple<std::size_t, std::size_t, std::size_t>, 4> shapes{{
      {8, 16, 6},
      {9, 7, 7},
      {64, 64, 64},
      {17, 33, 19},
  }};

  for (const auto &shape : shapes) {
    const auto [M, K, N] = shape;
    NDArray<float, 2> A({M, K});
    NDArray<float, 2> B({K, N});
    NDArray<float, 2> Cinit({M, N});
    fillRandom(A, 42U);
    fillRandom(B, 43U);
    fillRandom(Cinit, 44U);

    NDArray<float, 2> Cnaive = cloneArray(Cinit);
    NDArray<float, 2> Cgemm = cloneArray(Cinit);

    naiveGemm<float>(A, B, Cnaive, alpha, beta);
    runGemm<float, Layout::Contig, Layout::Contig>(A, B, Cgemm, alpha, beta);

    EXPECT_TRUE(clustering::math::allClose(Cnaive, Cgemm, 1e-5F, 1e-5F))
        << "alpha=" << alpha << " beta=" << beta << " M=" << M << " K=" << K << " N=" << N;
  }
}

INSTANTIATE_TEST_SUITE_P(AlphaBeta, GemmAlphaBetaFixture,
                         ::testing::Combine(::testing::Values<float>(1.0F, -2.0F, 0.5F),
                                            ::testing::Values<float>(0.0F, 1.0F, 0.3F)));

TEST(GemmScalarReference, BetaZeroOverwritesNaN) {
  constexpr std::size_t M = 16;
  constexpr std::size_t K = 8;
  constexpr std::size_t N = 12;
  NDArray<float, 2> A({M, K});
  NDArray<float, 2> B({K, N});
  NDArray<float, 2> C({M, N});
  fillRandom(A, 7U);
  fillRandom(B, 8U);
  fillConst(C, std::numeric_limits<float>::quiet_NaN());

  runGemm<float, Layout::Contig, Layout::Contig>(A, B, C, 1.0F, 0.0F);

  for (std::size_t i = 0; i < M; ++i) {
    for (std::size_t j = 0; j < N; ++j) {
      EXPECT_FALSE(std::isnan(C[i][j])) << "i=" << i << " j=" << j;
    }
  }
}

TEST(GemmScalarReference, EmptyM) {
  // describeMatrix takes by const&; describeMatrixMut takes by &, so C cannot be const here.
  const NDArray<float, 2> A({0, 32});
  const NDArray<float, 2> B({32, 64});
  NDArray<float, 2> C({0, 64});
  // No element accesses possible on a 0-row C; primary contract is "no crash, no UB".
  runGemm<float, Layout::Contig, Layout::Contig>(A, B, C, 1.0F, 0.0F);
  EXPECT_EQ(C.dim(0), 0u);
  EXPECT_EQ(C.dim(1), 64u);
}

TEST(GemmScalarReference, EmptyN) {
  const NDArray<float, 2> A({64, 32});
  const NDArray<float, 2> B({32, 0});
  NDArray<float, 2> C({64, 0});
  runGemm<float, Layout::Contig, Layout::Contig>(A, B, C, 1.0F, 0.0F);
  EXPECT_EQ(C.dim(0), 64u);
  EXPECT_EQ(C.dim(1), 0u);
}

TEST(GemmScalarReference, EmptyKBetaZeroZeroesOutput) {
  constexpr std::size_t M = 32;
  constexpr std::size_t N = 16;
  const NDArray<float, 2> A({M, 0});
  const NDArray<float, 2> B({0, N});
  NDArray<float, 2> C({M, N});
  fillConst(C, 7.5F);

  runGemm<float, Layout::Contig, Layout::Contig>(A, B, C, 1.0F, 0.0F);
  for (std::size_t i = 0; i < M; ++i) {
    for (std::size_t j = 0; j < N; ++j) {
      EXPECT_FLOAT_EQ(C[i][j], 0.0F);
    }
  }
}

TEST(GemmScalarReference, EmptyKBetaScalesC) {
  constexpr std::size_t M = 32;
  constexpr std::size_t N = 16;
  const NDArray<float, 2> A({M, 0});
  const NDArray<float, 2> B({0, N});
  NDArray<float, 2> C({M, N});
  fillConst(C, 3.0F);

  runGemm<float, Layout::Contig, Layout::Contig>(A, B, C, 1.0F, 2.0F);
  for (std::size_t i = 0; i < M; ++i) {
    for (std::size_t j = 0; j < N; ++j) {
      EXPECT_FLOAT_EQ(C[i][j], 6.0F);
    }
  }
}

TEST(GemmScalarReference, TransposedRhs) {
  // B contiguous of shape (N, K); B.t() supplies a (K, N) Layout::MaybeStrided view so the
  // multiplication is A * B^T at the math level even though B itself is row-major (N, K).
  constexpr std::size_t M = 17;
  constexpr std::size_t K = 13;
  constexpr std::size_t N = 11;
  NDArray<float, 2> A({M, K});
  NDArray<float, 2> B({N, K});
  NDArray<float, 2> Cgemm({M, N});
  NDArray<float, 2> Cnaive({M, N});
  fillRandom(A, 101U);
  fillRandom(B, 102U);
  fillConst(Cgemm, 0.0F);
  fillConst(Cnaive, 0.0F);

  auto Bt = B.t();
  runGemm<float, Layout::Contig, Layout::MaybeStrided>(A, Bt, Cgemm, 1.0F, 0.0F);
  naiveGemmStrided<float>(A, Bt, Cnaive, 1.0F, 0.0F);

  EXPECT_TRUE(clustering::math::allClose(Cnaive, Cgemm, 1e-5F, 1e-5F));
}

TEST(GemmScalarReference, TailEdgeCornerTileBetaZero) {
  // M=9 -> 1 full Mr=8 panel + 1-row tail. N=7 -> 1 full Nr=6 panel + 1-col tail.
  constexpr std::size_t M = 9;
  constexpr std::size_t K = 64;
  constexpr std::size_t N = 7;
  NDArray<float, 2> A({M, K});
  NDArray<float, 2> B({K, N});
  NDArray<float, 2> Cgemm({M, N});
  NDArray<float, 2> Cnaive({M, N});
  fillRandom(A, 201U);
  fillRandom(B, 202U);
  fillConst(Cgemm, 0.0F);
  fillConst(Cnaive, 0.0F);

  runGemm<float, Layout::Contig, Layout::Contig>(A, B, Cgemm, 1.0F, 0.0F);
  naiveGemm<float>(A, B, Cnaive, 1.0F, 0.0F);

  EXPECT_TRUE(clustering::math::allClose(Cnaive, Cgemm, 1e-5F, 1e-5F));
}

TEST(GemmScalarReference, MultipleKcBlocksAccumulate) {
  // K=600 with kKc=256 forces 3 Kc passes; the second/third use effBeta=1 even when caller
  // beta=0, accumulating into prior Kc results. Bug here would leak the kZero overwrite
  // semantics across Kc passes, dropping the first 256 (or 512) inner-product contributions.
  constexpr std::size_t M = 17;
  constexpr std::size_t K = 600;
  constexpr std::size_t N = 13;
  NDArray<float, 2> A({M, K});
  NDArray<float, 2> B({K, N});
  NDArray<float, 2> Cgemm({M, N});
  NDArray<float, 2> Cnaive({M, N});
  fillRandom(A, 401U);
  fillRandom(B, 402U);
  fillConst(Cgemm, 0.0F);
  fillConst(Cnaive, 0.0F);

  runGemm<float, Layout::Contig, Layout::Contig>(A, B, Cgemm, 1.0F, 0.0F);
  naiveGemm<float>(A, B, Cnaive, 1.0F, 0.0F);

  EXPECT_TRUE(clustering::math::allClose(Cnaive, Cgemm, 1e-4F, 1e-4F));
}

TEST(GemmScalarReference, MultipleMcBlocks) {
  // M=200 with kMc=96 forces 3 Mc passes (96 + 96 + 8); ensures the ic loop's panelA index and
  // the writeback offset both advance correctly across Mc-block boundaries.
  constexpr std::size_t M = 200;
  constexpr std::size_t K = 32;
  constexpr std::size_t N = 24;
  NDArray<float, 2> A({M, K});
  NDArray<float, 2> B({K, N});
  NDArray<float, 2> Cgemm({M, N});
  NDArray<float, 2> Cnaive({M, N});
  fillRandom(A, 501U);
  fillRandom(B, 502U);
  fillConst(Cgemm, 0.0F);
  fillConst(Cnaive, 0.0F);

  runGemm<float, Layout::Contig, Layout::Contig>(A, B, Cgemm, 1.0F, 0.0F);
  naiveGemm<float>(A, B, Cnaive, 1.0F, 0.0F);

  EXPECT_TRUE(clustering::math::allClose(Cnaive, Cgemm, 1e-5F, 1e-5F));
}

TEST(GemmScalarReference, TailEdgeCornerTileBetaGeneral) {
  constexpr std::size_t M = 9;
  constexpr std::size_t K = 64;
  constexpr std::size_t N = 7;
  NDArray<float, 2> A({M, K});
  NDArray<float, 2> B({K, N});
  NDArray<float, 2> Cinit({M, N});
  fillRandom(A, 301U);
  fillRandom(B, 302U);
  fillRandom(Cinit, 303U);

  NDArray<float, 2> Cgemm = cloneArray(Cinit);
  NDArray<float, 2> Cnaive = cloneArray(Cinit);

  runGemm<float, Layout::Contig, Layout::Contig>(A, B, Cgemm, -1.5F, 0.75F);
  naiveGemm<float>(A, B, Cnaive, -1.5F, 0.75F);

  EXPECT_TRUE(clustering::math::allClose(Cnaive, Cgemm, 1e-5F, 1e-5F));
}
