#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <random>
#include <type_traits>

#include "clustering/math/equality.h"
#include "clustering/math/gemm.h"
#include "clustering/math/thread.h"
#include "clustering/ndarray.h"

using clustering::Layout;
using clustering::NDArray;
using clustering::math::gemm;
using clustering::math::Pool;

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

} // namespace

TEST(GemmPublicF32, ContigContigMatchesNaive) {
  constexpr std::size_t M = 17;
  constexpr std::size_t K = 33;
  constexpr std::size_t N = 11;

  NDArray<float, 2> A({M, K});
  NDArray<float, 2> B({K, N});
  fillRandom(A, 1U);
  fillRandom(B, 2U);

  NDArray<float, 2> Cgemm({M, N});
  NDArray<float, 2> Cnaive({M, N});
  fillConst(Cgemm, 0.0F);
  fillConst(Cnaive, 0.0F);

  gemm(A, B, Cgemm, Pool{nullptr}, 1.0F, 0.0F);
  naiveGemm<float>(A, B, Cnaive, 1.0F, 0.0F);

  EXPECT_TRUE(clustering::math::allClose(Cnaive, Cgemm, 1e-5F, 1e-5F));
}

TEST(GemmPublicF32, ContigContigAlphaBeta) {
  constexpr std::size_t M = 8;
  constexpr std::size_t K = 16;
  constexpr std::size_t N = 6;

  NDArray<float, 2> A({M, K});
  NDArray<float, 2> B({K, N});
  fillRandom(A, 10U);
  fillRandom(B, 11U);

  NDArray<float, 2> Cinit({M, N});
  fillRandom(Cinit, 12U);

  NDArray<float, 2> Cgemm({M, N});
  NDArray<float, 2> Cnaive({M, N});
  for (std::size_t i = 0; i < M; ++i) {
    for (std::size_t j = 0; j < N; ++j) {
      Cgemm(i, j) = Cinit(i, j);
      Cnaive(i, j) = Cinit(i, j);
    }
  }

  gemm(A, B, Cgemm, Pool{nullptr}, -2.0F, 0.5F);
  naiveGemm<float>(A, B, Cnaive, -2.0F, 0.5F);

  EXPECT_TRUE(clustering::math::allClose(Cnaive, Cgemm, 1e-5F, 1e-5F));
}

TEST(GemmPublicF32, TransposedRhsViaCtad) {
  // Y is (N x K) so Y.t() has shape (K x N). gemm(A, Y.t(), C, pool) should compute A * Y^T and
  // match the naive reference threaded through the strided accessor.
  constexpr std::size_t M = 13;
  constexpr std::size_t K = 24;
  constexpr std::size_t N = 9;

  NDArray<float, 2> A({M, K});
  NDArray<float, 2> Y({N, K});
  fillRandom(A, 20U);
  fillRandom(Y, 21U);

  NDArray<float, 2> Cgemm({M, N});
  NDArray<float, 2> Cnaive({M, N});
  fillConst(Cgemm, 0.0F);
  fillConst(Cnaive, 0.0F);

  // CTAD resolves LB = Layout::MaybeStrided from Y.t()'s type -- no explicit template args.
  gemm(A, Y.t(), Cgemm, Pool{nullptr}, 1.0F, 0.0F);
  naiveGemmStrided<float, Layout::MaybeStrided>(A, Y.t(), Cnaive, 1.0F, 0.0F);

  EXPECT_TRUE(clustering::math::allClose(Cnaive, Cgemm, 1e-5F, 1e-5F));
}

TEST(GemmPublicF32, TransposedRhsCtadDeductionPin) {
  // Compile-time pin: the gemm(A, Y.t(), C, pool) call site resolves LA=Contig and
  // LB=MaybeStrided without any explicit template argument. If CTAD ever starts resolving
  // .t()'s return type to Contig, this static_assert becomes the first signal.
  const NDArray<float, 2> A({2, 3});
  const NDArray<float, 2> Y({4, 3});
  NDArray<float, 2> C({2, 4});

  using TransposedB = decltype(Y.t());
  static_assert(std::is_same_v<TransposedB, NDArray<float, 2, Layout::MaybeStrided>>,
                "Y.t() must yield Layout::MaybeStrided so gemm's LB deduces accordingly");

  // Force an ODR use so the body is instantiated under the deduced template arguments; any
  // CTAD breakage would surface as an overload-resolution failure here rather than at a
  // dynamic test site.
  gemm(A, Y.t(), C, Pool{nullptr}, 1.0F, 0.0F);
}

TEST(GemmPublicF32, EmptyShapeEarlyReturn) {
  // M=0 path: the backend must never see the call. A real backend that allocates Bp/Ap arenas
  // would show up in allocator counters; here we just check the call returns without touching
  // C or segfaulting on the zero-sized shape.
  const NDArray<float, 2> A({0, 8});
  const NDArray<float, 2> B({8, 6});
  NDArray<float, 2> C({0, 6});
  gemm(A, B, C, Pool{nullptr}, 1.0F, 0.0F);
  EXPECT_EQ(C.dim(0), 0u);

  // N=0 path.
  const NDArray<float, 2> A2({4, 8});
  const NDArray<float, 2> B2({8, 0});
  NDArray<float, 2> C2({4, 0});
  gemm(A2, B2, C2, Pool{nullptr}, 1.0F, 0.0F);
  EXPECT_EQ(C2.dim(1), 0u);
}

TEST(GemmPublicF32Death, ShapeMismatchAborts) {
  // A.dim(1) must equal B.dim(0); 4 != 5 trips the K-mismatch assert.
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  const NDArray<float, 2> A({6, 4});
  const NDArray<float, 2> B({5, 7});
  NDArray<float, 2> C({6, 7});
  EXPECT_DEATH(gemm(A, B, C, Pool{nullptr}, 1.0F, 0.0F),
               "always-assert failed: A\\.dim\\(1\\) == B\\.dim\\(0\\)");
}

TEST(GemmPublicF32Death, RowMismatchAborts) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  const NDArray<float, 2> A({6, 4});
  const NDArray<float, 2> B({4, 7});
  NDArray<float, 2> C({5, 7});
  EXPECT_DEATH(gemm(A, B, C, Pool{nullptr}, 1.0F, 0.0F),
               "always-assert failed: A\\.dim\\(0\\) == C\\.dim\\(0\\)");
}

TEST(GemmPublicF32Death, ColMismatchAborts) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  const NDArray<float, 2> A({6, 4});
  const NDArray<float, 2> B({4, 7});
  NDArray<float, 2> C({6, 8});
  EXPECT_DEATH(gemm(A, B, C, Pool{nullptr}, 1.0F, 0.0F),
               "always-assert failed: B\\.dim\\(1\\) == C\\.dim\\(1\\)");
}
