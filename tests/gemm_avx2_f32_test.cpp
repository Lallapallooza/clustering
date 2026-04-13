#include <gtest/gtest.h>

#ifdef CLUSTERING_USE_AVX2

#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <random>
#include <tuple>
#include <vector>

#include "clustering/math/detail/gemm_kernel_scalar.h"
#include "clustering/math/detail/gemm_outer.h"
#include "clustering/math/detail/gemm_pack.h"
#include "clustering/math/detail/matrix_desc.h"
#include "clustering/math/equality.h"
#include "clustering/math/thread.h"
#include "clustering/ndarray.h"

using clustering::Layout;
using clustering::NDArray;
using clustering::detail::describeMatrix;
using clustering::detail::describeMatrixMut;
using clustering::math::Pool;
using clustering::math::detail::BetaKind;
using clustering::math::detail::gemmKernelMrNrScalar;
using clustering::math::detail::gemmRunReference;
using clustering::math::detail::kKc;
using clustering::math::detail::kKernelMr;
using clustering::math::detail::kKernelNr;
using clustering::math::detail::kMc;
using clustering::math::detail::kNc;

namespace {

template <class T> using AlignedVec = std::vector<T, clustering::detail::AlignedAllocator<T, 32>>;

// Drives the shared outer loop with the AVX2 path active. The outer loop selects
// gemmKernel8x6Avx2F32 for T=float when CLUSTERING_USE_AVX2 is defined.
template <class T, Layout LA, Layout LB>
void runGemm(const NDArray<T, 2, LA> &A, const NDArray<T, 2, LB> &B, NDArray<T, 2> &C, T alpha,
             T beta) {
  AlignedVec<T> apArena(kMc<T> * kKc<T>, T{0});
  AlignedVec<T> bpArena(kKc<T> * kNc<T>, T{0});
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

// Drives the scalar kernel path directly so the "AVX2 vs scalar" oracle does not depend on an
// external re-implementation. Feeds the same packed layouts the AVX2 kernel consumes, i.e.
// column-major tile and the outer loop's Mc/Kc/Nc structure — by swapping the kernel fn
// pointer to the scalar clone via a wrapper outer-loop call.
template <class T, Layout LA, Layout LB>
void runGemmScalarOnly(const NDArray<T, 2, LA> &A, const NDArray<T, 2, LB> &B, NDArray<T, 2> &C,
                       T alpha, T beta) {
  // The outer loop auto-selects AVX2 for T=float when CLUSTERING_USE_AVX2 is defined; to exercise
  // the scalar kernel on the same packed layout, we emulate the outer loop here with the scalar
  // kernel pinned. Keeping the structure identical to gemm_outer.h guarantees the packed layout
  // (column-major tile + panel indexing) matches.
  constexpr std::size_t kMr = kKernelMr<T>;
  constexpr std::size_t kNr = kKernelNr<T>;
  constexpr std::size_t kMcVal = kMc<T>;
  constexpr std::size_t kNcVal = kNc<T>;
  constexpr std::size_t kKcVal = kKc<T>;

  const std::size_t M = C.dim(0);
  const std::size_t N = C.dim(1);
  const std::size_t K = A.dim(1);

  if (M == 0 || N == 0) {
    return;
  }

  auto Cd = describeMatrixMut(C);
  T *cBase = Cd.ptr;
  const std::ptrdiff_t cRowStride = Cd.rowStride;
  const std::ptrdiff_t cColStride = Cd.colStride;

  if (K == 0) {
    const bool zero = (beta == T{0});
    for (std::size_t i = 0; i < M; ++i) {
      for (std::size_t j = 0; j < N; ++j) {
        T &cell = cBase[(static_cast<std::ptrdiff_t>(i) * cRowStride) +
                        (static_cast<std::ptrdiff_t>(j) * cColStride)];
        cell = zero ? T{0} : (beta * cell);
      }
    }
    return;
  }

  auto Ad = describeMatrix(A);
  auto Bd = describeMatrix(B);

  AlignedVec<T> apArena(kMc<T> * kKc<T>, T{0});
  AlignedVec<T> bpArena(kKc<T> * kNc<T>, T{0});

  const auto kernelZero = &gemmKernelMrNrScalar<T, BetaKind::kZero>;
  const auto kernelGeneral = &gemmKernelMrNrScalar<T, BetaKind::kGeneral>;

  for (std::size_t jc = 0; jc < N; jc += kNcVal) {
    const std::size_t nc = (jc + kNcVal <= N) ? kNcVal : (N - jc);
    for (std::size_t pc = 0; pc < K; pc += kKcVal) {
      const std::size_t kc = (pc + kKcVal <= K) ? kKcVal : (K - pc);
      const bool firstKBlock = (pc == 0);
      const T effBeta = firstKBlock ? beta : T{1};
      const auto kernel = (effBeta == T{0}) ? kernelZero : kernelGeneral;

      clustering::math::detail::packB<T>(Bd, pc, kc, jc, nc, bpArena.data());

      for (std::size_t ic = 0; ic < M; ic += kMcVal) {
        const std::size_t mc = (ic + kMcVal <= M) ? kMcVal : (M - ic);
        clustering::math::detail::packA<T>(Ad, ic, mc, pc, kc, apArena.data());

        for (std::size_t ir = 0; ir < mc; ir += kMr) {
          const std::size_t mTail = (ir + kMr <= mc) ? kMr : (mc - ir);
          const std::size_t panelA = ir / kMr;
          const T *apPanel = apArena.data() + (panelA * kMr * kc);

          for (std::size_t jr = 0; jr < nc; jr += kNr) {
            const std::size_t nTail = (jr + kNr <= nc) ? kNr : (nc - jr);
            const std::size_t panelB = jr / kNr;
            const T *bpPanel = bpArena.data() + (panelB * kc * kNr);

            alignas(32) std::array<T, kMr * kNr> tile{};
            if (effBeta != T{0}) {
              for (std::size_t c = 0; c < kNr; ++c) {
                for (std::size_t r = 0; r < kMr; ++r) {
                  if (r < mTail && c < nTail) {
                    const T &cell = cBase[(static_cast<std::ptrdiff_t>(ic + ir + r) * cRowStride) +
                                          (static_cast<std::ptrdiff_t>(jc + jr + c) * cColStride)];
                    tile[(c * kMr) + r] = cell;
                  } else {
                    tile[(c * kMr) + r] = T{0};
                  }
                }
              }
            }

            kernel(apPanel, bpPanel, tile.data(), kc, alpha, effBeta);

            for (std::size_t c = 0; c < nTail; ++c) {
              for (std::size_t r = 0; r < mTail; ++r) {
                T &cell = cBase[(static_cast<std::ptrdiff_t>(ic + ir + r) * cRowStride) +
                                (static_cast<std::ptrdiff_t>(jc + jr + c) * cColStride)];
                cell = tile[(c * kMr) + r];
              }
            }
          }
        }
      }
    }
  }
}

} // namespace

class GemmAvx2F32SweepFixture
    : public ::testing::TestWithParam<std::tuple<std::size_t, std::size_t, std::size_t>> {};

TEST_P(GemmAvx2F32SweepFixture, AvxMatchesScalar) {
  const auto [M, K, N] = GetParam();
  NDArray<float, 2> A({M, K});
  NDArray<float, 2> B({K, N});
  NDArray<float, 2> Cavx({M, N});
  NDArray<float, 2> Cscalar({M, N});

  const auto seed = static_cast<std::uint32_t>((M * 9301U) + (K * 49297U) + (N * 233280U) + 17U);
  fillRandom(A, seed);
  fillRandom(B, seed + 1U);
  fillConst(Cavx, 0.0F);
  fillConst(Cscalar, 0.0F);

  runGemm<float, Layout::Contig, Layout::Contig>(A, B, Cavx, 1.0F, 0.0F);
  runGemmScalarOnly<float, Layout::Contig, Layout::Contig>(A, B, Cscalar, 1.0F, 0.0F);

  EXPECT_TRUE(clustering::math::allClose(Cscalar, Cavx, 1e-5F, 1e-5F))
      << "M=" << M << " K=" << K << " N=" << N;
}

INSTANTIATE_TEST_SUITE_P(Shapes, GemmAvx2F32SweepFixture,
                         ::testing::Combine(::testing::Values<std::size_t>(1, 7, 8, 9, 63, 64, 65,
                                                                           97, 1000),
                                            ::testing::Values<std::size_t>(1, 5, 6, 7, 16, 64),
                                            ::testing::Values<std::size_t>(1, 5, 6, 7, 10, 100)));

class GemmAvx2F32AlphaBetaFixture : public ::testing::TestWithParam<std::tuple<float, float>> {};

TEST_P(GemmAvx2F32AlphaBetaFixture, AvxAlphaBetaMatchesScalar) {
  const auto [alpha, beta] = GetParam();
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
    fillRandom(A, 142U);
    fillRandom(B, 143U);
    fillRandom(Cinit, 144U);

    NDArray<float, 2> Cavx = cloneArray(Cinit);
    NDArray<float, 2> Cscalar = cloneArray(Cinit);

    runGemm<float, Layout::Contig, Layout::Contig>(A, B, Cavx, alpha, beta);
    runGemmScalarOnly<float, Layout::Contig, Layout::Contig>(A, B, Cscalar, alpha, beta);

    EXPECT_TRUE(clustering::math::allClose(Cscalar, Cavx, 1e-5F, 1e-5F))
        << "alpha=" << alpha << " beta=" << beta << " M=" << M << " K=" << K << " N=" << N;
  }
}

INSTANTIATE_TEST_SUITE_P(AlphaBeta, GemmAvx2F32AlphaBetaFixture,
                         ::testing::Combine(::testing::Values<float>(1.0F, -2.0F, 0.5F),
                                            ::testing::Values<float>(0.0F, 1.0F, 0.3F)));

TEST(GemmAvx2F32, BetaZeroOverwritesNaN) {
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

TEST(GemmAvx2F32, EmptyShapes) {
  {
    const NDArray<float, 2> A({0, 32});
    const NDArray<float, 2> B({32, 64});
    NDArray<float, 2> C({0, 64});
    runGemm<float, Layout::Contig, Layout::Contig>(A, B, C, 1.0F, 0.0F);
    EXPECT_EQ(C.dim(0), 0u);
    EXPECT_EQ(C.dim(1), 64u);
  }
  {
    const NDArray<float, 2> A({64, 32});
    const NDArray<float, 2> B({32, 0});
    NDArray<float, 2> C({64, 0});
    runGemm<float, Layout::Contig, Layout::Contig>(A, B, C, 1.0F, 0.0F);
    EXPECT_EQ(C.dim(0), 64u);
    EXPECT_EQ(C.dim(1), 0u);
  }
  {
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
}

TEST(GemmAvx2F32, TransposedRhs) {
  constexpr std::size_t M = 17;
  constexpr std::size_t K = 13;
  constexpr std::size_t N = 11;
  NDArray<float, 2> A({M, K});
  NDArray<float, 2> B({N, K});
  NDArray<float, 2> Cavx({M, N});
  NDArray<float, 2> Cscalar({M, N});
  fillRandom(A, 601U);
  fillRandom(B, 602U);
  fillConst(Cavx, 0.0F);
  fillConst(Cscalar, 0.0F);

  auto Bt = B.t();
  runGemm<float, Layout::Contig, Layout::MaybeStrided>(A, Bt, Cavx, 1.0F, 0.0F);
  runGemmScalarOnly<float, Layout::Contig, Layout::MaybeStrided>(A, Bt, Cscalar, 1.0F, 0.0F);

  EXPECT_TRUE(clustering::math::allClose(Cscalar, Cavx, 1e-5F, 1e-5F));
}

TEST(GemmAvx2F32, TailEdgeCorner) {
  constexpr std::size_t M = 9;
  constexpr std::size_t K = 64;
  constexpr std::size_t N = 7;
  NDArray<float, 2> A({M, K});
  NDArray<float, 2> B({K, N});
  NDArray<float, 2> Cavx({M, N});
  NDArray<float, 2> Cscalar({M, N});
  fillRandom(A, 701U);
  fillRandom(B, 702U);
  fillConst(Cavx, 0.0F);
  fillConst(Cscalar, 0.0F);

  runGemm<float, Layout::Contig, Layout::Contig>(A, B, Cavx, 1.0F, 0.0F);
  runGemmScalarOnly<float, Layout::Contig, Layout::Contig>(A, B, Cscalar, 1.0F, 0.0F);
  EXPECT_TRUE(clustering::math::allClose(Cscalar, Cavx, 1e-5F, 1e-5F));
}

TEST(GemmAvx2F32, BetaZeroWallTimeIsNotWorseThanGeneralAtCanonicalShape) {
  // Compare wall time of the canonical clustering shape (M=10000, N=10, K=64) between the
  // beta=0 skip-load clone and the beta=general RMW clone. This is not a hard assertion on the
  // delta (gtest wall-time is coarse); the goal is to record measurable data on stderr. The
  // assertion is the weaker claim that beta=0 is not dramatically slower than beta=general,
  // which would contradict the skip-load premise outright.
  constexpr std::size_t M = 10000;
  constexpr std::size_t N = 10;
  constexpr std::size_t K = 64;

  NDArray<float, 2> A({M, K});
  NDArray<float, 2> B({K, N});
  NDArray<float, 2> C0({M, N});
  NDArray<float, 2> Cg({M, N});
  fillRandom(A, 8001U);
  fillRandom(B, 8002U);
  fillRandom(C0, 8003U);
  fillRandom(Cg, 8003U);

  const auto t0Start = std::chrono::steady_clock::now();
  runGemm<float, Layout::Contig, Layout::Contig>(A, B, C0, 1.0F, 0.0F);
  const auto t0End = std::chrono::steady_clock::now();

  const auto tgStart = std::chrono::steady_clock::now();
  runGemm<float, Layout::Contig, Layout::Contig>(A, B, Cg, 1.0F, 0.5F);
  const auto tgEnd = std::chrono::steady_clock::now();

  const auto zeroNs = std::chrono::duration_cast<std::chrono::nanoseconds>(t0End - t0Start).count();
  const auto genNs = std::chrono::duration_cast<std::chrono::nanoseconds>(tgEnd - tgStart).count();

  // Report to stderr for eyeball comparison; the wall-time delta between the two kernel clones
  // is too noisy for a tight assertion but useful data to surface when something changes.
  const double ratio = static_cast<double>(zeroNs) / static_cast<double>(genNs);
  std::cerr << "kZero/kGeneral wall (M=" << M << ", N=" << N << ", K=" << K << "): kZero=" << zeroNs
            << "ns, kGeneral=" << genNs << "ns, ratio=" << ratio << '\n';

  // Single-call timing is noisy; assert only the floor: kZero must not be dramatically worse
  // than kGeneral (which would contradict the skip-load premise).
  EXPECT_LT(zeroNs, genNs * 2);
}

#endif // CLUSTERING_USE_AVX2
