#include <gtest/gtest.h>

#include <BS_thread_pool.hpp>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <random>
#include <thread>
#include <vector>

#include "clustering/math/equality.h"
#include "clustering/math/gemm.h"
#include "clustering/math/thread.h"
#include "clustering/ndarray.h"

using clustering::NDArray;
using clustering::math::allClose;
using clustering::math::arrayEqual;
using clustering::math::gemm;
using clustering::math::Pool;

namespace {

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

} // namespace

// Parallel Mc-dispatch produces the same bytes as the serial-pool reference at a shape big
// enough to trigger the shouldParallelize threshold on a 4-worker pool. Disjoint-row writes mean
// the FMA accumulation order is deterministic across thread assignments.
TEST(GemmThreadingF32, ParallelMatchesSerialAtLargeM) {
  // M=2000 -> mcBlockCount = ceil(2000/96) = 21; with workerCount=4, 21 >= 4*2 -> parallel fires.
  constexpr std::size_t M = 2000;
  constexpr std::size_t K = 64;
  constexpr std::size_t N = 64;

  NDArray<float, 2> A({M, K});
  NDArray<float, 2> B({K, N});
  fillRandom(A, 1U);
  fillRandom(B, 2U);

  NDArray<float, 2> Cserial({M, N});
  NDArray<float, 2> Cparallel({M, N});
  fillConst(Cserial, 0.0F);
  fillConst(Cparallel, 0.0F);

  gemm(A, B, Cserial, Pool{nullptr}, 1.0F, 0.0F);

  BS::light_thread_pool pool(4);
  gemm(A, B, Cparallel, Pool{&pool}, 1.0F, 0.0F);

  EXPECT_TRUE(arrayEqual(Cserial, Cparallel));
}

// Running the same gemm five times against the same pool yields five bit-identical outputs.
TEST(GemmThreadingF32, DeterministicAcrossRuns) {
  constexpr std::size_t M = 2000;
  constexpr std::size_t K = 64;
  constexpr std::size_t N = 64;

  NDArray<float, 2> A({M, K});
  NDArray<float, 2> B({K, N});
  fillRandom(A, 10U);
  fillRandom(B, 11U);

  BS::light_thread_pool pool(4);

  NDArray<float, 2> first({M, N});
  fillConst(first, 0.0F);
  gemm(A, B, first, Pool{&pool}, 1.0F, 0.0F);

  for (int trial = 0; trial < 4; ++trial) {
    NDArray<float, 2> Ctry({M, N});
    fillConst(Ctry, 0.0F);
    gemm(A, B, Ctry, Pool{&pool}, 1.0F, 0.0F);
    EXPECT_TRUE(arrayEqual(first, Ctry)) << "run " << trial << " diverged from run 0";
  }
}

// Two application threads each drive their own one-shot gemm (transient plans, disjoint outputs)
// against the same backing pool. The one-shot path allocates per-call scratch, so two concurrent
// callers do not alias arenas even when they share the same pool.
TEST(GemmThreadingF32, ConcurrentExecuteFromTwoApplicationThreads) {
  constexpr std::size_t M = 2000;
  constexpr std::size_t K = 64;
  constexpr std::size_t N = 64;

  NDArray<float, 2> A1({M, K});
  NDArray<float, 2> B1({K, N});
  NDArray<float, 2> A2({M, K});
  NDArray<float, 2> B2({K, N});
  fillRandom(A1, 21U);
  fillRandom(B1, 22U);
  fillRandom(A2, 23U);
  fillRandom(B2, 24U);

  NDArray<float, 2> C1({M, N});
  NDArray<float, 2> C2({M, N});
  fillConst(C1, 0.0F);
  fillConst(C2, 0.0F);

  BS::light_thread_pool pool(4);

  std::thread t1([&]() { gemm(A1, B1, C1, Pool{&pool}, 1.0F, 0.0F); });
  std::thread t2([&]() { gemm(A2, B2, C2, Pool{&pool}, 1.0F, 0.0F); });
  t1.join();
  t2.join();

  NDArray<float, 2> expect1({M, N});
  NDArray<float, 2> expect2({M, N});
  fillConst(expect1, 0.0F);
  fillConst(expect2, 0.0F);
  naiveGemm<float>(A1, B1, expect1, 1.0F, 0.0F);
  naiveGemm<float>(A2, B2, expect2, 1.0F, 0.0F);

  EXPECT_TRUE(allClose(expect1, C1, 1e-4F, 1e-4F));
  EXPECT_TRUE(allClose(expect2, C2, 1e-4F, 1e-4F));
}

// M=200 with 8 workers and kMc=96 yields mcBlockCount=3, which is below 8*2=16 and forces the
// serial fallback. The test pins end-to-end correctness on that path.
TEST(GemmThreadingF32, BelowThresholdRunsSerial) {
  constexpr std::size_t M = 200;
  constexpr std::size_t K = 64;
  constexpr std::size_t N = 64;

  NDArray<float, 2> A({M, K});
  NDArray<float, 2> B({K, N});
  fillRandom(A, 30U);
  fillRandom(B, 31U);

  NDArray<float, 2> Cgot({M, N});
  NDArray<float, 2> Cexpect({M, N});
  fillConst(Cgot, 0.0F);
  fillConst(Cexpect, 0.0F);

  BS::light_thread_pool pool(8);
  gemm(A, B, Cgot, Pool{&pool}, 1.0F, 0.0F);
  naiveGemm<float>(A, B, Cexpect, 1.0F, 0.0F);

  EXPECT_TRUE(allClose(Cexpect, Cgot, 1e-4F, 1e-4F));
}
