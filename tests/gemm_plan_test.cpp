#include <gtest/gtest.h>

#include <BS_thread_pool.hpp>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <random>
#include <type_traits>
#include <utility>

#include "clustering/math/detail/gemm_outer.h"
#include "clustering/math/equality.h"
#include "clustering/math/gemm_plan.h"
#include "clustering/math/thread.h"
#include "clustering/ndarray.h"

using clustering::NDArray;
using clustering::math::GemmPlan;
using clustering::math::Pool;
using clustering::math::detail::kKc;
using clustering::math::detail::kMc;

static_assert(!std::is_copy_constructible_v<GemmPlan<float>>);
static_assert(!std::is_copy_assignable_v<GemmPlan<float>>);
static_assert(std::is_nothrow_move_constructible_v<GemmPlan<float>>);
static_assert(std::is_nothrow_move_assignable_v<GemmPlan<float>>);

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

TEST(GemmPlanF32, KDimNDimReportShape) {
  NDArray<float, 2> B({64, 10});
  fillRandom(B, 1U);
  const GemmPlan<float> plan(B, Pool{nullptr});
  EXPECT_EQ(plan.kDim(), 64u);
  EXPECT_EQ(plan.nDim(), 10u);
}

TEST(GemmPlanF32, BpAlignedTo32Bytes) {
  NDArray<float, 2> B({64, 10});
  fillRandom(B, 2U);
  const GemmPlan<float> plan(B, Pool{nullptr});
  const auto addr = reinterpret_cast<std::uintptr_t>(plan.debugBpData());
  // Empty Bp reports 0 from aligned_alloc normalization — but this B is non-empty, so addr must
  // be a real aligned allocation.
  ASSERT_NE(addr, 0u);
  EXPECT_EQ(addr % 32u, 0u);
}

TEST(GemmPlanF32, ScratchSizeIsWorkerCountTimesMcTimesKc) {
  NDArray<float, 2> B({32, 16});
  fillRandom(B, 3U);
  BS::light_thread_pool pool(8);
  const GemmPlan<float> plan(B, Pool{&pool});
  EXPECT_EQ(plan.debugScratchSize(), 8u * kMc<float> * kKc<float>);
}

TEST(GemmPlanF32, SerialPoolScratchSizeIsOneMcKc) {
  NDArray<float, 2> B({32, 16});
  fillRandom(B, 31U);
  const GemmPlan<float> plan(B, Pool{nullptr});
  EXPECT_EQ(plan.debugScratchSize(), 1u * kMc<float> * kKc<float>);
}

TEST(GemmPlanF32, PlanReuseProducesCorrectResult) {
  constexpr std::size_t M = 17;
  constexpr std::size_t K = 33;
  constexpr std::size_t N = 11;

  NDArray<float, 2> B({K, N});
  fillRandom(B, 10U);
  const GemmPlan<float> plan(B, Pool{nullptr});

  for (const std::uint32_t seed : {100U, 200U, 300U}) {
    NDArray<float, 2> A({M, K});
    NDArray<float, 2> Cplan({M, N});
    NDArray<float, 2> Cnaive({M, N});
    fillRandom(A, seed);
    fillConst(Cplan, 0.0F);
    fillConst(Cnaive, 0.0F);

    plan.execute(A, Cplan, 1.0F, 0.0F);
    naiveGemm<float>(A, B, Cnaive, 1.0F, 0.0F);

    EXPECT_TRUE(clustering::math::allClose(Cnaive, Cplan, 1e-5F, 1e-5F)) << "seed=" << seed;
  }
}

TEST(GemmPlanF32, BitIdenticalAcrossSequentialExecutes) {
  constexpr std::size_t M = 16;
  constexpr std::size_t K = 32;
  constexpr std::size_t N = 12;

  NDArray<float, 2> A({M, K});
  NDArray<float, 2> B({K, N});
  fillRandom(A, 50U);
  fillRandom(B, 51U);
  const GemmPlan<float> plan(B, Pool{nullptr});

  NDArray<float, 2> C1({M, N});
  NDArray<float, 2> C2({M, N});
  fillConst(C1, 0.0F);
  fillConst(C2, 0.0F);

  plan.execute(A, C1, 1.0F, 0.0F);
  plan.execute(A, C2, 1.0F, 0.0F);

  EXPECT_TRUE(clustering::math::arrayEqual(C1, C2));
}

TEST(GemmPlanF32, BLifetimeIndependence) {
  constexpr std::size_t M = 13;
  constexpr std::size_t K = 24;
  constexpr std::size_t N = 9;

  NDArray<float, 2> A({M, K});
  fillRandom(A, 60U);

  NDArray<float, 2> Bcopy({K, N});
  NDArray<float, 2> Cnaive({M, N});
  fillConst(Cnaive, 0.0F);

  // Construct plan in an inner scope so the source B dies before execute runs. Bcopy captures
  // the value for the naive reference computation.
  const GemmPlan<float> plan = [&] {
    NDArray<float, 2> B({K, N});
    fillRandom(B, 61U);
    for (std::size_t i = 0; i < K; ++i) {
      for (std::size_t j = 0; j < N; ++j) {
        Bcopy(i, j) = B(i, j);
      }
    }
    return GemmPlan<float>(B, Pool{nullptr});
  }();

  NDArray<float, 2> Cplan({M, N});
  fillConst(Cplan, 0.0F);
  plan.execute(A, Cplan, 1.0F, 0.0F);
  naiveGemm<float>(A, Bcopy, Cnaive, 1.0F, 0.0F);

  EXPECT_TRUE(clustering::math::allClose(Cnaive, Cplan, 1e-5F, 1e-5F));
}

TEST(GemmPlanF32, MovedPlanProducesSameResult) {
  constexpr std::size_t M = 11;
  constexpr std::size_t K = 20;
  constexpr std::size_t N = 7;

  NDArray<float, 2> A({M, K});
  NDArray<float, 2> B({K, N});
  fillRandom(A, 70U);
  fillRandom(B, 71U);

  GemmPlan<float> planA(B, Pool{nullptr});

  NDArray<float, 2> Cbefore({M, N});
  fillConst(Cbefore, 0.0F);
  planA.execute(A, Cbefore, 1.0F, 0.0F);

  const GemmPlan<float> planB(std::move(planA));

  NDArray<float, 2> Cafter({M, N});
  fillConst(Cafter, 0.0F);
  planB.execute(A, Cafter, 1.0F, 0.0F);

  EXPECT_TRUE(clustering::math::arrayEqual(Cbefore, Cafter));
}

TEST(GemmPlanF32, AlphaBetaReuse) {
  constexpr std::size_t M = 8;
  constexpr std::size_t K = 16;
  constexpr std::size_t N = 6;

  NDArray<float, 2> A({M, K});
  NDArray<float, 2> B({K, N});
  fillRandom(A, 80U);
  fillRandom(B, 81U);

  const GemmPlan<float> plan(B, Pool{nullptr});

  NDArray<float, 2> Cinit({M, N});
  fillRandom(Cinit, 82U);

  NDArray<float, 2> Cplan({M, N});
  NDArray<float, 2> Cnaive({M, N});
  for (std::size_t i = 0; i < M; ++i) {
    for (std::size_t j = 0; j < N; ++j) {
      Cplan(i, j) = Cinit(i, j);
      Cnaive(i, j) = Cinit(i, j);
    }
  }

  plan.execute(A, Cplan, -1.5F, 0.75F);
  naiveGemm<float>(A, B, Cnaive, -1.5F, 0.75F);

  EXPECT_TRUE(clustering::math::allClose(Cnaive, Cplan, 1e-5F, 1e-5F));
}

TEST(GemmPlanF32, MultipleKcBlocksPlanReuse) {
  // K=600 forces 3 Kc blocks (kKc=256) so the pre-packed layout exercises pcOffInJc stepping.
  constexpr std::size_t M = 17;
  constexpr std::size_t K = 600;
  constexpr std::size_t N = 13;

  NDArray<float, 2> A({M, K});
  NDArray<float, 2> B({K, N});
  fillRandom(A, 90U);
  fillRandom(B, 91U);

  const GemmPlan<float> plan(B, Pool{nullptr});

  NDArray<float, 2> Cplan({M, N});
  NDArray<float, 2> Cnaive({M, N});
  fillConst(Cplan, 0.0F);
  fillConst(Cnaive, 0.0F);

  plan.execute(A, Cplan, 1.0F, 0.0F);
  naiveGemm<float>(A, B, Cnaive, 1.0F, 0.0F);

  EXPECT_TRUE(clustering::math::allClose(Cnaive, Cplan, 1e-4F, 1e-4F));
}

TEST(GemmPlanF32, TailShapeReuse) {
  // M=9, N=7 — Mr=8 and Nr=6 tails on both output axes; pre-packed Bp stores one roundedNc=12
  // column panel even though nc=7.
  constexpr std::size_t M = 9;
  constexpr std::size_t K = 64;
  constexpr std::size_t N = 7;

  NDArray<float, 2> A({M, K});
  NDArray<float, 2> B({K, N});
  fillRandom(A, 110U);
  fillRandom(B, 111U);

  const GemmPlan<float> plan(B, Pool{nullptr});

  NDArray<float, 2> Cplan({M, N});
  NDArray<float, 2> Cnaive({M, N});
  fillConst(Cplan, 0.0F);
  fillConst(Cnaive, 0.0F);

  plan.execute(A, Cplan, 1.0F, 0.0F);
  naiveGemm<float>(A, B, Cnaive, 1.0F, 0.0F);

  EXPECT_TRUE(clustering::math::allClose(Cnaive, Cplan, 1e-5F, 1e-5F));
}

TEST(GemmPlanF32, EmptyNPlan) {
  const NDArray<float, 2> B({32, 0});
  const GemmPlan<float> plan(B, Pool{nullptr});
  EXPECT_EQ(plan.nDim(), 0u);
  const NDArray<float, 2> A({16, 32});
  NDArray<float, 2> C({16, 0});
  plan.execute(A, C, 1.0F, 0.0F);
  EXPECT_EQ(C.dim(1), 0u);
}

TEST(GemmPlanF32, EmptyMExecuteIsNoop) {
  NDArray<float, 2> B({32, 8});
  fillRandom(B, 200U);
  const GemmPlan<float> plan(B, Pool{nullptr});
  const NDArray<float, 2> A({0, 32});
  NDArray<float, 2> C({0, 8});
  plan.execute(A, C, 1.0F, 0.0F);
  EXPECT_EQ(C.dim(0), 0u);
}

TEST(GemmPlanF32, EmptyKBetaScalesC) {
  // K==0 via B with zero rows. plan.execute must honor the BLAS identity C <- beta*C.
  const NDArray<float, 2> B({0, 8});
  const GemmPlan<float> plan(B, Pool{nullptr});
  EXPECT_EQ(plan.kDim(), 0u);

  const NDArray<float, 2> A({16, 0});
  NDArray<float, 2> C({16, 8});
  fillConst(C, 3.0F);

  plan.execute(A, C, 1.0F, 2.0F);
  for (std::size_t i = 0; i < 16; ++i) {
    for (std::size_t j = 0; j < 8; ++j) {
      EXPECT_FLOAT_EQ(C[i][j], 6.0F);
    }
  }
}

TEST(GemmPlanF32, PoolPlanExecuteMatchesSerial) {
  // Construct on an 8-worker pool. execute() calls workerIndex() from the caller thread
  // (outside any pool task body), which must report 0 — slicing into arena 0, identical
  // storage regardless of pool size. Output must match the serial-pool plan.
  constexpr std::size_t M = 40;
  constexpr std::size_t K = 48;
  constexpr std::size_t N = 12;

  NDArray<float, 2> A({M, K});
  NDArray<float, 2> B({K, N});
  fillRandom(A, 300U);
  fillRandom(B, 301U);

  BS::light_thread_pool pool(8);
  const GemmPlan<float> poolPlan(B, Pool{&pool});
  const GemmPlan<float> serialPlan(B, Pool{nullptr});

  NDArray<float, 2> Cpool({M, N});
  NDArray<float, 2> Cserial({M, N});
  fillConst(Cpool, 0.0F);
  fillConst(Cserial, 0.0F);

  poolPlan.execute(A, Cpool, 1.0F, 0.0F);
  serialPlan.execute(A, Cserial, 1.0F, 0.0F);

  EXPECT_TRUE(clustering::math::arrayEqual(Cpool, Cserial));
}
