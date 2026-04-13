#include <gtest/gtest.h>

#include <BS_thread_pool.hpp>
#include <array>
#include <atomic>
#include <cstddef>

#include "clustering/math/thread.h"

using clustering::math::Pool;

namespace {

constexpr std::size_t kFixtureWorkers = 4;
constexpr std::size_t kSubmitLoopRange = 4096;

} // namespace

TEST(MathThreadPool, WorkerIndexInPoolTaskIsStableAndCoversAllWorkers) {
  BS::light_thread_pool pool(kFixtureWorkers);

  std::array<std::atomic<bool>, kFixtureWorkers> seen{};
  for (auto &flag : seen) {
    flag.store(false, std::memory_order_relaxed);
  }

  pool.submit_loop(std::size_t{0}, kSubmitLoopRange,
                   [&seen](std::size_t /*i*/) {
                     const std::size_t id = Pool::workerIndex();
                     ASSERT_LT(id, kFixtureWorkers);
                     seen[id].store(true, std::memory_order_relaxed);
                   })
      .wait();

  for (std::size_t i = 0; i < kFixtureWorkers; ++i) {
    EXPECT_TRUE(seen[i].load(std::memory_order_relaxed)) << "worker " << i << " never observed";
  }
}

TEST(MathThreadPool, WorkerIndexMatchesBSGetIndex) {
  BS::light_thread_pool pool(kFixtureWorkers);
  std::atomic<bool> mismatch{false};

  pool.submit_loop(std::size_t{0}, kSubmitLoopRange,
                   [&mismatch](std::size_t /*i*/) {
                     const std::size_t wrapped = Pool::workerIndex();
                     const std::size_t native =
                         BS::this_thread::get_index().value_or(std::size_t{99999});
                     if (wrapped != native) {
                       mismatch.store(true, std::memory_order_relaxed);
                     }
                   })
      .wait();

  EXPECT_FALSE(mismatch.load(std::memory_order_relaxed));
}

TEST(MathThreadPool, SerialPoolReportsCountOneAndIndexZero) {
  const Pool wrapper{nullptr};
  EXPECT_EQ(wrapper.workerCount(), std::size_t{1});
  EXPECT_EQ(Pool::workerIndex(), std::size_t{0});
}

TEST(MathThreadPool, ShouldParallelizeBelowThresholdReturnsFalse) {
  BS::light_thread_pool pool(8);
  const Pool wrapper{&pool};
  EXPECT_FALSE(wrapper.shouldParallelize(100, 96, 2));
}

TEST(MathThreadPool, ShouldParallelizeAboveThresholdReturnsTrue) {
  BS::light_thread_pool pool(8);
  const Pool wrapper{&pool};
  EXPECT_TRUE(wrapper.shouldParallelize(10000, 96, 2));
}

TEST(MathThreadPool, ShouldParallelizeOnSerialPoolAlwaysFalse) {
  const Pool wrapper{nullptr};
  EXPECT_FALSE(wrapper.shouldParallelize(10000, 96, 2));
}

TEST(MathThreadPool, ShouldParallelizeWithZeroChunkReturnsFalse) {
  BS::light_thread_pool pool(8);
  const Pool wrapper{&pool};
  EXPECT_FALSE(wrapper.shouldParallelize(10000, 0, 2));
}
