#include <gtest/gtest.h>

#include <BS_thread_pool.hpp>
#include <atomic>
#include <cstddef>

#include "clustering/math/thread.h"

using clustering::math::Pool;

namespace {

constexpr std::size_t kFixtureWorkers = 4;
constexpr std::size_t kSubmitLoopRange = 4096;

} // namespace

TEST(MathThreadPool, WorkerIndexInPoolTaskIsAlwaysInRange) {
  // BS::light_thread_pool's work-stealing scheduler does not guarantee every worker
  // receives at least one chunk -- a fast worker can drain the queue before a slower
  // peer wakes up. The durable invariant is that whichever worker runs a task body
  // reports an id strictly less than the configured worker count; per-invocation
  // equality with BS::this_thread::get_index() is pinned by the next test.
  BS::light_thread_pool pool(kFixtureWorkers);

  pool.submit_loop(std::size_t{0}, kSubmitLoopRange,
                   [](std::size_t /*i*/) {
                     const std::size_t id = Pool::workerIndex();
                     ASSERT_LT(id, kFixtureWorkers);
                   })
      .wait();
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
