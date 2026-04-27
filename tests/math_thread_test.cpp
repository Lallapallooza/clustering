#include <gtest/gtest.h>

#include <atomic>
#include <cstddef>

#include "clustering/math/thread.h"

using clustering::math::OwnedPool;
using clustering::math::Pool;

namespace {

constexpr std::size_t kFixtureWorkers = 4;
constexpr std::size_t kSubmitLoopRange = 4096;

} // namespace

TEST(MathThreadPool, WorkerIndexInPoolTaskIsAlwaysInRange) {
  // The pool's work-stealing scheduler does not guarantee every worker receives at least
  // one chunk -- a fast worker can drain the queue before a slower peer wakes up. The
  // durable invariant is that whichever worker runs a task body reports an id strictly
  // less than the configured worker count.
  OwnedPool pool(kFixtureWorkers);
  Pool wrapper{&pool};

  wrapper.parallelForBlocks(std::size_t{0}, kSubmitLoopRange, std::size_t{0},
                            [](std::size_t lo, std::size_t hi) {
                              for (std::size_t i = lo; i < hi; ++i) {
                                const std::size_t id = Pool::workerIndex();
                                ASSERT_LT(id, kFixtureWorkers);
                              }
                            });
}

TEST(MathThreadPool, SerialPoolReportsCountOneAndIndexZero) {
  const Pool wrapper{nullptr};
  EXPECT_EQ(wrapper.workerCount(), std::size_t{1});
  EXPECT_EQ(Pool::workerIndex(), std::size_t{0});
}

TEST(MathThreadPool, ShouldParallelizeBelowThresholdReturnsFalse) {
  OwnedPool pool(8);
  const Pool wrapper{&pool};
  EXPECT_FALSE(wrapper.shouldParallelize(100, 96, 2));
}

TEST(MathThreadPool, ShouldParallelizeAboveThresholdReturnsTrue) {
  OwnedPool pool(8);
  const Pool wrapper{&pool};
  EXPECT_TRUE(wrapper.shouldParallelize(10000, 96, 2));
}

TEST(MathThreadPool, ShouldParallelizeOnSerialPoolAlwaysFalse) {
  const Pool wrapper{nullptr};
  EXPECT_FALSE(wrapper.shouldParallelize(10000, 96, 2));
}

TEST(MathThreadPool, ShouldParallelizeWithZeroChunkReturnsFalse) {
  OwnedPool pool(8);
  const Pool wrapper{&pool};
  EXPECT_FALSE(wrapper.shouldParallelize(10000, 0, 2));
}
