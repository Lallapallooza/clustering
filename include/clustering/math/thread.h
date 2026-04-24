#pragma once

#include <BS_thread_pool.hpp>
#include <cstddef>
#include <optional>
#include <thread>

namespace clustering::math {

/**
 * @brief Clamp a caller-supplied @p nJobs to a valid @c BS::light_thread_pool worker count.
 *
 * Algorithm wrappers (@c KMeans, @c DBSCAN) accept @c nJobs via the scikit-learn convention:
 * @c 0 means "use hardware concurrency." @c std::thread::hardware_concurrency() can itself
 * return @c 0 on exotic targets; this helper promotes that to @c 1 so the pool-worker count
 * is always valid.
 *
 * @param nJobs Caller-supplied worker count; @c 0 means "match hardware concurrency."
 * @return Non-zero worker count suitable for constructing a @c BS::light_thread_pool.
 */
inline std::size_t clampedJobCount(std::size_t nJobs) noexcept {
  if (nJobs == 0) {
    const std::size_t hw = std::thread::hardware_concurrency();
    return hw == 0 ? std::size_t{1} : hw;
  }
  return nJobs;
}

/**
 * @brief Decide whether spawning a pool with @p nJobs workers is worth it for @p totalOps
 *        of arithmetic work.
 *
 * Complements @c Pool::shouldParallelizeWork: that method gates fan-out once a pool is
 * already attached, whereas this free helper decides whether to pay the one-time thread-spawn
 * cost at all. Returning @c false at small shapes lets the algorithm wrappers skip tens of
 * microseconds of pthread-create traffic a fully-serial fit would never amortize.
 *
 * @param totalOps         Approximate total arithmetic operation count across all workers.
 * @param nJobs            Target worker count (see @ref clampedJobCount for the clamp rule).
 * @param minOpsPerWorker  Minimum per-worker op budget that amortizes dispatch overhead.
 * @return @c true when the pool should be spawned, @c false when serial-only pays.
 */
inline bool shouldSpawnPool(std::size_t totalOps, std::size_t nJobs,
                            std::size_t minOpsPerWorker = std::size_t{1} << 15) noexcept {
  if (nJobs <= 1) {
    return false;
  }
  return (totalOps / nJobs) >= minOpsPerWorker;
}

/**
 * @brief Thin injection wrapper around a @c BS::light_thread_pool.
 *
 * Carries an optional pool pointer plus the helpers math kernels need to decide whether
 * a given workload is worth fanning out. A null pool is the explicit serial mode --
 * @c shouldParallelize then always reports @c false and the kernel runs on the calling
 * thread without touching any pool machinery.
 *
 * @note @c workerIndex returns @c 0 outside any pool task. Callers that rely on per-worker
 *       scratch isolation must either invoke it from inside a pool task body or pass
 *       @c Pool{nullptr} deliberately.
 */
struct Pool {
  /// Underlying pool, or @c nullptr to force serial execution.
  BS::light_thread_pool *pool = nullptr;

  /**
   * @brief Number of worker threads available, or @c 1 in serial mode.
   *
   * @return `pool->get_thread_count`() when a pool is attached, otherwise @c 1.
   */
  [[nodiscard]] std::size_t workerCount() const noexcept {
    return (pool != nullptr) ? pool->get_thread_count() : std::size_t{1};
  }

  /**
   * @brief Stable index of the calling worker thread within the owning pool.
   *
   * @return The worker id reported by @c BS::this_thread::get_index() when invoked from a
   *         pool task body, otherwise @c 0.
   */
  [[nodiscard]] static std::size_t workerIndex() noexcept {
    return BS::this_thread::get_index().value_or(std::size_t{0});
  }

  /**
   * @brief Decide whether @p totalWork warrants parallel dispatch.
   *
   * Returns @c true only when a pool is attached and the work splits into at least
   * `workerCount()` * @p minTasksPerWorker chunks of size @p minChunk. Guards against
   * `minChunk == 0` by reporting @c false rather than dividing by zero.
   *
   * @param totalWork         Total number of work units (e.g. matrix elements, rows).
   * @param minChunk          Minimum chunk size that amortizes per-task overhead.
   * @param minTasksPerWorker Minimum chunks per worker required to bother fanning out.
   * @return @c true when parallel dispatch should yield speedup, @c false otherwise.
   */
  [[nodiscard]] bool shouldParallelize(std::size_t totalWork, std::size_t minChunk,
                                       std::size_t minTasksPerWorker = 2) const noexcept {
    if (pool == nullptr || minChunk == 0) {
      return false;
    }
    return (totalWork / minChunk) >= (workerCount() * minTasksPerWorker);
  }

  /**
   * @brief Decide whether @p totalOps warrants parallel dispatch, based on work volume.
   *
   * Complements @ref shouldParallelize by gating on total arithmetic work rather than task
   * count. At very low per-unit cost (e.g. distance kernels at @c d=2) the chunk-count gate
   * can pass while the per-worker workload is dwarfed by dispatch overhead; this check prevents
   * fan-out when the per-worker op budget would not amortize the pool submit/wait syscalls.
   *
   * @param totalOps         Approximate total arithmetic operation count across all workers.
   * @param minOpsPerWorker  Minimum per-worker op budget that amortizes dispatch overhead.
   * @return @c true when fan-out pays, @c false otherwise.
   */
  [[nodiscard]] bool shouldParallelizeWork(std::size_t totalOps,
                                           std::size_t minOpsPerWorker = std::size_t{1}
                                                                         << 15) const noexcept {
    if (pool == nullptr) {
      return false;
    }
    return (totalOps / workerCount()) >= minOpsPerWorker;
  }
};

} // namespace clustering::math
