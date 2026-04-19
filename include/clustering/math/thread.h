#pragma once

#include <BS_thread_pool.hpp>
#include <cstddef>
#include <optional>

namespace clustering::math {

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
   * @return @c pool->get_thread_count() when a pool is attached, otherwise @c 1.
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
   * @c workerCount() * @p minTasksPerWorker chunks of size @p minChunk. Guards against
   * @c minChunk == 0 by reporting @c false rather than dividing by zero.
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
