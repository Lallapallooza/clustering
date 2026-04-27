#pragma once

#include <citor/example_hints.h>

#include <cstddef>
#include <cstdint>
#include <thread>
#include <utility>

#if defined(CLUSTERING_USE_BS_POOL)
#include <BS_thread_pool.hpp>
#else
#include <citor/thread_pool.h>
#endif

namespace clustering::math {

/**
 * @brief Type alias for the owning pool the algorithm wrappers (@c KMeans, @c DBSCAN,
 *        @c HDBSCAN) hold inside an @c std::optional.
 *
 * Selects @c citor::ThreadPool by default and @c BS::light_thread_pool when
 * @c CLUSTERING_USE_BS_POOL is defined. The two backends share an identical owning shape
 * (constructible from a single worker count) so the algorithm wrappers do not branch on
 * backend choice anywhere except the type alias.
 */
#if defined(CLUSTERING_USE_BS_POOL)
using OwnedPool = BS::light_thread_pool;
#else
using OwnedPool = citor::ThreadPool;
#endif

/**
 * @brief Clamp a caller-supplied @p nJobs to a valid worker count.
 *
 * Algorithm wrappers (@c KMeans, @c DBSCAN, @c HDBSCAN) accept @c nJobs via the
 * scikit-learn convention: @c 0 means "use hardware concurrency."
 * @c std::thread::hardware_concurrency() can itself return @c 0 on exotic targets;
 * this helper promotes that to @c 1 so the constructed pool is always valid.
 *
 * @param nJobs Caller-supplied worker count; @c 0 means "match hardware concurrency."
 * @return Non-zero worker count suitable for constructing an @c OwnedPool.
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
 * already attached, whereas this free helper decides whether to pay the one-time
 * thread-spawn cost at all. Returning @c false at small shapes lets the algorithm
 * wrappers skip tens of microseconds of pthread-create traffic a fully-serial fit would
 * never amortize.
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
 * @brief Thin compile-time-templated wrapper around the underlying @ref OwnedPool.
 *
 * Carries an optional pool pointer plus the helpers math kernels need to decide whether
 * a given workload is worth fanning out. A null pool is the explicit serial mode --
 * @c shouldParallelize then always reports @c false and the kernel runs on the calling
 * thread without touching any pool machinery.
 *
 * The dispatch API methods (@ref parallelForBlocks, @ref parallelForChunks,
 * @ref parallelRunPlex) are templated on @c HintsT and forward through to
 * @c citor::parallelFor<HintsT>(...) (or @c citor::ThreadPool::runPlex<HintsT> for the
 * persistent-worker form). On the @c CLUSTERING_USE_BS_POOL backend they map to the
 * matching @c BS::light_thread_pool primitives; the @c HintsT template parameter is
 * accepted for API uniformity and silently ignored at codegen time because BS does not
 * carry a hint concept. Each call therefore monomorphizes into the same code the direct
 * @c citor::parallelFor<HintsT> / @c BS::submit_blocks call would produce, with no
 * runtime branching on hint fields.
 *
 * @note @c workerIndex returns @c 0 outside any pool task. Callers that rely on
 *       per-worker scratch isolation must invoke it from inside a pool task body or
 *       pass @c Pool{nullptr} deliberately.
 */
struct Pool {
  /// Underlying pool, or @c nullptr to force serial execution.
  OwnedPool *pool = nullptr;

  /**
   * @brief Number of worker threads available, or @c 1 in serial mode.
   *
   * @return The pool's participant / worker count when attached, otherwise @c 1.
   */
  [[nodiscard]] std::size_t workerCount() const noexcept {
    if (pool == nullptr) {
      return std::size_t{1};
    }
#if defined(CLUSTERING_USE_BS_POOL)
    return pool->get_thread_count();
#else
    return pool->participants();
#endif
  }

  /**
   * @brief Stable index of the calling worker thread within the owning pool.
   *
   * @return The worker id reported by the backend's per-thread accessor when invoked
   *         from a pool task body, otherwise @c 0.
   */
  [[nodiscard]] static std::size_t workerIndex() noexcept {
#if defined(CLUSTERING_USE_BS_POOL)
    return BS::this_thread::get_index().value_or(std::size_t{0});
#else
    return citor::ThreadPool::workerIndex();
#endif
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
   * Complements @ref shouldParallelize by gating on total arithmetic work rather than
   * task count. At very low per-unit cost (e.g. distance kernels at @c d=2) the
   * chunk-count gate can pass while the per-worker workload is dwarfed by dispatch
   * overhead; this check prevents fan-out when the per-worker op budget would not
   * amortize the pool submit/wait syscalls.
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

  /**
   * @brief Run @p body in parallel over `[first, last)` partitioned into @p numBlocks blocks.
   *
   * The body is invoked once per block as `body(blockFirst, blockAfterLast)`. When @p pool is
   * unset the call runs the entire range inline on the calling thread. Synchronous: the call
   * returns only after every block has completed. @p HintsT is a citor hint type whose
   * static-constexpr members drive compile-time policy on the citor backend; on the BS
   * backend it is accepted for API uniformity and ignored.
   *
   * @tparam HintsT Hint type whose static-constexpr members drive compile-time policy.
   * @tparam Body   Callable invocable as `Body(std::size_t lo, std::size_t hi)`.
   * @param first     Inclusive lower bound of the iteration range.
   * @param last      Exclusive upper bound of the iteration range.
   * @param numBlocks Suggested block count; @c 0 selects `workerCount()`.
   * @param body      Callable invoked once per block.
   */
  template <class HintsT = citor::BulkBalancedHints, class Body>
  void parallelForBlocks(std::size_t first, std::size_t last, std::size_t numBlocks, Body body) {
    if (pool == nullptr || first >= last) {
      body(first, last);
      return;
    }
#if defined(CLUSTERING_USE_BS_POOL)
    const std::size_t blocks = numBlocks == 0 ? workerCount() : numBlocks;
    pool->submit_blocks(first, last, body, blocks).wait();
#else
    (void)numBlocks; // citor partitions per its own balance hint
    pool->template parallelFor<HintsT>(first, last, body);
#endif
  }

  /**
   * @brief Run @p body in parallel with exactly @p numBlocks contiguous ranges.
   *
   * Each block @c s receives `[first + (last-first)*s/numBlocks, first +
   * (last-first)*(s+1)/numBlocks)` as its `(lo, hi)` argument. The partition is identical
   * across backends, so callers that need to map a starting @p lo back to the originating
   * block index can do so deterministically (see @c kmeans::detail::BlockPartition for the
   * matching decoder).
   *
   * @tparam HintsT Hint type whose static-constexpr members drive compile-time policy.
   * @tparam Body   Callable invocable as `Body(std::size_t lo, std::size_t hi)`.
   * @param first     Inclusive lower bound of the iteration range.
   * @param last      Exclusive upper bound of the iteration range.
   * @param numBlocks Number of contiguous blocks; must be at least 1.
   * @param body      Callable invoked once per block.
   */
  template <class HintsT = citor::BulkBalancedHints, class Body>
  void parallelForExactBlocks(std::size_t first, std::size_t last, std::size_t numBlocks,
                              Body body) {
    if (first >= last || numBlocks == 0) {
      return;
    }
    if (pool == nullptr || numBlocks == 1) {
      body(first, last);
      return;
    }
    const std::size_t span = last - first;
    auto sliceLo = [first, span, numBlocks](std::size_t s) noexcept {
      return first + ((span * s) / numBlocks);
    };
#if defined(CLUSTERING_USE_BS_POOL)
    pool->submit_blocks(
            std::size_t{0}, numBlocks,
            [&](std::size_t loSlot, std::size_t hiSlot) {
              for (std::size_t s = loSlot; s < hiSlot; ++s) {
                const std::size_t blockLo = sliceLo(s);
                const std::size_t blockHi = sliceLo(s + 1);
                body(blockLo, blockHi);
              }
            },
            numBlocks)
        .wait();
#else
    pool->template parallelFor<HintsT>(std::size_t{0}, numBlocks,
                                       [&](std::size_t loSlot, std::size_t hiSlot) {
                                         for (std::size_t s = loSlot; s < hiSlot; ++s) {
                                           const std::size_t blockLo = sliceLo(s);
                                           const std::size_t blockHi = sliceLo(s + 1);
                                           body(blockLo, blockHi);
                                         }
                                       });
#endif
  }

  /**
   * @brief Slot-aware variant of @ref parallelForExactBlocks.
   *
   * Same partition semantics as @ref parallelForExactBlocks but the body receives the
   * originating slot index as a third argument. Use this when the body needs to slot
   * into per-block scratch (one slab per block) without reverse-engineering the slot
   * from the @c lo argument.
   *
   * @tparam HintsT Hint type whose static-constexpr members drive compile-time policy.
   * @tparam Body   Callable invocable as `Body(std::size_t lo, std::size_t hi, std::size_t slot)`.
   * @param first     Inclusive lower bound of the iteration range.
   * @param last      Exclusive upper bound of the iteration range.
   * @param numBlocks Number of contiguous blocks; must be at least 1.
   * @param body      Callable invoked once per block with the block's slot index.
   */
  template <class HintsT = citor::BulkBalancedHints, class Body>
  void parallelForExactBlocksWithSlot(std::size_t first, std::size_t last, std::size_t numBlocks,
                                      Body body) {
    if (first >= last || numBlocks == 0) {
      return;
    }
    if (pool == nullptr || numBlocks == 1) {
      body(first, last, std::size_t{0});
      return;
    }
    const std::size_t span = last - first;
    auto sliceLo = [first, span, numBlocks](std::size_t s) noexcept {
      return first + ((span * s) / numBlocks);
    };
#if defined(CLUSTERING_USE_BS_POOL)
    pool->submit_blocks(
            std::size_t{0}, numBlocks,
            [&](std::size_t loSlot, std::size_t hiSlot) {
              for (std::size_t s = loSlot; s < hiSlot; ++s) {
                body(sliceLo(s), sliceLo(s + 1), s);
              }
            },
            numBlocks)
        .wait();
#else
    pool->template parallelFor<HintsT>(std::size_t{0}, numBlocks,
                                       [&](std::size_t loSlot, std::size_t hiSlot) {
                                         for (std::size_t s = loSlot; s < hiSlot; ++s) {
                                           body(sliceLo(s), sliceLo(s + 1), s);
                                         }
                                       });
#endif
  }

  /**
   * @brief Run @p body once per chunk over `[0, numChunks)` in parallel.
   *
   * The body is invoked as `body(chunkIdx)` with a single integer chunk index. Used by
   * the symmetric-tile threshold path where work per chunk is heavily unbalanced (chunk 0
   * does the full triangle, the last chunk only the diagonal band) so each chunk maps to
   * one task and the pool steals across them.
   *
   * @tparam HintsT Hint type whose static-constexpr members drive compile-time policy.
   * @tparam Body   Callable invocable as `Body(std::size_t chunkIdx)`.
   * @param numChunks Total number of chunks to dispatch.
   * @param body      Callable invoked once per chunk.
   */
  template <class HintsT = citor::BulkBalancedHints, class Body>
  void parallelForChunks(std::size_t numChunks, Body body) {
    if (pool == nullptr || numChunks == 0) {
      for (std::size_t c = 0; c < numChunks; ++c) {
        body(c);
      }
      return;
    }
#if defined(CLUSTERING_USE_BS_POOL)
    pool->submit_sequence(std::size_t{0}, numChunks, [&](std::size_t c) { body(c); }).wait();
#else
    pool->template parallelFor<HintsT>(std::size_t{0}, numChunks,
                                       [&](std::size_t lo, std::size_t hi) {
                                         for (std::size_t c = lo; c < hi; ++c) {
                                           body(c);
                                         }
                                       });
#endif
  }

  /**
   * @brief Run @p phaseFn for @p nPhases persistent-worker phases over `[0, n)`.
   *
   * Maps to @c citor::ThreadPool::runPlex<HintsT> on the citor backend, which keeps
   * background workers spin-resident between phases (no per-phase futex round-trip). On
   * the BS backend the form is emulated with a per-phase @c submit_blocks loop -- the
   * work is functionally equivalent but pays a per-phase submit/wait round-trip because
   * BS lacks a persistent-plex primitive.
   *
   * The body signature matches citor's: `phaseFn(phaseIdx, slot, lo, hi, tlsArena)`.
   * `tlsArena` is always @c nullptr on the BS path; callers that need per-slot scratch
   * should derive it from @c slot themselves.
   *
   * @tparam HintsT  Hint type whose static-constexpr members drive compile-time policy.
   * @tparam Phase   Callable invocable as
   *                 `Phase(std::size_t phaseIdx, std::uint32_t slot, std::size_t lo,
   *                  std::size_t hi, void* tlsArena)`.
   * @param nPhases  Number of phases to run; @c 0 is a no-op.
   * @param n        Row-range upper bound; partitioned per-slot as
   *                 `[n*slot/P, n*(slot+1)/P)`.
   * @param phaseFn  Callable invoked once per `(phase, slot)` pair.
   */
  template <class HintsT = citor::BulkBalancedHints, class Phase>
  void parallelRunPlex(std::size_t nPhases, std::size_t n, Phase phaseFn) {
    auto noPrePhase = [](std::size_t /*phaseIdx*/) noexcept {};
    parallelRunPlex<HintsT>(nPhases, n, std::move(phaseFn), noPrePhase);
  }

  /**
   * @brief @ref parallelRunPlex with a producer-serial pre-phase hook.
   *
   * Same shape as @ref parallelRunPlex but @p prePhaseFn runs serially on the producer
   * BEFORE phase @c p publishes, with happens-before to every worker's per-phase body.
   * Use the hook to read the previous phase's per-slot results and update the shared
   * state the upcoming phase reads.
   *
   * @tparam HintsT     Hint type whose static-constexpr members drive compile-time policy.
   * @tparam Phase      Callable invocable as
   *                    `Phase(std::size_t phaseIdx, std::uint32_t slot, std::size_t lo,
   *                     std::size_t hi, void* tlsArena)`.
   * @tparam PrePhase   Callable invocable as `PrePhase(std::size_t phaseIdx)`.
   * @param nPhases     Number of phases to run; @c 0 is a no-op.
   * @param n           Row-range upper bound; partitioned per-slot.
   * @param phaseFn     Callable invoked once per `(phase, slot)` pair.
   * @param prePhaseFn  Callable invoked serially on the producer before each phase publish.
   */
  template <class HintsT = citor::BulkBalancedHints, class Phase, class PrePhase>
  void parallelRunPlex(std::size_t nPhases, std::size_t n, Phase phaseFn, PrePhase prePhaseFn) {
    if (nPhases == 0) {
      return;
    }
    if (pool == nullptr) {
      // Single-slot inline emulation when no pool is attached.
      for (std::size_t p = 0; p < nPhases; ++p) {
        prePhaseFn(p);
        phaseFn(p, std::uint32_t{0}, std::size_t{0}, n, static_cast<void *>(nullptr));
      }
      return;
    }
#if defined(CLUSTERING_USE_BS_POOL)
    const std::size_t participants = workerCount();
    if (participants <= 1) {
      for (std::size_t p = 0; p < nPhases; ++p) {
        prePhaseFn(p);
        phaseFn(p, std::uint32_t{0}, std::size_t{0}, n, static_cast<void *>(nullptr));
      }
      return;
    }
    for (std::size_t p = 0; p < nPhases; ++p) {
      prePhaseFn(p);
      pool->submit_blocks(
              std::size_t{0}, participants,
              [&, p](std::size_t loSlot, std::size_t hiSlot) {
                for (std::size_t s = loSlot; s < hiSlot; ++s) {
                  const std::size_t lo = (n * s) / participants;
                  const std::size_t hi = (n * (s + 1)) / participants;
                  phaseFn(p, static_cast<std::uint32_t>(s), lo, hi, static_cast<void *>(nullptr));
                }
              },
              participants)
          .wait();
    }
#else
    pool->template runPlex<HintsT>(nPhases, n, std::forward<Phase>(phaseFn),
                                   std::forward<PrePhase>(prePhaseFn));
#endif
  }
};

} // namespace clustering::math
