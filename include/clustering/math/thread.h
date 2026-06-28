#pragma once

#include <citor/hints.h>
#include <citor/thread_pool.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "clustering/math/detail/coherence_cache.h"

namespace clustering::math {

/**
 * @brief Type alias for the owning pool the algorithm wrappers (@c KMeans, @c DBSCAN,
 *        @c HDBSCAN) hold inside an @c std::optional.
 *
 * Aliases @c citor::ThreadPool, constructible from a single worker count.
 */
using OwnedPool = citor::ThreadPool;

/**
 * @brief Process-wide pool registry, keyed by worker count.
 *
 * Returns the same @ref OwnedPool every time it's called with a given @p nJobs. Algorithm
 * wrappers (@c KMeans, @c DBSCAN, @c HDBSCAN) borrow from this registry by default so a
 * Python harness or repeated C++ caller doesn't pay the per-call thread-spawn cost. The
 * registry is created lazily on first request and lives for the duration of the process;
 * destroying long-lived worker pools is expensive enough that we deliberately leak the
 * lifetime to the loader.
 *
 * Callers who need a private pool (tests, embedded use, multi-tenant scoping) construct
 * their own @ref OwnedPool and pass it to the algorithm's external-pool constructor.
 *
 * @param nJobs Worker count; passed through @ref clampedJobCount before lookup.
 * @return Reference to the cached pool. Stable across calls with the same effective @p nJobs.
 */
inline OwnedPool &sharedPool(std::size_t nJobs);

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
 * persistent-worker form). Each call monomorphizes into the same code a direct
 * @c citor::parallelFor<HintsT> call would produce, with no runtime branching on hint
 * fields.
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
    return pool->participants();
  }

  /**
   * @brief Stable index of the calling worker thread within the owning pool.
   *
   * @return The worker id reported by the backend's per-thread accessor when invoked
   *         from a pool task body, otherwise @c 0.
   */
  [[nodiscard]] static std::size_t workerIndex() noexcept {
    return citor::ThreadPool::workerIndex();
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
   * @param numBlocks Requested block count; @c 0 selects the backend's default partition.
   * @param body      Callable invoked once per block.
   */
  template <class HintsT = citor::HintsDefaults, class Body>
  void parallelForBlocks(std::size_t first, std::size_t last, std::size_t numBlocks, Body body) {
    if (pool == nullptr || first >= last) {
      body(first, last);
      return;
    }
    if (numBlocks != 0) {
      parallelForExactBlocks<HintsT>(first, last, numBlocks, std::move(body));
      return;
    }
    pool->template parallelFor<HintsT>(first, last, body);
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
  template <class HintsT = citor::HintsDefaults, class Body>
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
    pool->template parallelFor<HintsT>(std::size_t{0}, numBlocks,
                                       [&](std::size_t loSlot, std::size_t hiSlot) {
                                         for (std::size_t s = loSlot; s < hiSlot; ++s) {
                                           const std::size_t blockLo = sliceLo(s);
                                           const std::size_t blockHi = sliceLo(s + 1);
                                           body(blockLo, blockHi);
                                         }
                                       });
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
  template <class HintsT = citor::HintsDefaults, class Body>
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
    pool->template parallelFor<HintsT>(std::size_t{0}, numBlocks,
                                       [&](std::size_t loSlot, std::size_t hiSlot) {
                                         for (std::size_t s = loSlot; s < hiSlot; ++s) {
                                           body(sliceLo(s), sliceLo(s + 1), s);
                                         }
                                       });
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
  template <class HintsT = citor::HintsDefaults, class Body>
  void parallelForChunks(std::size_t numChunks, Body body) {
    if (pool == nullptr || numChunks == 0) {
      for (std::size_t c = 0; c < numChunks; ++c) {
        body(c);
      }
      return;
    }
    pool->template parallelFor<HintsT>(std::size_t{0}, numChunks,
                                       [&](std::size_t lo, std::size_t hi) {
                                         for (std::size_t c = lo; c < hi; ++c) {
                                           body(c);
                                         }
                                       });
  }

  /**
   * @brief Reduce `[first, last)` with the backend's reduction primitive.
   *
   * On the citor backend this forwards to `ThreadPool::parallelReduce<HintsT>`, preserving
   * citor's chunk-id tree and determinism policy. On the BS backend the same surface is
   * emulated with one partial per worker-count slot, then folded by the producer in slot order.
   *
   * @tparam HintsT  citor reduction hint type, e.g. `citor::HintsDefaults`.
   * @tparam T       Reduction value type.
   * @tparam Map     Callable invocable as `T(std::size_t lo, std::size_t hi)`.
   * @tparam Combine Callable invocable as `T(T a, T b)`.
   * @param first    Inclusive lower bound of the iteration range.
   * @param last     Exclusive upper bound of the iteration range.
   * @param init     Identity value.
   * @param map      Per-block mapper.
   * @param combine  Producer-side combiner.
   * @return Combined reduction value.
   */
  template <class HintsT = citor::HintsDefaults, class T, class Map, class Combine>
  [[nodiscard]] T parallelReduce(std::size_t first, std::size_t last, T init, Map map,
                                 Combine combine) {
    if (first >= last) {
      return init;
    }
    if (pool == nullptr) {
      return combine(std::move(init), map(first, last));
    }
    return pool->template parallelReduce<HintsT>(first, last, std::move(init), std::move(map),
                                                 std::move(combine));
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
  template <class HintsT = citor::HintsDefaults, class Phase>
  void parallelRunPlex(std::size_t nPhases, std::size_t n, Phase phaseFn) {
    auto noPrePhase = [](std::size_t /*phaseIdx*/) noexcept {};
    parallelRunPlex<HintsT>(nPhases, n, std::move(phaseFn), noPrePhase);
  }

  /**
   * @brief Two-pass exclusive-prefix scan over `[0, n)`.
   *
   * Forwards to `citor::ThreadPool::parallelScan<HintsT>`: pass 1 invokes @p body once per
   * slot with `initial = @p identity` to compute the chunk's partial; the producer then
   * runs an `O(slots)` reduce via @p prefix; pass 2 re-invokes @p body with `initial` set to
   * the chunk's exclusive prefix so the body can finish writing its slice. Returns the
   * inclusive accumulator at the right edge.
   *
   * On the BS backend the form is emulated with one `parallelForExactBlocksWithSlot` for the
   * first pass, a serial prefix walk on the producer, and a second
   * `parallelForExactBlocksWithSlot` for the offset add. Output is functionally equivalent;
   * the serialization between passes is identical to citor's.
   *
   * @tparam HintsT   Hint type whose static-constexpr members drive compile-time policy.
   * @tparam T        Reduction value type.
   * @tparam BodyFn   Callable: `T(chunkId, lo, hi, initial, out)`. `out` is unused on the BS
   *                  path (always `nullptr`) and the citor path (the body owns the
   *                  destination buffer captured by reference); kept in the signature so
   *                  citor's CPO surface monomorphizes identically.
   * @tparam PrefixFn Callable: `T(T a, T b)` cross-chunk reduce.
   * @param n        Range length.
   * @param identity Identity value seeded into pass 1's body and returned for empty ranges.
   * @param body     Per-chunk body invoked twice (once per pass).
   * @param prefix   Cross-chunk binary combiner.
   * @return Inclusive accumulator at the right edge of the scan.
   */
  template <class HintsT = citor::HintsDefaults, class T, class BodyFn, class PrefixFn>
  T parallelScan(std::size_t n, T identity, BodyFn body, PrefixFn prefix) {
    if (n == 0) {
      return identity;
    }
    if (pool == nullptr) {
      T partial = body(std::size_t{0}, std::size_t{0}, n, identity, static_cast<T *>(nullptr));
      return prefix(std::move(identity), std::move(partial));
    }
    return pool->template parallelScan<HintsT>(n, std::move(identity), std::move(body),
                                               std::move(prefix));
  }

  template <class HintsT = citor::HintsDefaults, class Phase, class PrePhase>
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
    pool->template runPlex<HintsT>(nPhases, n, std::forward<Phase>(phaseFn),
                                   std::forward<PrePhase>(prePhaseFn));
  }
};

inline OwnedPool &sharedPool(std::size_t nJobs) {
  const std::size_t effective = clampedJobCount(nJobs);
  // Inline-static map: shared across translation units thanks to the inline keyword on the
  // enclosing function. Mutex protects the registry against first-time-init races; lookups
  // are O(1) and gated by Python's GIL or other caller-side serialization in practice.
  static std::unordered_map<std::size_t, std::unique_ptr<OwnedPool>> registry;
  static std::mutex registryMutex;
  const std::scoped_lock guard{registryMutex};
  auto &slot = registry[effective];
  if (!slot) {
    // Seed citor's coherence-probe cache from disk before constructing the pool
    // so a short-lived or single-fit process skips the live inter-core
    // calibration. On a cold cache the ctor runs the probe; persist the fresh
    // result so the next process replays it.
    const bool seeded = detail::importPersistedCoherenceProbe(effective);
    slot = std::make_unique<OwnedPool>(effective);
    if (!seeded) {
      detail::exportPersistedCoherenceProbe(*slot, effective);
    }
  }
  return *slot;
}

} // namespace clustering::math
