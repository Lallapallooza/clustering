#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <utility>
#include <vector>

#include "clustering/index/nn_descent/detail/neighbor_heap.h"
#include "clustering/math/detail/avx2_helpers.h"
#include "clustering/math/thread.h"
#include "clustering/ndarray.h"

namespace clustering::index::nn_descent::detail {

/**
 * @brief One NN-Descent local-join iteration over a neighbor bank.
 *
 * Implements the core of Dong, Charikar, Li (2011): for each node @c u the step collects
 * @c u 's current neighbors, then probes candidate pairs drawn from those neighbors. Candidates
 * that pass @ref NeighborHeapBank::push are admitted in place and tracked as "updated edges"
 * for convergence.
 *
 * @par Why "new" vs "old" matters
 * Admitted edges wear a "new" flag for one iteration. The join step considers only candidate
 * pairs where at least one endpoint is new; previously-admitted pairs have already been compared
 * and re-computing the same distance would not discover new admissions. After the iteration the
 * caller ages the epoch so every admitted-this-round entry becomes "old" for the next round.
 *
 * @par Concurrency model
 * Parallelises over the outer @c u loop via @ref math::Pool. Each worker collects its own
 * candidate pairs into a thread-local buffer and flushes them at the end of its chunk. The heap
 * bank is written under a light mutex during the flush because admissions can hit the same
 * node's heap from multiple workers. For the target workload where per-pair distance compute
 * dominates, the mutex is a second-order cost; concrete TSan-safe and straightforward. No
 * nested pool dispatch: the worker never calls @c submit_blocks.
 *
 * @par Reciprocal admission
 * Every admitted @c (u, v) edge is also pushed as @c (v, u), which is the standard Dong 2011
 * bidirectional join. Without reciprocal admission, the join graph becomes asymmetric and
 * convergence stalls.
 *
 * @tparam T Element type of the point cloud (must be @c float or @c double).
 */
template <class T> struct JoinStep {
  /**
   * @brief One iteration of the local-join loop.
   *
   * @param X    @c n x d contiguous point cloud; not owned.
   * @param bank Per-node neighbor heaps; updated in place.
   * @param pool Parallelism injection.
   * @return Number of neighbor-slot updates applied this iteration (across every node). Used by
   *         the caller for convergence: when @c updates / (n * k) < delta the loop terminates.
   */
  static std::size_t run(const NDArray<T, 2> &X, NeighborHeapBank<T> &bank, math::Pool pool) {
    const std::size_t n = bank.n();
    const std::size_t k = bank.k();
    if (n == 0 || k == 0) {
      return 0;
    }
    const std::size_t d = X.dim(1);
    const T *data = X.data();

    // Partition each node's neighbors into "new" (admitted last iteration) and "old" for this
    // iteration's local-join. Both lists are flattened into a single (sorted-by-node) buffer.
    std::vector<std::vector<std::int32_t>> newList(n);
    std::vector<std::vector<std::int32_t>> oldList(n);
    for (std::size_t i = 0; i < n; ++i) {
      const auto ii = static_cast<std::int32_t>(i);
      const std::size_t sz = bank.sizeAt(ii);
      newList[i].reserve(sz);
      oldList[i].reserve(sz);
      for (std::size_t s = 0; s < sz; ++s) {
        const std::int32_t nb = bank.idxAt(ii, s);
        if (bank.isNew(ii, s)) {
          newList[i].push_back(nb);
        } else {
          oldList[i].push_back(nb);
        }
      }
    }

    // Age the bank's epoch so the next call sees only freshly-admitted entries as "new." The
    // snapshot above captured this iteration's new set; subsequent admissions retag entries.
    bank.ageEpoch();

    // Build reverse neighbor lists: for each node u, include every node v that has u among its
    // k-nearest. Local-join explores both forward and reverse neighbors per Dong 2011.
    std::vector<std::vector<std::int32_t>> reverseNew(n);
    std::vector<std::vector<std::int32_t>> reverseOld(n);
    for (std::size_t u = 0; u < n; ++u) {
      for (const std::int32_t v : newList[u]) {
        if (v >= 0 && std::cmp_less(v, n)) {
          reverseNew[static_cast<std::size_t>(v)].push_back(static_cast<std::int32_t>(u));
        }
      }
      for (const std::int32_t v : oldList[u]) {
        if (v >= 0 && std::cmp_less(v, n)) {
          reverseOld[static_cast<std::size_t>(v)].push_back(static_cast<std::int32_t>(u));
        }
      }
    }

    // Bank mutex sharding by target node. Two pushes to different targets touch disjoint
    // heap slots in the bank and are independent; sharding by @c target % kShardCount lets
    // disjoint-target pushes from different workers proceed in parallel instead of serializing
    // on a single mutex.
    constexpr std::size_t kShardCount = 64;
    std::array<std::mutex, kShardCount> bankShards;
    std::atomic<std::size_t> totalUpdates{0};

    const bool serial = (pool.pool == nullptr);

    auto runRange = [&](std::size_t lo, std::size_t hi) {
      // Parallel-path staging buffer, flushed under the bank mutex at the end of each node to
      // keep peak occupancy at O(k^2) and the mutex hold bounded per flush.
      struct Cand {
        std::int32_t target;
        std::int32_t candidate;
        T sqDist;
      };
      std::vector<Cand> cands;
      if (!serial) {
        cands.reserve(8 * k * k);
      }

      // Fused forward + reverse neighbor list per node. Deduplication within the merged list is
      // handled by the bank's push uniqueness check; we can push duplicates cheaply because the
      // admission gate rejects them.
      std::vector<std::int32_t> mergedNew;
      std::vector<std::int32_t> mergedOld;
      mergedNew.reserve(4 * k);
      mergedOld.reserve(4 * k);

      std::size_t localUpdates = 0;

      auto emit = [&](std::int32_t target, std::int32_t candidate, T sqDist) {
        if (serial) {
          if (bank.push(target, candidate, sqDist)) {
            ++localUpdates;
          }
        } else {
          cands.push_back(Cand{target, candidate, sqDist});
        }
      };

      for (std::size_t uSize = lo; uSize < hi; ++uSize) {
        mergedNew.clear();
        mergedOld.clear();
        mergedNew.insert(mergedNew.end(), newList[uSize].begin(), newList[uSize].end());
        mergedNew.insert(mergedNew.end(), reverseNew[uSize].begin(), reverseNew[uSize].end());
        mergedOld.insert(mergedOld.end(), oldList[uSize].begin(), oldList[uSize].end());
        mergedOld.insert(mergedOld.end(), reverseOld[uSize].begin(), reverseOld[uSize].end());

        // new x new: every pair with both endpoints flagged new this iteration.
        for (std::size_t a = 0; a < mergedNew.size(); ++a) {
          const std::int32_t va = mergedNew[a];
          const T *pa = data + (static_cast<std::size_t>(va) * d);
          for (std::size_t b = a + 1; b < mergedNew.size(); ++b) {
            const std::int32_t vb = mergedNew[b];
            if (va == vb) {
              continue;
            }
            const T *pb = data + (static_cast<std::size_t>(vb) * d);
            const T sqDist = math::detail::sqEuclideanRowPtr(pa, pb, d);
            emit(va, vb, sqDist);
            emit(vb, va, sqDist);
          }
        }
        // new x old: every cross pair where the new endpoint has not yet been compared to the
        // old endpoint.
        for (const std::int32_t va : mergedNew) {
          const T *pa = data + (static_cast<std::size_t>(va) * d);
          for (const std::int32_t vb : mergedOld) {
            if (va == vb) {
              continue;
            }
            const T *pb = data + (static_cast<std::size_t>(vb) * d);
            const T sqDist = math::detail::sqEuclideanRowPtr(pa, pb, d);
            emit(va, vb, sqDist);
            emit(vb, va, sqDist);
          }
        }

        // Per-node flush for the parallel path. Cands are grouped by shard (target %
        // kShardCount) so each shard's mutex is acquired at most once per flush and
        // disjoint-target workers make progress concurrently.
        if (!serial && !cands.empty()) {
          std::array<std::vector<const Cand *>, kShardCount> buckets;
          for (const Cand &c : cands) {
            buckets[static_cast<std::size_t>(c.target) % kShardCount].push_back(&c);
          }
          for (std::size_t s = 0; s < kShardCount; ++s) {
            if (buckets[s].empty()) {
              continue;
            }
            const std::scoped_lock lock(bankShards[s]);
            for (const Cand *c : buckets[s]) {
              if (bank.push(c->target, c->candidate, c->sqDist)) {
                ++localUpdates;
              }
            }
          }
          cands.clear();
        }
      }

      totalUpdates.fetch_add(localUpdates, std::memory_order_relaxed);
    };

    // Work gate: parallelism only when the per-worker work amortizes pool dispatch.
    if (pool.shouldParallelize(n, 64, 2) && pool.pool != nullptr) {
      pool.pool
          ->submit_blocks(std::size_t{0}, n,
                          [&](std::size_t lo, std::size_t hi) { runRange(lo, hi); })
          .wait();
    } else {
      runRange(0, n);
    }

    return totalUpdates.load(std::memory_order_relaxed);
  }
};

} // namespace clustering::index::nn_descent::detail
