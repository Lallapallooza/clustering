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
    // iteration's local-join in a CSR-style packed layout. Two pairs of (offset, length) arrays
    // (one for new, one for old) point into a single contiguous buffer per epoch. The buffer
    // length per node is at most @c k, so a flat (n * k) allocation upper-bounds storage and the
    // exact length per node is recorded in @c newLen[i] / @c oldLen[i].
    std::vector<std::int32_t> newBuf(n * k);
    std::vector<std::int32_t> oldBuf(n * k);
    std::vector<std::int32_t> newLen(n, 0);
    std::vector<std::int32_t> oldLen(n, 0);

    auto buildForwardRange = [&](std::size_t lo, std::size_t hi) {
      for (std::size_t i = lo; i < hi; ++i) {
        const auto ii = static_cast<std::int32_t>(i);
        const std::size_t sz = bank.sizeAt(ii);
        std::int32_t *newRow = newBuf.data() + (i * k);
        std::int32_t *oldRow = oldBuf.data() + (i * k);
        std::int32_t nl = 0;
        std::int32_t ol = 0;
        for (std::size_t s = 0; s < sz; ++s) {
          const std::int32_t nb = bank.idxAt(ii, s);
          if (bank.isNew(ii, s)) {
            newRow[nl++] = nb;
          } else {
            oldRow[ol++] = nb;
          }
        }
        newLen[i] = nl;
        oldLen[i] = ol;
      }
    };

    if (pool.shouldParallelize(n, 256, 2) && pool.pool != nullptr) {
      pool.pool->submit_blocks(std::size_t{0}, n, buildForwardRange).wait();
    } else {
      buildForwardRange(0, n);
    }

    // Age the bank's epoch so the next call sees only freshly-admitted entries as "new." The
    // snapshot above captured this iteration's new set; subsequent admissions retag entries.
    bank.ageEpoch();

    // Build reverse neighbor lists in CSR form. Forward (i -> v) becomes reverse (v -> i). Two
    // passes: counting (atomic increments per v) followed by scatter (atomic-fetch-add to claim
    // a slot per v). Output is two CSR-style buffers (offsets, indices) per epoch.
    std::vector<std::atomic<std::int32_t>> revNewCnt(n);
    std::vector<std::atomic<std::int32_t>> revOldCnt(n);
    auto countRange = [&](std::size_t lo, std::size_t hi) {
      for (std::size_t u = lo; u < hi; ++u) {
        const std::int32_t nl = newLen[u];
        const std::int32_t *newRow = newBuf.data() + (u * k);
        for (std::int32_t s = 0; s < nl; ++s) {
          const std::int32_t v = newRow[s];
          if (v >= 0 && std::cmp_less(v, n)) {
            revNewCnt[static_cast<std::size_t>(v)].fetch_add(1, std::memory_order_relaxed);
          }
        }
        const std::int32_t ol = oldLen[u];
        const std::int32_t *oldRow = oldBuf.data() + (u * k);
        for (std::int32_t s = 0; s < ol; ++s) {
          const std::int32_t v = oldRow[s];
          if (v >= 0 && std::cmp_less(v, n)) {
            revOldCnt[static_cast<std::size_t>(v)].fetch_add(1, std::memory_order_relaxed);
          }
        }
      }
    };

    if (pool.shouldParallelize(n, 256, 2) && pool.pool != nullptr) {
      pool.pool->submit_blocks(std::size_t{0}, n, countRange).wait();
    } else {
      countRange(0, n);
    }

    // Prefix sum to convert per-v counts to CSR offsets. Serial; n integer adds is cheap.
    std::vector<std::int32_t> revNewOffsets(n + 1, 0);
    std::vector<std::int32_t> revOldOffsets(n + 1, 0);
    {
      std::int32_t accNew = 0;
      std::int32_t accOld = 0;
      for (std::size_t v = 0; v < n; ++v) {
        revNewOffsets[v] = accNew;
        revOldOffsets[v] = accOld;
        accNew += revNewCnt[v].load(std::memory_order_relaxed);
        accOld += revOldCnt[v].load(std::memory_order_relaxed);
      }
      revNewOffsets[n] = accNew;
      revOldOffsets[n] = accOld;
    }

    std::vector<std::int32_t> revNewIdx(static_cast<std::size_t>(revNewOffsets[n]));
    std::vector<std::int32_t> revOldIdx(static_cast<std::size_t>(revOldOffsets[n]));

    // Reset the per-v counters; we'll reuse them as fetch-add cursors during scatter so each
    // worker writes into a unique slot of the CSR index buffer.
    for (std::size_t v = 0; v < n; ++v) {
      revNewCnt[v].store(0, std::memory_order_relaxed);
      revOldCnt[v].store(0, std::memory_order_relaxed);
    }

    auto scatterRange = [&](std::size_t lo, std::size_t hi) {
      for (std::size_t u = lo; u < hi; ++u) {
        const std::int32_t nl = newLen[u];
        const std::int32_t *newRow = newBuf.data() + (u * k);
        for (std::int32_t s = 0; s < nl; ++s) {
          const std::int32_t v = newRow[s];
          if (v >= 0 && std::cmp_less(v, n)) {
            const auto vSize = static_cast<std::size_t>(v);
            const auto slot =
                static_cast<std::size_t>(revNewCnt[vSize].fetch_add(1, std::memory_order_relaxed));
            revNewIdx[static_cast<std::size_t>(revNewOffsets[vSize]) + slot] =
                static_cast<std::int32_t>(u);
          }
        }
        const std::int32_t ol = oldLen[u];
        const std::int32_t *oldRow = oldBuf.data() + (u * k);
        for (std::int32_t s = 0; s < ol; ++s) {
          const std::int32_t v = oldRow[s];
          if (v >= 0 && std::cmp_less(v, n)) {
            const auto vSize = static_cast<std::size_t>(v);
            const auto slot =
                static_cast<std::size_t>(revOldCnt[vSize].fetch_add(1, std::memory_order_relaxed));
            revOldIdx[static_cast<std::size_t>(revOldOffsets[vSize]) + slot] =
                static_cast<std::int32_t>(u);
          }
        }
      }
    };

    if (pool.shouldParallelize(n, 256, 2) && pool.pool != nullptr) {
      pool.pool->submit_blocks(std::size_t{0}, n, scatterRange).wait();
    } else {
      scatterRange(0, n);
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
        const std::int32_t *newRow = newBuf.data() + (uSize * k);
        const std::int32_t *oldRow = oldBuf.data() + (uSize * k);
        mergedNew.insert(mergedNew.end(), newRow, newRow + newLen[uSize]);
        const std::int32_t *revNewRow = revNewIdx.data() + revNewOffsets[uSize];
        const std::int32_t revNewSize = revNewOffsets[uSize + 1] - revNewOffsets[uSize];
        mergedNew.insert(mergedNew.end(), revNewRow, revNewRow + revNewSize);
        mergedOld.insert(mergedOld.end(), oldRow, oldRow + oldLen[uSize]);
        const std::int32_t *revOldRow = revOldIdx.data() + revOldOffsets[uSize];
        const std::int32_t revOldSize = revOldOffsets[uSize + 1] - revOldOffsets[uSize];
        mergedOld.insert(mergedOld.end(), revOldRow, revOldRow + revOldSize);

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
