#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <future>
#include <limits>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include "clustering/always_assert.h"
#include "clustering/hdbscan/mst_output.h"
#include "clustering/index/kdtree.h"
#include "clustering/math/detail/avx2_helpers.h"
#include "clustering/math/detail/sq_distances_block.h"
#include "clustering/math/thread.h"
#include "clustering/ndarray.h"

namespace clustering::hdbscan {

/**
 * @brief Compute budget that gates the streaming Prim backend, expressed as the maximum point
 *        count it will accept.
 *
 * The streaming Prim variant materialises only `O(n)` state (no @c n*n MRD matrix), but the
 * inner relax recomputes @c d-wide squared-Euclidean distances per popped vertex, costing
 * `O(n^2 * d)` total scalar work. Beyond this @c n the dispatcher should prefer NN-Descent
 * (high @c d) or Boruvka (low @c d), both of which scale better than `O(n^2)`.
 */
inline constexpr std::size_t kPrimMaxN = std::size_t{16384};

/// Equivalent byte-budget phrasing of @ref kPrimMaxN, kept so callers that gate on
/// @c n*n*sizeof(T) <= kPrimMrdMatrixByteBudget compile unchanged. Numerically equal to
/// @ref kPrimMaxN squared, scaled by `sizeof(float)`.
inline constexpr std::size_t kPrimMrdMatrixByteBudget = kPrimMaxN * kPrimMaxN * sizeof(float);

/**
 * @brief Thresholds gating the dense symmetric core-distance pass.
 *
 * The dense pass computes all @c n*(n-1)/2 squared distances once and feeds both endpoints into
 * a per-row top-@c minSamples tracker. Its total work is `O(n^2 * d)` distance compute plus
 * `O(n^2 * minSamples)` top-@c k bookkeeping. It beats the @ref KDTree kNN fallback once
 * @c minSamples is small enough for the top-@c k write amp to stay bounded and `(n, d)` is
 * large enough to amortise the symmetric scan's fixed cost. Below these bounds the KDTree path
 * wins; above @c kPrimDenseCoreMaxMinSamples the top-@c k rescan per update dominates.
 */
inline constexpr std::size_t kPrimDenseCoreMinN = 1024;
inline constexpr std::size_t kPrimDenseCoreMinD = 17;
inline constexpr std::size_t kPrimDenseCoreMaxMinSamples = 64;
inline constexpr std::size_t kPrimPersistentRelaxMinWorkers = 8;
inline constexpr std::size_t kPrimPersistentRelaxMinOpsPerWorker = std::size_t{1} << 14;

/**
 * @brief Exact minimum-spanning-tree backend over mutual-reachability distance, streaming Prim.
 *
 * Given a point matrix @p X and a core-distance parameter @p minSamples, the backend:
 *   1. Computes per-point core distances as the @c minSamples-th nearest-neighbour distance
 *      (self-excluded, squared). The large-@c n / small-@c minSamples path uses an exact dense
 *      symmetric pass; other shapes fall back to @ref KDTree::knnQuery.
 *   2. Runs Prim's algorithm rooted at vertex @c 0. Per popped vertex, the relax step
 *      recomputes the @c d-wide squared Euclidean distance to every unvisited point on the fly
 *      (no @c n*n matrix is ever materialised) and lifts to MRD via the per-pair @c max with
 *      both core distances. The dense-core path reuses row norms and dot products for the
 *      squared-distance identity.
 *   3. Selects the next vertex while relaxing the current row on the serial path, avoiding a
 *      second full scan over @c edgeWeight.
 *
 * @par Memory
 * Working state is `O(n)` plus `O(n * minSamples)` while dense core distances are computed.
 * No @c n*n distance matrix is materialised.
 *
 * @par Cache
 * Each Prim iteration streams @c X sequentially in @c d-wide rows. Total bytes touched per
 * iteration is @c n * d * sizeof(T), which fits in @c L2 for the dispatcher's `(n, d)` window.
 *
 * The backend satisfies @ref MstBackendStrategy and carries no persistent state.
 *
 * @tparam T Element type of the point cloud. Only @c float is supported.
 */
template <class T> class PrimMstBackend {
  static_assert(std::is_same_v<T, float>,
                "PrimMstBackend<T> supports only float; a double specialization is out of scope.");

public:
  PrimMstBackend() = default;

  /**
   * @brief Build the MRD-weighted minimum spanning tree of @p X.
   *
   * @pre @p minSamples is positive and strictly less than `X.dim(0)`.
   * @pre `X.dim(0)` does not exceed @c kPrimMaxN. A violation fires
   *      @c CLUSTERING_ALWAYS_ASSERT before any allocation.
   *
   * @param X          Contiguous `(n x d)` dataset; caller retains ownership.
   * @param minSamples Neighbour count driving the core-distance definition.
   * @param pool       Worker pool; forwarded to the KDTree build and kNN query.
   * @param out        Destination; @c edges filled with @c n - 1 entries in insertion order and
   *                   @c coreDistances sized to @c n.
   */
  void run(const NDArray<T, 2> &X, std::size_t minSamples, math::Pool pool, MstOutput<T> &out) {
    const std::size_t n = X.dim(0);
    const std::size_t d = X.dim(1);
    CLUSTERING_ALWAYS_ASSERT(minSamples >= 1);
    CLUSTERING_ALWAYS_ASSERT(minSamples < n);

    // Refuse @c n that would push the `O(n^2 * d)` inner work past the dispatcher's intended
    // Prim window. Phrased as `n <= kNsqBudget` / n rather than @c n*n <= kNsqBudget to avoid
    // the intermediate overflowing @c std::size_t at large @c n. Fires before any allocation so
    // out-of-budget callers surface deterministically.
    constexpr std::size_t kNsqBudget = kPrimMrdMatrixByteBudget / sizeof(T);
    CLUSTERING_ALWAYS_ASSERT(n <= kNsqBudget / n);

    out.edges.clear();
    out.edges.reserve(n - 1);
    out.coreDistances = NDArray<T, 1>(std::array<std::size_t, 1>{n});
    T *coreDistData = out.coreDistances.data();
    const T *xData = X.data();
    const bool useDenseCore = shouldUseDenseCore(n, d, minSamples);
    // Row @c i starts at @c xData + i*d, so every row is 32-byte aligned iff the base pointer
    // is aligned and the row stride @c d*sizeof(T) is a multiple of 32. Either condition can
    // fail independently: NumPy buffers only guarantee element alignment, and @c d is caller-
    // driven. When both hold we can use the strict-aligned dot kernel; otherwise the generic
    // kernel's per-operand alignment check is required to stay correct.
    const bool rowsAligned32 =
        X.template isAligned<32>() && (d % (std::size_t{32} / sizeof(T)) == 0);

    std::vector<T> rowNorms;
    if (useDenseCore) {
      rowNorms.resize(n);
      for (std::size_t i = 0; i < n; ++i) {
        const T *row = xData + (i * d);
        rowNorms[i] = rowsAligned32 ? math::detail::dotRowAligned32Ptr(row, row, d)
                                    : math::detail::dotRowPtr(row, row, d);
      }
      computeDenseCoreDistances(X, rowNorms, minSamples, rowsAligned32, pool, coreDistData);
    } else {
      // Shapes that fail @c shouldUseDenseCore take the KDTree kNN path: the dense symmetric
      // scan does not amortise at small @c n, low @c d, or large @c minSamples (where the per-
      // update top-@c k rescan dominates).
      const KDTree<T> tree(X);
      const auto kSigned = static_cast<std::int32_t>(minSamples);
      auto [knnIdx, knnSqDist] = tree.knnQuery(kSigned, pool);
      (void)knnIdx;
      for (std::size_t i = 0; i < n; ++i) {
        coreDistData[i] = knnSqDist(i, minSamples - 1);
      }
    }

    // Phase 2: streaming Prim. Maintain `edgeWeight[v]` = best-known incident MRD weight to
    // the growing tree, `parent[v]` = the in-tree vertex realising that weight, and a visited
    // bitmap. Each iteration picks the smallest-weight unvisited @c target via a linear scan,
    // emits the edge `(parent[target], target, edgeWeight[target])`, then relaxes every other
    // unvisited @c v by recomputing `sqDist(target, v)` and lifting to MRD.
    std::vector<std::uint8_t> visited(n, std::uint8_t{0});
    std::vector<std::int32_t> parent(n, std::int32_t{0});
    std::vector<T> edgeWeight(n, std::numeric_limits<T>::max());

    auto sqDistance = [&](const T *rowT, std::size_t tIdx, std::size_t v) noexcept {
      if (useDenseCore) {
        const T *rowV = xData + (v * d);
        const T dot = rowsAligned32 ? math::detail::dotRowAligned32Ptr(rowT, rowV, d)
                                    : math::detail::dotRowPtr(rowT, rowV, d);
        return math::detail::sqEuclideanFromDot(rowNorms[tIdx], rowNorms[v], dot);
      }
      return math::detail::sqEuclideanRowPtr(rowT, xData + (v * d), d);
    };

    auto relaxRange = [&](std::size_t lo, std::size_t hi, std::int32_t target, std::size_t tIdx,
                          T coreT, const T *rowT) noexcept {
      for (std::size_t v = lo; v < hi; ++v) {
        if (visited[v] != 0U) {
          continue;
        }
        const T sq = sqDistance(rowT, tIdx, v);
        T w = sq;
        if (coreT > w) {
          w = coreT;
        }
        const T coreV = coreDistData[v];
        if (coreV > w) {
          w = coreV;
        }
        if (w < edgeWeight[v]) {
          parent[v] = target;
          edgeWeight[v] = w;
        }
      }
    };

    auto relaxRangeAndFindNext = [&](std::size_t lo, std::size_t hi, std::int32_t target,
                                     std::size_t tIdx, T coreT,
                                     const T *rowT) noexcept -> std::pair<std::int32_t, T> {
      std::int32_t bestV = -1;
      T bestW = std::numeric_limits<T>::max();
      for (std::size_t v = lo; v < hi; ++v) {
        if (visited[v] != 0U) {
          continue;
        }
        const T sq = sqDistance(rowT, tIdx, v);
        T w = sq;
        if (coreT > w) {
          w = coreT;
        }
        const T coreV = coreDistData[v];
        if (coreV > w) {
          w = coreV;
        }
        if (w < edgeWeight[v]) {
          parent[v] = target;
          edgeWeight[v] = w;
        }
        if (edgeWeight[v] < bestW) {
          bestW = edgeWeight[v];
          bestV = static_cast<std::int32_t>(v);
        }
      }
      return {bestV, bestW};
    };

    auto findNext = [&]() noexcept -> std::pair<std::int32_t, T> {
      std::int32_t bestV = -1;
      T bestW = std::numeric_limits<T>::max();
      for (std::size_t v = 0; v < n; ++v) {
        if (visited[v] != 0U) {
          continue;
        }
        if (edgeWeight[v] < bestW) {
          bestW = edgeWeight[v];
          bestV = static_cast<std::int32_t>(v);
        }
      }
      return {bestV, bestW};
    };

    auto persistentRelaxFrom = [&]() -> bool {
      if (!shouldUsePersistentParallelRelax(n, d, useDenseCore, pool)) {
        return false;
      }

      // Pack the per-worker barrier flag into the same 64 B cache line as its local-best
      // reduction slot. Each line is written only by its owning participant (worker or main),
      // and read by main only when @c done matches the current phase counter, so the release
      // store on @c done synchronises the non-atomic @c vertex and @c weight writes that
      // preceded it. Per-line ownership eliminates the cross-core RMW contention a shared
      // @c completed counter would incur on each phase close.
      struct alignas(64) LocalBest {
        std::atomic<std::uint32_t> done{0};
        std::int32_t vertex = -1;
        T weight = std::numeric_limits<T>::max();
      };

      const std::size_t workerTasks = pool.workerCount() - 1;
      const std::size_t participantCount = workerTasks + 1;
      std::vector<LocalBest> localBest(participantCount);

      auto blockBegin = [&](std::size_t id) noexcept { return (n * id) / participantCount; };
      auto blockEnd = [&](std::size_t id) noexcept { return (n * (id + 1)) / participantCount; };
      auto relaxBlock = [&](std::size_t id,
                            std::int32_t target) noexcept -> std::pair<std::int32_t, T> {
        const auto tIdx = static_cast<std::size_t>(target);
        const T coreT = coreDistData[tIdx];
        const T *const rowT = xData + (tIdx * d);
        std::int32_t bestV = -1;
        T bestW = std::numeric_limits<T>::max();
        for (std::size_t v = blockBegin(id); v < blockEnd(id); ++v) {
          if (visited[v] != 0U) {
            continue;
          }
          const T sq = sqDistance(rowT, tIdx, v);
          T w = sq;
          if (coreT > w) {
            w = coreT;
          }
          const T coreV = coreDistData[v];
          if (coreV > w) {
            w = coreV;
          }
          if (w < edgeWeight[v]) {
            parent[v] = target;
            edgeWeight[v] = w;
          }
          if (edgeWeight[v] < bestW) {
            bestW = edgeWeight[v];
            bestV = static_cast<std::int32_t>(v);
          }
        }
        return {bestV, bestW};
      };

      std::atomic<std::uint32_t> phase{0};
      std::atomic<std::uint32_t> ready{0};
      std::atomic<bool> stop{false};
      std::int32_t currentTarget = 0;

      auto workerLoop = [&](std::size_t id) {
        std::uint32_t seen = phase.load(std::memory_order_acquire);
        ready.fetch_add(1, std::memory_order_release);
        for (;;) {
          std::uint32_t next = phase.load(std::memory_order_acquire);
          while (next == seen) {
            spinPause();
            next = phase.load(std::memory_order_acquire);
          }
          seen = next;
          if (stop.load(std::memory_order_acquire)) {
            return;
          }
          auto [bv, bw] = relaxBlock(id, currentTarget);
          localBest[id].vertex = bv;
          localBest[id].weight = bw;
          localBest[id].done.store(seen, std::memory_order_release);
        }
      };

      std::vector<std::future<void>> futures;
      futures.reserve(workerTasks);
      for (std::size_t id = 1; id < participantCount; ++id) {
        futures.emplace_back(pool.pool->submit_task([&, id] { workerLoop(id); }));
      }
      while (ready.load(std::memory_order_acquire) != workerTasks) {
        spinPause();
      }

      auto reduceBest = [&]() noexcept -> std::pair<std::int32_t, T> {
        std::int32_t bestV = -1;
        T bestW = std::numeric_limits<T>::max();
        for (const LocalBest &candidate : localBest) {
          if (candidate.vertex >= 0 && candidate.weight < bestW) {
            bestW = candidate.weight;
            bestV = candidate.vertex;
          }
        }
        return {bestV, bestW};
      };

      auto relaxRound = [&](std::int32_t target) noexcept -> std::pair<std::int32_t, T> {
        currentTarget = target;
        const std::uint32_t newPhase =
            phase.fetch_add(1, std::memory_order_acq_rel) + std::uint32_t{1};
        auto [bv, bw] = relaxBlock(0, target);
        localBest[0].vertex = bv;
        localBest[0].weight = bw;
        // Per-worker flag wait: one cache line per worker, written exactly once per phase by
        // its owner and read exactly once per phase by main. Replaces a shared atomic counter
        // whose cross-core RMW serialised the barrier close.
        for (std::size_t id = 1; id < participantCount; ++id) {
          while (localBest[id].done.load(std::memory_order_acquire) != newPhase) {
            spinPause();
          }
        }
        return reduceBest();
      };

      visited[0] = 1U;
      edgeWeight[0] = T{0};
      auto [nextV, nextW] = relaxRound(static_cast<std::int32_t>(0));

      while (out.edges.size() + 1 < n) {
        CLUSTERING_ALWAYS_ASSERT(nextV >= 0);

        const auto bIdx = static_cast<std::size_t>(nextV);
        visited[bIdx] = 1U;
        out.edges.push_back(MstEdge<T>{parent[bIdx], nextV, nextW});

        if (out.edges.size() + 1 == n) {
          break;
        }
        auto next = relaxRound(nextV);
        nextV = next.first;
        nextW = next.second;
      }

      stop.store(true, std::memory_order_release);
      phase.fetch_add(1, std::memory_order_release);
      for (auto &future : futures) {
        future.get();
      }
      return true;
    };

    if (persistentRelaxFrom()) {
      return;
    }

    auto relaxFrom = [&](std::int32_t target) noexcept -> std::pair<std::int32_t, T> {
      const auto tIdx = static_cast<std::size_t>(target);
      const T coreT = coreDistData[tIdx];
      const T *rowT = xData + (tIdx * d);
      // Per-iter parallel dispatch: the gate uses the per-worker op budget `(n*d / nWorkers)`
      // so very small @c n stays serial and avoids submit_blocks overhead.
      if (pool.pool != nullptr && pool.shouldParallelizeWork(n * d)) {
        pool.pool
            ->submit_blocks(std::size_t{0}, n,
                            [&](std::size_t lo, std::size_t hi) {
                              relaxRange(lo, hi, target, tIdx, coreT, rowT);
                            })
            .wait();
        return findNext();
      }
      return relaxRangeAndFindNext(0, n, target, tIdx, coreT, rowT);
    };

    // Seed: vertex 0 is in the tree with weight 0. The first relax populates @c edgeWeight for
    // every other vertex so the first argmin scan has finite values.
    visited[0] = 1U;
    edgeWeight[0] = T{0};
    auto [nextV, nextW] = relaxFrom(static_cast<std::int32_t>(0));

    while (out.edges.size() + 1 < n) {
      // The graph is complete (every pair has a finite MRD), so on a connected workload the
      // argmin always finds a finite entry. Asserting here flags any contract violation that
      // would otherwise leave the spanning tree short of @c n - 1 edges.
      CLUSTERING_ALWAYS_ASSERT(nextV >= 0);

      const auto bIdx = static_cast<std::size_t>(nextV);
      visited[bIdx] = 1U;
      out.edges.push_back(MstEdge<T>{parent[bIdx], nextV, nextW});

      if (out.edges.size() + 1 == n) {
        break;
      }
      auto next = relaxFrom(nextV);
      nextV = next.first;
      nextW = next.second;
    }
  }

private:
  [[nodiscard]] static constexpr bool shouldUseDenseCore(std::size_t n, std::size_t d,
                                                         std::size_t minSamples) noexcept {
    return n >= kPrimDenseCoreMinN && d >= kPrimDenseCoreMinD &&
           minSamples <= kPrimDenseCoreMaxMinSamples;
  }

  [[nodiscard]] static bool shouldUsePersistentParallelRelax(std::size_t n, std::size_t d,
                                                             bool useDenseCore,
                                                             math::Pool pool) noexcept {
    return useDenseCore && pool.pool != nullptr &&
           pool.workerCount() >= kPrimPersistentRelaxMinWorkers &&
           pool.shouldParallelizeWork(n * d, kPrimPersistentRelaxMinOpsPerWorker);
  }

  static void spinPause() noexcept {
#ifdef CLUSTERING_USE_AVX2
    _mm_pause();
#else
    std::this_thread::yield();
#endif
  }

  /// Maintain the per-row top-@c minSamples set of smallest squared distances seen so far,
  /// keyed by the slot holding the current worst (largest) entry. The cached @c worstSlot lets
  /// the common "new distance is not smaller than our worst" case exit in O(1); only on a swap
  /// do we rescan the @c minSamples slots to find the new worst. This beats a min-heap because
  /// @c minSamples stays small (<= @ref kPrimDenseCoreMaxMinSamples) and the branch-heavy hot
  /// path vectorises poorly inside a heap sift.
  static void updateTopK(T *topK, std::vector<std::size_t> &worstSlot, std::size_t minSamples,
                         std::size_t row, T sq) noexcept {
    T *const rowTopK = topK + (row * minSamples);
    std::size_t worst = worstSlot[row];
    if (!(sq < rowTopK[worst])) {
      return;
    }
    rowTopK[worst] = sq;
    worst = 0;
    T worstValue = rowTopK[0];
    for (std::size_t s = 1; s < minSamples; ++s) {
      if (rowTopK[s] > worstValue) {
        worstValue = rowTopK[s];
        worst = s;
      }
    }
    worstSlot[row] = worst;
  }

  /// Dense symmetric core-distance pass. Each unordered pair `(i, j)` is visited once via the
  /// upper triangle @c j = i+1..n-1; the resulting squared distance updates both row @c i and
  /// row @c j so each pair contributes to both endpoints without a second compute pass. Squared
  /// distance is derived from precomputed row norms and a single dot product via the identity
  /// `||a-b||^2` = ||a||^2 + ||b||^2 - 2<a,b>, which lets the inner kernel be a fused-multiply-
  /// add dot instead of a compensated subtract-and-square.
  static void computeDenseCoreDistances(const NDArray<T, 2> &X, const std::vector<T> &rowNorms,
                                        std::size_t minSamples, bool rowsAligned32, math::Pool pool,
                                        T *coreDistData) {
    const std::size_t n = X.dim(0);
    const std::size_t d = X.dim(1);
    const T *const xData = X.data();
    std::vector<T> topK(n * minSamples, std::numeric_limits<T>::max());
    std::vector<std::size_t> worstSlot(n, 0);

    // With enough workers, row-independent scans win despite computing each pair twice: every row
    // owns its top-k state, so the pool path has no cross-row writes and can reuse the batched
    // four-neighbour distance kernel that amortises AVX2 horizontal sums.
    if (pool.pool != nullptr && pool.workerCount() >= 4 && pool.shouldParallelizeWork(n * n * d)) {
      pool.pool
          ->submit_blocks(std::size_t{0}, n,
                          [&](std::size_t lo, std::size_t hi) {
                            computeDenseCoreDistancesRows(X, minSamples, lo, hi, topK.data(),
                                                          worstSlot);
                          })
          .wait();
      for (std::size_t i = 0; i < n; ++i) {
        coreDistData[i] = topK[(i * minSamples) + worstSlot[i]];
      }
      return;
    }

    for (std::size_t i = 0; i < n; ++i) {
      const T *const rowI = xData + (i * d);
      const T normI = rowNorms[i];
      for (std::size_t j = i + 1; j < n; ++j) {
        const T *const rowJ = xData + (j * d);
        const T dot = rowsAligned32 ? math::detail::dotRowAligned32Ptr(rowI, rowJ, d)
                                    : math::detail::dotRowPtr(rowI, rowJ, d);
        const T sq = math::detail::sqEuclideanFromDot(normI, rowNorms[j], dot);
        updateTopK(topK.data(), worstSlot, minSamples, i, sq);
        updateTopK(topK.data(), worstSlot, minSamples, j, sq);
      }
    }

    for (std::size_t i = 0; i < n; ++i) {
      coreDistData[i] = topK[(i * minSamples) + worstSlot[i]];
    }
  }

  static void computeDenseCoreDistancesRows(const NDArray<T, 2> &X, std::size_t minSamples,
                                            std::size_t lo, std::size_t hi, T *topK,
                                            std::vector<std::size_t> &worstSlot) noexcept {
    constexpr std::size_t kBlockRows = 64;
    const std::size_t n = X.dim(0);
    const std::size_t d = X.dim(1);
    const T *const xData = X.data();
    std::array<T, kBlockRows> distances{};

    for (std::size_t i = lo; i < hi; ++i) {
      const T *const rowI = xData + (i * d);
      for (std::size_t base = 0; base < n; base += kBlockRows) {
        const std::size_t count = std::min(kBlockRows, n - base);
        math::detail::sqDistancesAosBlock(rowI, xData + (base * d), count, d, distances.data());
        for (std::size_t offset = 0; offset < count; ++offset) {
          const std::size_t j = base + offset;
          if (j != i) {
            updateTopK(topK, worstSlot, minSamples, i, distances[offset]);
          }
        }
      }
    }
  }
};

} // namespace clustering::hdbscan
