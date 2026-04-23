#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

#include "clustering/always_assert.h"
#include "clustering/hdbscan/mst_output.h"
#include "clustering/index/kdtree.h"
#include "clustering/math/detail/avx2_helpers.h"
#include "clustering/math/thread.h"
#include "clustering/ndarray.h"

namespace clustering::hdbscan {

/**
 * @brief Compute budget that gates the streaming Prim backend, expressed as the maximum point
 *        count it will accept.
 *
 * The streaming Prim variant materialises only @c O(n) state (no @c n*n MRD matrix), but the
 * inner relax recomputes @c d-wide squared-Euclidean distances per popped vertex, costing
 * @c O(n^2 * d) total scalar work. At @c d <= 60 (the dispatcher's Prim window) and modern AVX2
 * throughput, @c n = 16384 lands at roughly 200-300 ms of wall-clock per fit -- a comfortable
 * upper bound. Beyond this @c n the dispatcher should prefer NN-Descent (high @c d) or Boruvka
 * (low @c d), both of which scale better than @c O(n^2).
 */
inline constexpr std::size_t kPrimMaxN = std::size_t{16384};

/// Backwards-compatible alias for the historical byte-budget name. The streaming variant no
/// longer allocates an @c (n x n) matrix, so the gate is now expressed as a max @c n directly;
/// the alias preserves any external constants that referenced the old name. The numeric value
/// equals @ref kPrimMaxN squared, scaled by @c sizeof(float), so the legacy comparison
/// @c n*n*sizeof(T) <= kPrimMrdMatrixByteBudget yields the same admissible set.
inline constexpr std::size_t kPrimMrdMatrixByteBudget = kPrimMaxN * kPrimMaxN * sizeof(float);

/**
 * @brief Exact minimum-spanning-tree backend over mutual-reachability distance, streaming Prim.
 *
 * Given a point matrix @p X and a core-distance parameter @p minSamples, the backend:
 *   1. Computes per-point core distances as the @c minSamples-th nearest-neighbour distance
 *      (self-excluded, squared) via @ref KDTree::knnQuery.
 *   2. Runs Prim's algorithm rooted at vertex @c 0. Per popped vertex, the relax step
 *      recomputes the @c d-wide squared Euclidean distance to every unvisited point on the fly
 *      (no @c n*n matrix is ever materialised) and lifts to MRD via the per-pair @c max with
 *      both core distances.
 *   3. Picks the next vertex via a serial argmin scan over the @c edgeWeight array. At dense
 *      Prim's @c O(n^2) overall complexity, the linear-scan argmin is asymptotically free
 *      relative to the @c d-wide distance recompute and avoids the heap allocations that
 *      dominate the previous heap-based variant.
 *
 * @par Memory
 * Working state is @c O(n) -- no quadratic allocation. At @c n = 10000 d = 32 the previous
 * heap+matrix variant peaked at ~400 MiB; this variant peaks at ~120 KiB plus the @c X view.
 *
 * @par Cache
 * Each Prim iteration streams @c X sequentially in @c d-wide rows. Total bytes touched per
 * iteration is @c n * d * sizeof(T) which fits in @c L2 for the dispatcher's @c (n, d) window;
 * the previous variant's @c n*n matrix exceeded @c L3 by an order of magnitude and was
 * memory-bandwidth bound on every iteration.
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
   * @pre @p minSamples is positive and strictly less than @c X.dim(0).
   * @pre @c X.dim(0) does not exceed @ref kPrimMaxN. A violation fires
   *      @c CLUSTERING_ALWAYS_ASSERT before any allocation.
   *
   * @param X          Contiguous @c (n x d) dataset; caller retains ownership.
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

    // Compute-budget guard: refuse @c n that would push the @c O(n^2 * d) inner work past the
    // dispatcher's intended Prim window. Dies before any allocation so out-of-budget calls
    // surface deterministically. The legacy assertion message ("n <= kNsqBudget / n") is
    // preserved so the existing budget-death test continues to match.
    constexpr std::size_t kNsqBudget = kPrimMrdMatrixByteBudget / sizeof(T);
    CLUSTERING_ALWAYS_ASSERT(n <= kNsqBudget / n);

    out.edges.clear();
    out.edges.reserve(n - 1);
    out.coreDistances = NDArray<T, 1>(std::array<std::size_t, 1>{n});

    // Phase 1: core distances from the kNN query. The tree is built on @p X directly; the query
    // returns neighbours sorted ascending by squared distance. Core distance is the squared
    // distance to the @c minSamples-th nearest neighbour, consistent with the on-the-fly
    // distance compute in the Prim loop below (squared distances on both sides).
    const KDTree<T> tree(X);
    const auto kSigned = static_cast<std::int32_t>(minSamples);
    auto [knnIdx, knnSqDist] = tree.knnQuery(kSigned, pool);
    (void)knnIdx;
    T *coreDistData = out.coreDistances.data();
    for (std::size_t i = 0; i < n; ++i) {
      coreDistData[i] = knnSqDist(i, minSamples - 1);
    }

    // Phase 2: streaming Prim. Maintain @c edgeWeight[v] = best-known incident MRD weight to
    // the growing tree, @c parent[v] = the in-tree vertex realising that weight, and a visited
    // bitmap. Each iteration picks the smallest-weight unvisited @c target via a linear scan,
    // emits the edge @c (parent[target], target, edgeWeight[target]), then relaxes every other
    // unvisited @c v by recomputing @c sqDist(target, v) and lifting to MRD.
    std::vector<std::uint8_t> visited(n, std::uint8_t{0});
    std::vector<std::int32_t> parent(n, std::int32_t{0});
    std::vector<T> edgeWeight(n, std::numeric_limits<T>::max());
    const T *xData = X.data();

    auto relaxFrom = [&](std::int32_t target) noexcept {
      const auto tIdx = static_cast<std::size_t>(target);
      const T coreT = coreDistData[tIdx];
      const T *rowT = xData + (tIdx * d);
      for (std::size_t v = 0; v < n; ++v) {
        if (visited[v] != 0U) {
          continue;
        }
        const T sq = math::detail::sqEuclideanRowPtr(rowT, xData + (v * d), d);
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

    // Seed: vertex 0 is in the tree with weight 0. The first relax populates @c edgeWeight for
    // every other vertex so the first argmin scan has finite values.
    visited[0] = 1U;
    edgeWeight[0] = T{0};
    relaxFrom(static_cast<std::int32_t>(0));

    while (out.edges.size() + 1 < n) {
      // Argmin over unvisited. Linear scan is asymptotically free relative to the @c O(n*d)
      // relax that follows; avoids the heap allocations and stale-entry filtering of the
      // previous variant.
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
      // The graph is complete (every pair has a finite MRD), so on a connected workload the
      // argmin always finds a finite entry. Asserting here flags any contract violation that
      // would otherwise leave the spanning tree short of @c n - 1 edges.
      CLUSTERING_ALWAYS_ASSERT(bestV >= 0);

      const auto bIdx = static_cast<std::size_t>(bestV);
      visited[bIdx] = 1U;
      out.edges.push_back(MstEdge<T>{parent[bIdx], bestV, bestW});

      if (out.edges.size() + 1 == n) {
        break;
      }
      relaxFrom(bestV);
    }
  }
};

} // namespace clustering::hdbscan
