#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>
#include <vector>

#include "clustering/index/range_query.h"
#include "clustering/math/pairwise.h"
#include "clustering/math/thread.h"
#include "clustering/ndarray.h"

namespace clustering {

/**
 * @brief Range-index backend that builds the full eps-neighborhood adjacency in one fused
 *        pairwise sweep.
 *
 * At high dim, where tree pruning collapses, a blocked pairwise sweep with the eps-threshold
 * fused into the microkernel epilogue is the right primitive for DBSCAN: core-point detection
 * and cluster expansion both reduce to adjacency lookups, so one N*N sweep retires all the
 * pairwise compute the algorithm needs.
 *
 * @tparam T Element type of the point cloud (@c float or @c double).
 *
 * @warning The instance borrows the input matrix. The caller must keep @p points alive for the
 *          lifetime of the @c BruteForcePairwise.
 */
template <class T> class BruteForcePairwise {
public:
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                "BruteForcePairwise<T> requires T to be float or double");

  /**
   * @brief Constructs the backend over a borrowed point matrix.
   *
   * @param points Row-major @c n x @c d point matrix. Must outlive the instance.
   */
  explicit BruteForcePairwise(const NDArray<T, 2> &points) noexcept : m_points(points) {}

  /**
   * @brief Returns the core-aware radius adjacency over the indexed point cloud.
   *
   * Emits surviving `(i, j)` pairs directly from the fused AVX2 threshold kernel; the outer
   * driver partitions X rows across @p pool so per-row pushes are race-free. Full two-sided
   * degrees come from per-worker histograms over the upper halves, so core flags never need
   * the mirrored edges materialized; only non-core rows receive their `j < i` neighbours,
   * which the border-assignment scan is the sole reader of.
   *
   * @param radius Non-negative neighbourhood radius; comparison runs on the squared distance.
   * @param minPts Core threshold on the self-inclusive neighbour count.
   * @param pool   Parallelism injection forwarded to the thresholded sweep.
   * @return Rows and core flags per the @ref clustering::index::CoreAdjacency contract.
   */
  [[nodiscard]] index::CoreAdjacency query(T radius, std::size_t minPts, math::Pool pool) const {
    const std::size_t n = m_points.dim(0);
    index::CoreAdjacency out;
    out.rows.resize(n);
    out.isCore.assign(n, 0);
    if (n == 0) {
      return out;
    }
    std::vector<std::vector<std::int32_t>> &adj = out.rows;

    const std::size_t d = m_points.dim(1);
    std::size_t adjReserveFloor = 16;
    if (d == 32) {
      adjReserveFloor = 24;
    } else if (d == 64) {
      adjReserveFloor = 20;
    }
    // Reserve a small floor per row so the first push_backs do not trigger the vector-doubling
    // reallocation cascade that otherwise dominates adjacency construction on dense fixtures.
    // Fanning the reserves out matters at high worker counts: the row allocations are a
    // serial malloc train that idles every worker before the sweep starts, and the allocator's
    // per-thread arenas let the fan-out scale.
    const auto reserveRows = [&](std::size_t lo, std::size_t hi) {
      for (std::size_t i = lo; i < hi; ++i) {
        adj[i].reserve(adjReserveFloor);
      }
    };
    const bool fanOut = pool.shouldParallelize(n, 256, 2);
    if (fanOut) {
      pool.parallelForBlocks(std::size_t{0}, n, std::size_t{0}, reserveRows);
    } else {
      reserveRows(0, n);
    }

    const T radiusSq = radius * radius;
    // Symmetric eps-neighbour graph: the kernel emits each unique upper-triangular cell once
    // with `row <= col`. The kernel-side emit only touches `adj[row]` so workers writing
    // disjoint row chunks remain race-free.
    auto emit = [&adj](std::size_t row, std::size_t col) {
      adj[row].push_back(static_cast<std::int32_t>(col));
    };
    math::pairwiseSqEuclideanThresholdedSymmetric(m_points, radiusSq, pool, emit);

    // Mirror degrees without mirror edges: workers accumulate `j < i` neighbour counts in
    // private histograms that stay cache-resident, then a row-partitioned reduction folds them.
    // Scattering the mirrored edges themselves would be a latency-bound write per edge into a
    // random destination row; the degree is all the core verdict needs.
    const std::size_t workers = pool.workerCount();
    std::vector<std::vector<std::uint32_t>> workerCounts(workers);
    const auto countRange = [&](std::size_t lo, std::size_t hi) {
      std::vector<std::uint32_t> &counts = workerCounts[math::Pool::workerIndex()];
      if (counts.empty()) {
        counts.assign(n, 0);
      }
      for (std::size_t i = lo; i < hi; ++i) {
        for (const std::int32_t neighbor : adj[i]) {
          const auto j = static_cast<std::size_t>(neighbor);
          if (j > i) {
            ++counts[j];
          }
        }
      }
    };
    std::vector<std::uint32_t> mirrorDeg(n, 0);
    const auto reduceRange = [&](std::size_t lo, std::size_t hi) {
      for (const std::vector<std::uint32_t> &counts : workerCounts) {
        if (counts.empty()) {
          continue;
        }
        for (std::size_t i = lo; i < hi; ++i) {
          mirrorDeg[i] += counts[i];
        }
      }
    };
    const auto flagRange = [&](std::size_t lo, std::size_t hi) {
      for (std::size_t i = lo; i < hi; ++i) {
        out.isCore[i] =
            (adj[i].size() + mirrorDeg[i] >= minPts) ? std::uint8_t{1} : std::uint8_t{0};
      }
    };
    if (fanOut) {
      pool.parallelForBlocks(std::size_t{0}, n, std::size_t{0}, countRange);
      pool.parallelForBlocks(std::size_t{0}, n, std::size_t{0}, reduceRange);
      pool.parallelForBlocks(std::size_t{0}, n, std::size_t{0}, flagRange);
    } else {
      countRange(0, n);
      reduceRange(0, n);
      flagRange(0, n);
    }

    // Complete the non-core rows only. Workers collect the surviving `(dest, src)` mirrors
    // into private lists; the survivor set is a vanishing fraction of the edge set, so the
    // ordered serial apply costs a handful of pushes and keeps row contents deterministic.
    std::vector<std::vector<std::pair<std::int32_t, std::int32_t>>> workerMirrors(workers);
    const auto collectRange = [&](std::size_t lo, std::size_t hi) {
      std::vector<std::pair<std::int32_t, std::int32_t>> &mirrors =
          workerMirrors[math::Pool::workerIndex()];
      for (std::size_t i = lo; i < hi; ++i) {
        for (const std::int32_t neighbor : adj[i]) {
          const auto j = static_cast<std::size_t>(neighbor);
          if (j > i && out.isCore[j] == 0) {
            mirrors.emplace_back(static_cast<std::int32_t>(j), static_cast<std::int32_t>(i));
          }
        }
      }
    };
    if (fanOut) {
      pool.parallelForBlocks(std::size_t{0}, n, std::size_t{0}, collectRange);
    } else {
      collectRange(0, n);
    }
    std::vector<std::pair<std::int32_t, std::int32_t>> mirrors;
    for (const auto &local : workerMirrors) {
      mirrors.insert(mirrors.end(), local.begin(), local.end());
    }
    std::sort(mirrors.begin(), mirrors.end());
    for (const auto &[dest, src] : mirrors) {
      adj[static_cast<std::size_t>(dest)].push_back(src);
    }
    return out;
  }

private:
  const NDArray<T, 2> &m_points; ///< Borrowed point matrix; the caller owns the storage.
};

} // namespace clustering
