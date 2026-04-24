#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "clustering/always_assert.h"
#include "clustering/index/nn_descent/detail/neighbor_heap.h"
#include "clustering/math/detail/avx2_helpers.h"
#include "clustering/math/rng.h"
#include "clustering/ndarray.h"

namespace clustering::index::nn_descent::detail {

/**
 * @brief Random-projection tree initialization of an NN-Descent neighbor bank.
 *
 * Recursively partitions the input point set along random hyperplanes (each generated as two
 * randomly-picked points; the hyperplane is the perpendicular bisector of that segment). When a
 * node holds fewer than @c leafLimit points, its point set becomes a leaf and every pair of
 * leaf points is compared under squared Euclidean distance; each distance is pushed into the
 * corresponding per-node neighbor heap. Points at random hyperplanes tend to cluster neighbors
 * together, so the leaf-pair seed initialises the bank close to the true kNN graph before the
 * NN-Descent join loop refines it.
 *
 * @par Why one tree is enough
 * NN-Descent's contribution is the local-join refinement. A single RP-tree's leaves give each
 * node a handful of coarse neighbors -- enough for the join step to bootstrap without falling
 * into a uniform-random cold start. Multiple trees would further reduce iteration count but
 * increase build cost linearly; one tree is the default per Dong 2011 and sufficient for the
 * recall thresholds this implementation is expected to meet.
 *
 * @par Tie-break
 * Distance ties resolve via the @ref NeighborHeapBank admission rule (smaller index wins);
 * hyperplane ties resolve on the deterministic PRNG sequence, so identical @p seed reproduces
 * identical initial state.
 *
 * @tparam T Element type of the point cloud (must be @c float or @c double).
 */
template <class T> class RpTreeInit {
public:
  /**
   * @brief Build the RP-tree and seed @p bank in place, then top up with uniform-random
   *        candidates.
   *
   * Single-tree RP initialization alone tends to saturate each leaf (every pair within a leaf is
   * already in each participant's heap), which starves the subsequent NN-Descent join step of
   * exploration candidates. Topping the heap up with uniform-random picks per node gives the
   * join step "old" neighbors outside the leaf to cross-join against, which unblocks global
   * exploration. The combination of coarse RP-tree locality plus random diversity matches the
   * initialization pynndescent ships by default.
   *
   * @param X          @c n x d contiguous point cloud; not owned.
   * @param leafLimit  Split threshold: nodes with fewer than this many points become leaves.
   *                   Must be at least 2 so a leaf has at least one pair.
   * @param seed       PRNG seed (any 64-bit value).
   * @param bank       Destination neighbor bank. Existing contents are preserved (leaf-pair
   *                   pushes are subject to the bank's admission gate).
   */
  static void build(const NDArray<T, 2> &X, std::size_t leafLimit, std::uint64_t seed,
                    NeighborHeapBank<T> &bank) {
    const std::size_t n = X.dim(0);
    if (n == 0) {
      return;
    }
    const std::size_t d = X.dim(1);
    CLUSTERING_ALWAYS_ASSERT(leafLimit >= 2);

    std::vector<std::int32_t> indices(n);
    for (std::size_t i = 0; i < n; ++i) {
      indices[i] = static_cast<std::int32_t>(i);
    }

    math::pcg64 rng;
    rng.seed(seed, /*stream=*/0x1E);

    const T *data = X.data();
    std::vector<T> projections; // Scratch buffer for hyperplane projections; grown on demand.
    projections.reserve(n);

    splitRange(data, d, indices, 0, n, leafLimit, rng, projections, bank);

    // Top-up pass: each node gets @c k uniform-random candidates so the join step has non-leaf
    // "old" neighbors to cross-join against. The heap's admission gate filters duplicates.
    const std::size_t k = bank.k();
    if (k == 0 || n < 2) {
      return;
    }
    math::pcg64 topupRng;
    topupRng.seed(seed, /*stream=*/0x17);
    const auto nSpan = static_cast<std::uint32_t>(n);
    for (std::size_t i = 0; i < n; ++i) {
      const T *pi = data + (i * d);
      const auto ii = static_cast<std::int32_t>(i);
      for (std::size_t s = 0; s < k; ++s) {
        std::uint32_t j = math::randUniformU32(topupRng) % nSpan;
        if (j == i) {
          j = (j + 1) % nSpan;
        }
        const T *pj = data + (static_cast<std::size_t>(j) * d);
        const T sqDist = math::detail::sqEuclideanRowPtr(pi, pj, d);
        bank.push(ii, static_cast<std::int32_t>(j), sqDist);
      }
    }
  }

private:
  /**
   * @brief Recursive splitter. `[lo, hi)` names the range of @p indices currently partitioned
   *        at this node.
   */
  static void splitRange(const T *data, std::size_t d, std::vector<std::int32_t> &indices,
                         std::size_t lo, std::size_t hi, std::size_t leafLimit, math::pcg64 &rng,
                         std::vector<T> &projections, NeighborHeapBank<T> &bank) {
    const std::size_t count = hi - lo;
    if (count <= leafLimit) {
      seedLeaf(data, d, indices, lo, hi, bank);
      return;
    }

    // Pick two random pivots from [lo, hi). The hyperplane is the perpendicular bisector of the
    // segment; projecting a point onto the normal sign-classifies it as "left" or "right."
    const auto span = static_cast<std::uint32_t>(count);
    const std::uint32_t aRel = math::randUniformU32(rng) % span;
    std::uint32_t bRel = math::randUniformU32(rng) % span;
    // Ensure two distinct pivots; retry on collision.
    for (std::size_t guard = 0; aRel == bRel && guard < 4; ++guard) {
      bRel = math::randUniformU32(rng) % span;
    }
    if (aRel == bRel) {
      // Degenerate: every retry matched. Fall back to the halfway cursor as the second pivot.
      bRel = (aRel + 1) % span;
    }
    const std::size_t aAbs = lo + aRel;
    const std::size_t bAbs = lo + bRel;
    const T *pa = data + (static_cast<std::size_t>(indices[aAbs]) * d);
    const T *pb = data + (static_cast<std::size_t>(indices[bAbs]) * d);

    // Compute projections of each point onto (pb - pa). Sign of projection - midpoint is the
    // side. @c midProj depends only on the pivots, so compute it once before the per-point loop.
    T midProj = T{0};
    for (std::size_t j = 0; j < d; ++j) {
      midProj += T{0.5} * (pa[j] + pb[j]) * (pb[j] - pa[j]);
    }
    projections.assign(count, T{0});
    for (std::size_t t = 0; t < count; ++t) {
      const T *p = data + (static_cast<std::size_t>(indices[lo + t]) * d);
      T proj = T{0};
      for (std::size_t j = 0; j < d; ++j) {
        proj += p[j] * (pb[j] - pa[j]);
      }
      projections[t] = proj - midProj;
    }

    // Partition indices in place: negative projection -> left, non-negative -> right. On ties
    // (projection == 0) bias toward the smaller side so degenerate uniform data still progresses.
    std::size_t leftCursor = lo;
    std::vector<std::int32_t> tmp;
    tmp.reserve(count);
    std::vector<std::int32_t> right;
    right.reserve(count);
    for (std::size_t t = 0; t < count; ++t) {
      const std::int32_t idx = indices[lo + t];
      if (projections[t] < T{0}) {
        tmp.push_back(idx);
      } else {
        right.push_back(idx);
      }
    }
    // Degenerate split: if either side is empty, bisect the range deterministically so
    // recursion still makes progress.
    if (tmp.empty() || right.empty()) {
      tmp.clear();
      right.clear();
      const std::size_t mid = count / 2;
      for (std::size_t t = 0; t < count; ++t) {
        if (t < mid) {
          tmp.push_back(indices[lo + t]);
        } else {
          right.push_back(indices[lo + t]);
        }
      }
    }
    for (const std::int32_t idx : tmp) {
      indices[leftCursor++] = idx;
    }
    for (const std::int32_t idx : right) {
      indices[leftCursor++] = idx;
    }
    const std::size_t midAbs = lo + tmp.size();

    splitRange(data, d, indices, lo, midAbs, leafLimit, rng, projections, bank);
    splitRange(data, d, indices, midAbs, hi, leafLimit, rng, projections, bank);
  }

  /**
   * @brief Admit every pair of points in `[lo, hi) into the bank (both directions)`.
   */
  static void seedLeaf(const T *data, std::size_t d, const std::vector<std::int32_t> &indices,
                       std::size_t lo, std::size_t hi, NeighborHeapBank<T> &bank) {
    for (std::size_t s = lo; s < hi; ++s) {
      const std::int32_t i = indices[s];
      const T *pi = data + (static_cast<std::size_t>(i) * d);
      for (std::size_t t = s + 1; t < hi; ++t) {
        const std::int32_t j = indices[t];
        const T *pj = data + (static_cast<std::size_t>(j) * d);
        const T sqDist = math::detail::sqEuclideanRowPtr(pi, pj, d);
        bank.push(i, j, sqDist);
        bank.push(j, i, sqDist);
      }
    }
  }
};

} // namespace clustering::index::nn_descent::detail
