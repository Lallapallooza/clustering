#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>
#include <utility>

#include "clustering/always_assert.h"
#include "clustering/kmeans/detail/convergence.h"
#include "clustering/kmeans/detail/empty_cluster.h"
#include "clustering/math/centroid_shift.h"
#include "clustering/math/defaults.h"
#include "clustering/math/detail/gemm_outer_prepacked.h"
#include "clustering/math/detail/gemm_pack.h"
#include "clustering/math/detail/matrix_desc.h"
#include "clustering/math/detail/pairwise_argmin_outer.h"
#include "clustering/math/pairwise.h"
#include "clustering/math/pairwise_argmin.h"
#include "clustering/math/reduce.h"
#include "clustering/math/thread.h"
#include "clustering/ndarray.h"

namespace clustering::kmeans {

namespace detail {

/**
 * @brief Work-unit partition for the label-grouped fold over @c n points into @c numBlocks bins.
 *
 * The partition maps @c [first_index, first_index + n) onto at most @c desired blocks with
 * near-equal size. Ascending-block-order reduction is the deterministic fold the label-
 * accumulation step relies on, which pins bit-identity across nJobs settings at @c n_jobs = 1.
 */
struct BlockPartition {
  std::size_t first_index = 0;
  std::size_t block_size = 0;
  std::size_t remainder = 0;
  std::size_t num_blocks = 0;

  BlockPartition(std::size_t first, std::size_t n, std::size_t desired) noexcept
      : first_index(first) {
    if (n == 0 || desired == 0) {
      num_blocks = 0;
      return;
    }
    num_blocks = std::min(desired, n);
    block_size = n / num_blocks;
    remainder = n % num_blocks;
    if (block_size == 0) {
      block_size = 1;
      num_blocks = n;
    }
  }

  [[nodiscard]] std::size_t blockIndexOf(std::size_t lo) const noexcept {
    const std::size_t rel = lo - first_index;
    const std::size_t big = remainder * (block_size + 1);
    if (rel < big) {
      return rel / (block_size + 1);
    }
    return remainder + ((rel - big) / block_size);
  }
};

/**
 * @brief Maximum @c d for the direct-compute argmin hot path.
 *
 * At @c d <= this threshold the fused argmin-GEMM driver's @c packA + packB overhead dominates
 * the handful of FMAs the microkernel performs, so the direct @c ||x - c||^2 formula with 8-row
 * SIMD accumulators beats the packed-GEMM path. Measured on Zen5: crossover sits near
 * @c d == 8 where the two paths tie; below that the direct path wins by the pack cost.
 */
inline constexpr std::size_t kDirectArgminMaxD = 8;

} // namespace detail

/**
 * @brief Fused-argmin-GEMM Lloyd driver.
 *
 * Runs the Lloyd iteration over caller-seeded centroids: assignment via the fused AVX2
 * argmin-GEMM hot path at @c d <= @c math::defaults::pairwiseArgminMaxD and the chunked
 * materialized fallback above it, label-grouped fold into per-cluster sums (Kahan-compensated
 * at @c n >= @ref kahanNThreshold), empty-cluster reseed against the current per-point
 * distance scratch, mean step, and convergence test on the Kahan-summed total squared shift.
 * All scratch buffers live inside the policy instance; no allocation fires between the first
 * assignment call and the convergence check of any iteration once the shape has been warmed.
 *
 * @tparam T Element type; @c float or @c double.
 */
template <class T> class LloydFusedGemm {
public:
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                "LloydFusedGemm<T> requires T to be float or double");

  LloydFusedGemm()
      : m_centroidsOld({0, 0}), m_cSqNorms({0}), m_sums({0, 0}), m_counts({0}), m_minDistSq({0}),
        m_shiftSq({0}), m_partialSums({0}), m_partialComps({0}), m_partialCounts({0}),
        m_foldComp({0}), m_packedB({0}), m_packedCSqNorms({0}), m_distsChunk({0, 0}),
        m_gemmApArena({0}), m_xChunkNormsSq({0}) {}

#ifdef CLUSTERING_KMEANS_KAHAN_N_THRESHOLD
  /**
   * @brief @c n threshold at which the centroid accumulator switches to the Kahan-compensated
   *        variant. Below this, the plain partial-sum + fold variant is used.
   *
   * Compensation is load-bearing for the 1%-inertia gate at the @c (n=1e6, k=1000) corner where
   * per-cluster running totals are dominated by a large sum plus many small addends. Override
   * with @c -DCLUSTERING_KMEANS_KAHAN_N_THRESHOLD=<value>.
   */
  static constexpr std::size_t kahanNThreshold = CLUSTERING_KMEANS_KAHAN_N_THRESHOLD;
#else
  static constexpr std::size_t kahanNThreshold = 100000;
#endif

  /**
   * @brief Run the Lloyd loop against caller-seeded centroids.
   *
   * @param X         Contiguous n x d dataset. Borrowed; caller retains ownership.
   * @param centroids k x d centroid matrix. Caller has already populated the rows with the
   *                  seeder's output; this routine overwrites them in place with the
   *                  iteration-final values.
   * @param k         Number of clusters.
   * @param maxIter   Iteration cap.
   * @param tol       Convergence tolerance on the L2 centroid shift (linear; compared
   *                  internally as @c tol*tol against the Kahan-summed per-centroid shift-squared).
   * @param pool      Parallelism injection.
   * @param outLabels Length-n assignment; each entry in @c [0, k).
   * @param outInertia Kahan-summed @c f64 total of per-point squared distance to assignment.
   * @param outNIter  Iterations executed before @p tol or @p maxIter fired.
   * @param outConverged @c true iff iteration stopped because centroid shift fell at or below @p
   * tol.
   */
  void run(const NDArray<T, 2, Layout::Contig> &X, NDArray<T, 2, Layout::Contig> &centroids,
           std::size_t k, std::size_t maxIter, T tol, math::Pool pool,
           NDArray<std::int32_t, 1> &outLabels, double &outInertia, std::size_t &outNIter,
           bool &outConverged) {
    const std::size_t n = X.dim(0);
    const std::size_t d = X.dim(1);

    CLUSTERING_ALWAYS_ASSERT(k >= 1);
    CLUSTERING_ALWAYS_ASSERT(n >= k);
    CLUSTERING_ALWAYS_ASSERT(centroids.dim(0) == k);
    CLUSTERING_ALWAYS_ASSERT(centroids.dim(1) == d);
    CLUSTERING_ALWAYS_ASSERT(outLabels.dim(0) == n);

    if (n == 0 || d == 0) {
      outNIter = 0;
      outConverged = true;
      outInertia = 0.0;
      return;
    }

    const std::size_t workerCount = pool.workerCount();
    ensureShape(n, d, k, workerCount);

    const T tolSq = tol * tol;
    const bool useKahan = n >= kahanNThreshold;

    refreshCentroidSqNorms(centroids);

    std::size_t iter = 0;
    bool converged = false;

    while (iter < maxIter) {
      runAssignment(X, centroids, outLabels, pool);

      std::memcpy(m_centroidsOld.data(), centroids.data(),
                  centroids.dim(0) * centroids.dim(1) * sizeof(T));

      if (useKahan) {
        scatterAndFoldKahan(X, outLabels, k, pool);
      } else {
        scatterAndFoldPlain(X, outLabels, k, pool);
      }

      // Empty-cluster reseed: furthest-point pass bounded by the counts scan. m_minDistSq still
      // holds the decomposed-formula residual from the assignment above; the noise tail is
      // bounded by per-point @c ||c||^2 + ||x||^2 cancellation, smaller than the inter-blob
      // distance the donor is selected against, so the argmax selection is preserved in
      // practice on benchmark data. The donor's minDistSq is zeroed so successive empties
      // cannot reseed to the same point.
      (void)::clustering::kmeans::detail::reseedEmptyClusters<T>(X, centroids, m_sums, m_counts,
                                                                 m_minDistSq);

      finalizeMeans(centroids);
      refreshCentroidSqNorms(centroids);

      math::centroidShift<T>(m_centroidsOld, centroids, m_shiftSq, pool);
      const T totalShift = ::clustering::kmeans::detail::totalShiftSqKahan<T>(m_shiftSq);

      ++iter;
      if (totalShift <= tolSq) {
        converged = true;
        break;
      }
    }

    // Re-assign labels against the final centroids. The direct small-@c d path already writes
    // true squared distances into m_minDistSq; the decomposed @c ||c||^2 - 2 x.c + ||x||^2
    // formula carries cancellation noise at large coordinate magnitudes, which inertia and
    // the argmax reseed both need faithful, so the recompute-direct pass runs on the other
    // two tiers.
    runAssignment(X, centroids, outLabels, pool);
    if (!assignmentProducesDirectMinDistSq(X, centroids)) {
      recomputeMinDistSqDirect(X, centroids, outLabels, pool);
    }

    // Inertia: Kahan-summed in f64 to pin the 1% gate at large (n, k) envelopes where the
    // naive single-pass f32 add would drift.
    double sum = 0.0;
    double comp = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      const auto addend = static_cast<double>(m_minDistSq(i));
      const double y = addend - comp;
      const double t = sum + y;
      comp = (t - sum) - y;
      sum = t;
    }

    outInertia = sum;
    outNIter = iter;
    outConverged = converged;
  }

private:
  void ensureShape(std::size_t n, std::size_t d, std::size_t k, std::size_t workerCount) {
    const bool shapeChanged = (n != m_n) || (d != m_d) || (k != m_k);
    const bool workerChanged = (workerCount != m_workerCount);
    if (!shapeChanged && !workerChanged) {
      return;
    }

    if (shapeChanged) {
      m_centroidsOld = NDArray<T, 2, Layout::Contig>({k, d});
      m_cSqNorms = NDArray<T, 1>({k});
      m_sums = NDArray<T, 2, Layout::Contig>({k, d});
      m_counts = NDArray<std::int32_t, 1>({k});
      m_minDistSq = NDArray<T, 1>({n});
      m_shiftSq = NDArray<T, 1>({k});
      m_foldComp = NDArray<T, 1>({k * d});
      const std::size_t packedBSize = math::detail::packedBScratchSizeFloats(k, d);
      const std::size_t packedNormsSize = math::detail::packedCSqNormsScratchSizeFloats(k);
      m_packedB = NDArray<T, 1>({packedBSize == 0 ? std::size_t{1} : packedBSize});
      m_packedCSqNorms = NDArray<T, 1>({packedNormsSize == 0 ? std::size_t{1} : packedNormsSize});
      // Sized only when the materialized fallback might fire (d > fused cutoff); smaller shapes
      // collapse to a 1x1 placeholder so allocations stay proportional to the workload.
      const bool needsChunk = d > math::defaults::pairwiseArgminMaxD;
      const auto chunkShape = needsChunk ? math::chunkedMaterializedScratchShape(n, k)
                                         : std::array<std::size_t, 2>{1, 1};
      m_distsChunk = NDArray<T, 2, Layout::Contig>({chunkShape[0], chunkShape[1]});
      const std::size_t chunkRows =
          (n < math::pairwiseArgminChunkRows) ? n : math::pairwiseArgminChunkRows;
      const std::size_t chunkRowsClamped = chunkRows == 0 ? std::size_t{1} : chunkRows;
      const std::size_t xNormsLen = needsChunk ? chunkRowsClamped : std::size_t{1};
      m_xChunkNormsSq = NDArray<T, 1>({xNormsLen});
    }

    // Per-block scratch sizing. Block count caps at workerCount (see BlockPartition); we size
    // to the upper bound so both serial and parallel dispatch fit without reallocation inside
    // the loop.
    const std::size_t blocks = workerCount == 0 ? std::size_t{1} : workerCount;
    m_partialSums = NDArray<T, 1>({blocks * k * d});
    m_partialComps = NDArray<T, 1>({blocks * k * d});
    m_partialCounts = NDArray<std::int32_t, 1>({blocks * k});

    // Gemm A-pack arena sized to @c blocks * kMc * kKc so @c gemmRunPrepacked's per-worker
    // slice indexing stays in-bounds on every fan-out path.
    const bool needsChunk = d > math::defaults::pairwiseArgminMaxD;
    const std::size_t apSize = blocks * math::detail::kMc<T> * math::detail::kKc<T>;
    m_gemmApArena = NDArray<T, 1>({needsChunk ? apSize : std::size_t{1}});

    m_n = n;
    m_d = d;
    m_k = k;
    m_workerCount = workerCount;
  }

  void refreshCentroidSqNorms(const NDArray<T, 2, Layout::Contig> &centroids) noexcept {
    const std::size_t k = centroids.dim(0);
    const std::size_t d = centroids.dim(1);
    for (std::size_t c = 0; c < k; ++c) {
      const T *row = centroids.data() + (c * d);
      T s = T{0};
      for (std::size_t t = 0; t < d; ++t) {
        s += row[t] * row[t];
      }
      m_cSqNorms(c) = s;
    }
  }

  void finalizeMeans(NDArray<T, 2, Layout::Contig> &centroids) noexcept {
    const std::size_t k = centroids.dim(0);
    const std::size_t d = centroids.dim(1);
    for (std::size_t c = 0; c < k; ++c) {
      const std::int32_t cnt = m_counts(c);
      if (cnt <= 0) {
        continue;
      }
      const T inv = T{1} / static_cast<T>(cnt);
      const T *src = m_sums.data() + (c * d);
      T *dst = centroids.data() + (c * d);
      for (std::size_t t = 0; t < d; ++t) {
        dst[t] = src[t] * inv;
      }
    }
  }

  /**
   * @brief Three-tier assignment dispatch on @c d.
   *
   *   - @c d <= @ref detail::kDirectArgminMaxD : direct small-@c d argmin (no GEMM packing).
   *   - @c d <= @c pairwiseArgminMaxD           : fused argmin-GEMM with pre-packed scratch.
   *   - otherwise                               : chunked materialized fallback.
   */
  void runAssignment(const NDArray<T, 2, Layout::Contig> &X,
                     const NDArray<T, 2, Layout::Contig> &centroids,
                     NDArray<std::int32_t, 1> &labels, math::Pool pool) {
#ifdef CLUSTERING_USE_AVX2
    if constexpr (std::is_same_v<T, float>) {
      const std::size_t d = X.dim(1);
      if (X.template isAligned<32>() && centroids.template isAligned<32>() && d != 0) {
        if (d <= detail::kDirectArgminMaxD) {
          math::detail::pairwiseArgminDirectSmallDF32(X, centroids, labels, m_minDistSq, pool);
          return;
        }
        if (d <= math::defaults::pairwiseArgminMaxD) {
          math::detail::pairwiseArgminOuterAvx2F32WithScratch(X, centroids, m_cSqNorms, labels,
                                                              m_minDistSq, m_packedB.data(),
                                                              m_packedCSqNorms.data(), pool);
          return;
        }
      }
    }
#endif
    runChunkedMaterializedAssignment(X, centroids, labels, pool);
  }

  /**
   * @brief Whether the direct small-@c d assignment fires for this shape. The direct path is
   *        the only tier whose @c m_minDistSq output is a true squared distance; the decomposed
   *        formula in the other tiers needs a post-pass correction before downstream reads.
   */
  [[nodiscard]] bool
  assignmentProducesDirectMinDistSq(const NDArray<T, 2, Layout::Contig> &X,
                                    const NDArray<T, 2, Layout::Contig> &C) noexcept {
#ifdef CLUSTERING_USE_AVX2
    if constexpr (std::is_same_v<T, float>) {
      const std::size_t d = X.dim(1);
      return X.template isAligned<32>() && C.template isAligned<32>() && d != 0 &&
             d <= detail::kDirectArgminMaxD;
    } else {
      (void)X;
      (void)C;
      return false;
    }
#else
    (void)X;
    (void)C;
    return false;
#endif
  }

  void runChunkedMaterializedAssignment(const NDArray<T, 2, Layout::Contig> &X,
                                        const NDArray<T, 2, Layout::Contig> &centroids,
                                        NDArray<std::int32_t, 1> &labels,
                                        math::Pool pool) noexcept {
    const std::size_t n = X.dim(0);
    const std::size_t k = centroids.dim(0);
    const std::size_t d = X.dim(1);
    if (n == 0 || k == 0) {
      return;
    }

    // Single-call @c packB emits one panel per row-strip; @c gemmRunPrepacked reads that panel
    // assuming the @c (jc, pc) tile walks match the packer's layout. That match only holds when
    // @c d <= kKc<T> and @c k <= kNc<T>. Outside the envelope the reader silently picks the
    // wrong centroid columns; guard loudly so a future envelope widening surfaces as an assert
    // rather than a numerical mystery.
    CLUSTERING_ALWAYS_ASSERT(d <= math::detail::kKc<T>);
    CLUSTERING_ALWAYS_ASSERT(k <= math::detail::kNc<T>);

    const auto cTransposed = centroids.t();
    const auto cDesc = ::clustering::detail::describeMatrix(cTransposed);
    math::detail::packB<T>(cDesc, 0, d, 0, k, m_packedB.data());

    const std::size_t chunkCap = math::pairwiseArgminChunkRows;
    T *apArena = m_gemmApArena.data();
    T *xChunkNorms = m_xChunkNormsSq.data();

    for (std::size_t iBase = 0; iBase < n; iBase += chunkCap) {
      const std::size_t chunkRows = (iBase + chunkCap <= n) ? chunkCap : (n - iBase);

      auto xChunk = NDArray<T, 2, Layout::Contig>::borrow(X.data() + (iBase * d), {chunkRows, d});

      for (std::size_t r = 0; r < chunkRows; ++r) {
        xChunkNorms[r] = math::detail::sqNormRow<T, Layout::Contig>(xChunk, r);
      }

      auto distsView = NDArray<T, 2>::borrow(m_distsChunk.data(), {chunkRows, k});
      const auto xDesc = ::clustering::detail::describeMatrix(xChunk);
      auto distsDesc = ::clustering::detail::describeMatrixMut(distsView);
      math::detail::gemmRunPrepacked<T>(xDesc, m_packedB.data(), d, k, distsDesc, T{-2}, T{0},
                                        apArena, pool);

      auto scanRange = [&](std::size_t lo, std::size_t hi) noexcept {
        for (std::size_t i = lo; i < hi; ++i) {
          const T xn = xChunkNorms[i];
          const T *row = m_distsChunk.data() + (i * k);
          T bestVal = std::numeric_limits<T>::infinity();
          std::int32_t bestIdx = 0;
          for (std::size_t j = 0; j < k; ++j) {
            // Reconstruct ||x||^2 + ||c||^2 - 2 x . c, clamping cancellation artefacts to zero
            // the same way @c pairwiseSqEuclideanGemm's broadcast pass does.
            T v = row[j] + xn + m_cSqNorms(j);
            if (v < T{0}) {
              v = T{0};
            }
            if (j == 0 || v < bestVal) {
              bestVal = v;
              bestIdx = static_cast<std::int32_t>(j);
            }
          }
          m_minDistSq(iBase + i) = bestVal;
          labels(iBase + i) = bestIdx;
        }
      };

      if (pool.shouldParallelize(chunkRows, 4, 2) && pool.pool != nullptr) {
        pool.pool
            ->submit_blocks(std::size_t{0}, chunkRows,
                            [&](std::size_t lo, std::size_t hi) { scanRange(lo, hi); })
            .wait();
      } else {
        scanRange(0, chunkRows);
      }
    }
  }

  void scatterAndFoldPlain(const NDArray<T, 2, Layout::Contig> &X,
                           const NDArray<std::int32_t, 1> &labels, std::size_t k, math::Pool pool) {
    const std::size_t n = X.dim(0);
    const std::size_t d = X.dim(1);

    T *partialSums = m_partialSums.data();
    std::int32_t *partialCounts = m_partialCounts.data();

    for (std::size_t c = 0; c < k; ++c) {
      m_counts(c) = 0;
      for (std::size_t t = 0; t < d; ++t) {
        m_sums(c, t) = T{0};
      }
    }
    if (n == 0 || d == 0) {
      return;
    }

    const bool willParallelize = pool.shouldParallelize(n, 64, 2) && pool.pool != nullptr;
    const std::size_t desiredBlocks = willParallelize ? pool.workerCount() : std::size_t{1};
    const detail::BlockPartition part(0, n, desiredBlocks);
    const std::size_t numBlocks = part.num_blocks == 0 ? std::size_t{1} : part.num_blocks;

    for (std::size_t b = 0; b < numBlocks; ++b) {
      T *slab = partialSums + (b * k * d);
      std::int32_t *cslab = partialCounts + (b * k);
      for (std::size_t e = 0; e < k * d; ++e) {
        slab[e] = T{0};
      }
      for (std::size_t c = 0; c < k; ++c) {
        cslab[c] = 0;
      }
    }

    auto scatterRange = [&](std::size_t lo, std::size_t hi) noexcept {
      const std::size_t b = part.blockIndexOf(lo);
      T *slab = partialSums + (b * k * d);
      std::int32_t *cslab = partialCounts + (b * k);
      for (std::size_t i = lo; i < hi; ++i) {
        const std::int32_t lbl = labels(i);
        if (lbl < 0 || std::cmp_greater_equal(lbl, k)) {
          continue;
        }
        const auto row = static_cast<std::size_t>(lbl);
        const T *xRow = X.data() + (i * d);
        T *dst = slab + (row * d);
        for (std::size_t t = 0; t < d; ++t) {
          dst[t] += xRow[t];
        }
        cslab[row] += 1;
      }
    };

    if (willParallelize) {
      pool.pool
          ->submit_blocks(
              std::size_t{0}, n, [&](std::size_t lo, std::size_t hi) { scatterRange(lo, hi); },
              numBlocks)
          .wait();
    } else {
      scatterRange(0, n);
    }

    // Ascending-block-index fold. Deterministic at fixed (n, k, d, nJobs); changing this order
    // changes the last-bit of the per-cluster sum and breaks bit-identity.
    for (std::size_t b = 0; b < numBlocks; ++b) {
      const T *slab = partialSums + (b * k * d);
      const std::int32_t *cslab = partialCounts + (b * k);
      for (std::size_t c = 0; c < k; ++c) {
        m_counts(c) += cslab[c];
        const T *src = slab + (c * d);
        T *dstRow = &m_sums(c, 0);
        for (std::size_t t = 0; t < d; ++t) {
          dstRow[t] += src[t];
        }
      }
    }
  }

  void scatterAndFoldKahan(const NDArray<T, 2, Layout::Contig> &X,
                           const NDArray<std::int32_t, 1> &labels, std::size_t k, math::Pool pool) {
    const std::size_t n = X.dim(0);
    const std::size_t d = X.dim(1);

    T *partialSums = m_partialSums.data();
    T *partialComps = m_partialComps.data();
    std::int32_t *partialCounts = m_partialCounts.data();
    T *foldComp = m_foldComp.data();

    for (std::size_t c = 0; c < k; ++c) {
      m_counts(c) = 0;
      for (std::size_t t = 0; t < d; ++t) {
        m_sums(c, t) = T{0};
      }
    }
    for (std::size_t e = 0; e < k * d; ++e) {
      foldComp[e] = T{0};
    }
    if (n == 0 || d == 0) {
      return;
    }

    const bool willParallelize = pool.shouldParallelize(n, 64, 2) && pool.pool != nullptr;
    const std::size_t desiredBlocks = willParallelize ? pool.workerCount() : std::size_t{1};
    const detail::BlockPartition part(0, n, desiredBlocks);
    const std::size_t numBlocks = part.num_blocks == 0 ? std::size_t{1} : part.num_blocks;

    for (std::size_t b = 0; b < numBlocks; ++b) {
      T *slab = partialSums + (b * k * d);
      T *cslab = partialComps + (b * k * d);
      std::int32_t *nslab = partialCounts + (b * k);
      for (std::size_t e = 0; e < k * d; ++e) {
        slab[e] = T{0};
        cslab[e] = T{0};
      }
      for (std::size_t c = 0; c < k; ++c) {
        nslab[c] = 0;
      }
    }

    auto scatterRange = [&](std::size_t lo, std::size_t hi) noexcept {
      const std::size_t b = part.blockIndexOf(lo);
      T *slab = partialSums + (b * k * d);
      T *cslab = partialComps + (b * k * d);
      std::int32_t *nslab = partialCounts + (b * k);
      for (std::size_t i = lo; i < hi; ++i) {
        const std::int32_t lbl = labels(i);
        if (lbl < 0 || std::cmp_greater_equal(lbl, k)) {
          continue;
        }
        const auto row = static_cast<std::size_t>(lbl);
        const T *xRow = X.data() + (i * d);
        T *sumRow = slab + (row * d);
        T *compRow = cslab + (row * d);
        for (std::size_t t = 0; t < d; ++t) {
          const T y = xRow[t] - compRow[t];
          const T tVal = sumRow[t] + y;
          compRow[t] = (tVal - sumRow[t]) - y;
          sumRow[t] = tVal;
        }
        nslab[row] += 1;
      }
    };

    if (willParallelize) {
      pool.pool
          ->submit_blocks(
              std::size_t{0}, n, [&](std::size_t lo, std::size_t hi) { scatterRange(lo, hi); },
              numBlocks)
          .wait();
    } else {
      scatterRange(0, n);
    }

    for (std::size_t b = 0; b < numBlocks; ++b) {
      const T *slab = partialSums + (b * k * d);
      const T *cslab = partialComps + (b * k * d);
      const std::int32_t *nslab = partialCounts + (b * k);
      for (std::size_t c = 0; c < k; ++c) {
        m_counts(c) += nslab[c];
        const T *src = slab + (c * d);
        const T *comp = cslab + (c * d);
        T *dstRow = &m_sums(c, 0);
        T *foldRow = foldComp + (c * d);
        for (std::size_t t = 0; t < d; ++t) {
          const T addend = src[t] - comp[t];
          const T y = addend - foldRow[t];
          const T tVal = dstRow[t] + y;
          foldRow[t] = (tVal - dstRow[t]) - y;
          dstRow[t] = tVal;
        }
      }
    }
  }

  void recomputeMinDistSqDirect(const NDArray<T, 2, Layout::Contig> &X,
                                const NDArray<T, 2, Layout::Contig> &centroids,
                                const NDArray<std::int32_t, 1> &labels, math::Pool pool) noexcept {
    const std::size_t n = X.dim(0);
    const std::size_t d = X.dim(1);
    const std::size_t k = centroids.dim(0);
    if (n == 0 || d == 0 || k == 0) {
      return;
    }

    auto runRowRange = [&](std::size_t lo, std::size_t hi) noexcept {
      for (std::size_t i = lo; i < hi; ++i) {
        const std::int32_t lbl = labels(i);
        if (lbl < 0 || std::cmp_greater_equal(lbl, k)) {
          m_minDistSq(i) = T{0};
          continue;
        }
        const T *xRow = X.data() + (i * d);
        const T *cRow = centroids.data() + (static_cast<std::size_t>(lbl) * d);
        T sum = T{0};
        for (std::size_t t = 0; t < d; ++t) {
          const T diff = xRow[t] - cRow[t];
          sum += diff * diff;
        }
        m_minDistSq(i) = sum;
      }
    };

    if (pool.shouldParallelize(n, 64, 2) && pool.pool != nullptr) {
      pool.pool
          ->submit_blocks(std::size_t{0}, n,
                          [&](std::size_t lo, std::size_t hi) { runRowRange(lo, hi); })
          .wait();
    } else {
      runRowRange(0, n);
    }
  }

  NDArray<T, 2, Layout::Contig> m_centroidsOld;
  NDArray<T, 1> m_cSqNorms;
  NDArray<T, 2, Layout::Contig> m_sums;
  NDArray<std::int32_t, 1> m_counts;
  NDArray<T, 1> m_minDistSq;
  NDArray<T, 1> m_shiftSq;
  NDArray<T, 1> m_partialSums;
  NDArray<T, 1> m_partialComps;
  NDArray<std::int32_t, 1> m_partialCounts;
  NDArray<T, 1> m_foldComp;
  NDArray<T, 1> m_packedB;
  NDArray<T, 1> m_packedCSqNorms;
  NDArray<T, 2, Layout::Contig> m_distsChunk;
  NDArray<T, 1> m_gemmApArena;
  NDArray<T, 1> m_xChunkNormsSq;

  std::size_t m_n = 0;
  std::size_t m_d = 0;
  std::size_t m_k = 0;
  std::size_t m_workerCount = 0;
};

} // namespace clustering::kmeans
