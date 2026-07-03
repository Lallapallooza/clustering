#pragma once

#include <citor/cancellation.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>
#include <utility>

#include "clustering/always_assert.h"
#include "clustering/kmeans/detail/convergence.h"
#include "clustering/kmeans/detail/empty_cluster.h"
#include "clustering/kmeans/policy/greedy_kmpp_seeder.h"
#include "clustering/math/centroid_shift.h"
#include "clustering/math/defaults.h"
#include "clustering/math/detail/avx2_helpers.h"
#include "clustering/math/detail/columnwise_reduce_avx2.h"
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
 * The partition maps `[first_index, first_index + span)` onto at most @c desired blocks; slot
 * @c s covers `[first_index + span*s/num_blocks, first_index + span*(s+1)/num_blocks)`. The
 * citor-style formula is identical to the one used by @c math::Pool::parallelForExactBlocks
 * on both backends so a starting @p lo round-trips to the originating slot index. Ascending-
 * block-order reduction is the deterministic fold the label-accumulation step relies on, which
 * pins bit-identity across nJobs settings at @c n_jobs = 1.
 */
struct BlockPartition {
  std::size_t first_index = 0;
  std::size_t span = 0;
  std::size_t num_blocks = 0;

  BlockPartition(std::size_t first, std::size_t n, std::size_t desired) noexcept
      : first_index(first), span(n) {
    if (n == 0 || desired == 0) {
      num_blocks = 0;
      span = 0;
      return;
    }
    num_blocks = std::min(desired, n);
  }

  [[nodiscard]] std::size_t blockIndexOf(std::size_t lo) const noexcept {
    if (num_blocks == 0 || span == 0) {
      return 0;
    }
    // Invert the citor-style partition `[first + span*s/P, first + span*(s+1)/P)` using
    // `ceil((rel + 1) * P / span) - 1`, so left boundaries map to their own slot.
    const std::size_t rel = lo - first_index;
    const std::size_t s = (((rel + 1) * num_blocks) - 1) / span;
    return s >= num_blocks ? num_blocks - 1 : s;
  }
};

/**
 * @brief Maximum @c d for the direct-compute argmin hot path.
 *
 * At `d <= this` threshold the fused argmin-GEMM driver's @c packA + packB overhead dominates
 * the handful of FMAs the microkernel performs, so the direct `||x - c||^2` formula with 8-row
 * SIMD accumulators beats the packed-GEMM path. Measured on Zen5: crossover sits near
 * `d == 8` where the two paths tie; below that the direct path wins by the pack cost.
 */
inline constexpr std::size_t kDirectArgminMaxD = 8;

} // namespace detail

/**
 * @brief Fused-argmin-GEMM Lloyd driver.
 *
 * Runs the Lloyd iteration over caller-seeded centroids: assignment via the fused AVX2
 * argmin-GEMM hot path at @c d <= @c math::defaults::pairwiseArgminMaxD and the chunked
 * materialized fallback above it, label-grouped fold into per-cluster sums (Kahan-compensated
 * at @c n >= @ref LloydFusedGemm::kahanNThreshold), empty-cluster reseed against the current
 * per-point distance scratch, mean step, and convergence test on the Kahan-summed total squared
 * shift. All scratch buffers live inside the policy instance; no allocation fires between the first
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
        m_gemmApArena({0}), m_xNormsSq({0}), m_varSum({0}), m_varSumSq({0}), m_varPartialSum({0}),
        m_varPartialSumSq({0}), m_u({0}), m_l({0}), m_shiftEuclidean({0}),
        m_halfDistToNearestOther({0}), m_elkanBounds({0, 0}), m_centerDist({0, 0}) {}

#ifdef CLUSTERING_KMEANS_KAHAN_N_THRESHOLD
  /**
   * @brief @c n threshold at which the centroid accumulator switches to the Kahan-compensated
   *        variant. Below this, the plain partial-sum + fold variant is used.
   *
   * Compensation is load-bearing for the 1%-inertia gate at the `(n=1e6, k=1000)` corner where
   * per-cluster running totals are dominated by a large sum plus many small addends. Override
   * with @c -DCLUSTERING_KMEANS_KAHAN_N_THRESHOLD=<value>.
   */
  static constexpr std::size_t kahanNThreshold = CLUSTERING_KMEANS_KAHAN_N_THRESHOLD;
#else
  /// @c n threshold at which the centroid accumulator switches to Kahan-compensated summation.
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
   * @param tol       Convergence tolerance relative to the mean column variance of @p X
   *                  (sklearn convention). Internally converted to a sum-of-shift-squared
   *                  threshold as @c tol * mean(var(X, axis=0)) and compared against the
   *                  Kahan-summed per-centroid shift-squared.
   * @param pool      Parallelism injection.
   * @param outLabels Length-n assignment; each entry in `[0, k)`.
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

    // Sklearn-compatible tol semantics: the threshold on sum(||deltac_j||^2) is @c tol * mean_var
    // where @c mean_var is the mean of per-column variances of @p X. This is scale-invariant,
    // which is the property callers expect when they pass the same numeric @c tol across
    // datasets of different magnitudes. computeXStatistics derives mean_var and the per-row
    // squared norms in one pass over @p X so the two O(n*d) sweeps share a single fan-out.
    // Cache against (X.data(), n, d): the best-of harness calls run() n_init times with the same
    // X, so the recompute is pure overhead once the data pointer and shape match.
    T meanVar;
    if (m_xstatsCachedXData == X.data() && m_xstatsCachedN == n && m_xstatsCachedD == d) {
      meanVar = m_xstatsCachedMeanVar;
    } else {
      meanVar = computeXStatistics(X, pool);
      m_xstatsCachedXData = X.data();
      m_xstatsCachedN = n;
      m_xstatsCachedD = d;
      m_xstatsCachedMeanVar = meanVar;
    }
    const T shiftSqThreshold = tol * meanVar;
    const bool useKahan = n >= kahanNThreshold;

    refreshCentroidSqNorms(centroids);

    std::size_t iter = 0;
    bool converged = false;

    // Hamerly pruning always runs above the direct small-D path. Fused-argmin shapes seed
    // valid per-point bounds after the first dense assignment; chunked shapes seed them inline
    // during the argmin post-pass; direct shapes seed them in a post-pass over the first dense
    // assignment and join only when the scan volume and pool width keep the pruning ahead of
    // the dense tile kernel (see @ref kHamerlyMinDirectScanDims, @ref kHamerlyDirectWorkerCap).
    // @c k is capped by @c kHamerlyMaxK because the per-row scan uses a stack-allocated
    // distance buffer; above that, Elkan handles bounded shapes and the rest fall back to
    // unbounded assignment.
    const bool directHamerly =
        (d * k >= kHamerlyMinDirectScanDims) && (workerCount <= kHamerlyDirectWorkerCap);
    const bool hamerlyEligible =
        ((d > detail::kDirectArgminMaxD) || directHamerly) && (k <= kHamerlyMaxK) && (k >= 2);
    // Elkan keeps k lower bounds per sample instead of Hamerly's one, pruning far more distance
    // work once k exceeds Hamerly's regime. The @c n * k bound matrix grows linearly in both,
    // so we gate on an @c n * k envelope bound (memory ceiling) and require @c k above the
    // Hamerly cap so the two paths don't overlap.
    const bool elkanEligible = (d > math::defaults::pairwiseArgminMaxD) && (k > kHamerlyMaxK) &&
                               (k <= kElkanMaxK) && (n * k <= kElkanNKLimit) && (k >= 2);

    bool ranPlex = false;
#ifdef CLUSTERING_USE_AVX2
    if constexpr (std::is_same_v<T, float>) {
      // The direct/fused assignment family runs every iteration inside one persistent-worker
      // plex: workers stay spin-resident across phases and the serial glue rides the
      // pre-phase hook, dropping the per-iteration fork/join. Chunked shapes join under
      // Hamerly: phase 0 carries the full chunked GEMM assignment, later phases the
      // bounds-aware row scan. Elkan and non-Hamerly chunked shapes keep the per-iteration
      // dispatch below, as do iterations too small to amortize the per-phase epoch cost,
      // where the dispatcher's small-work gates win.
      constexpr std::size_t kMinPlexElems = std::size_t{1} << 16;
      const bool plexChunkedHamerly = hamerlyEligible && d > math::defaults::pairwiseArgminMaxD;
      if (pool.pool != nullptr && workerCount > 1 && maxIter > 0 && (n * d >= kMinPlexElems) &&
          (assignmentProducesDirectMinDistSq(X, centroids) ||
           assignmentUsesFusedArgmin(X, centroids) || plexChunkedHamerly)) {
        runPlexLoop(X, centroids, outLabels, k, maxIter, shiftSqThreshold, useKahan,
                    hamerlyEligible, pool, iter, converged);
        ranPlex = true;
      }
    }
#endif

    while (!ranPlex && iter < maxIter) {
      std::memcpy(m_centroidsOld.data(), centroids.data(),
                  centroids.dim(0) * centroids.dim(1) * sizeof(T));

      if (hamerlyEligible && iter > 0) {
        runHamerlyAssignmentAndScatter(X, centroids, outLabels, k, useKahan, pool);
      } else if (elkanEligible && iter > 0) {
        runElkanAssignmentAndScatter(X, centroids, outLabels, k, useKahan, pool);
      } else {
        runAssignmentAndScatter(X, centroids, outLabels, k, useKahan, pool);
        if (hamerlyEligible && iter == 0 && assignmentUsesFusedArgmin(X, centroids)) {
          seedHamerlyBoundsFromAssignedMinDist(outLabels, k, d, pool);
        }
      }

      (void)::clustering::kmeans::detail::reseedEmptyClusters<T>(X, centroids, m_sums, m_counts,
                                                                 m_minDistSq);
      finalizeMeans(centroids);
      refreshCentroidSqNorms(centroids);

      math::centroidShift<T>(m_centroidsOld, centroids, m_shiftSq, pool);
      const T totalShift = ::clustering::kmeans::detail::totalShiftSqKahan<T>(m_shiftSq);

      ++iter;
      if (totalShift <= shiftSqThreshold) {
        converged = true;
        break;
      }
    }

    // Re-assign labels against the final centroids. Hamerly can refresh exact assigned
    // distances while proving labels, so final inertia can consume m_minDistSq directly.
    bool finalMinDistExact = assignmentProducesDirectMinDistSq(X, centroids);
    if (hamerlyEligible && iter > 0) {
      runHamerlyAssignment(X, centroids, outLabels, pool, true);
      finalMinDistExact = true;
    } else if (elkanEligible && iter > 0) {
      runElkanAssignment(X, centroids, outLabels, math::Pool{});
    } else {
      runAssignment(X, centroids, outLabels, pool);
    }
    if (!finalMinDistExact) {
      recomputeMinDistSqDirect(X, centroids, outLabels, pool);
    }

    outInertia = inertiaKahan(n, pool);
    outNIter = iter;
    outConverged = converged;
  }

private:
  /// Top-2 Euclidean centroid shifts feeding Hamerly's lower-bound decay: `l(x)` loses the
  /// largest shift unless x's own cluster is the mover, in which case it loses the runner-up.
  struct HamerlyShiftTop2 {
    T sMax = T{0};
    T s2Max = T{0};
    std::size_t argMax = 0;
  };

  /// Scatter row @p i into slot @p slot's partial slab keyed by the row's label. Force-inlined
  /// so the low-d tile loops keep the slab bases hoisted instead of paying a call per row.
  [[gnu::always_inline]] inline void scatterRowToSlab(const T *xBase,
                                                      const std::int32_t *labelsBase, std::size_t i,
                                                      std::size_t slot, std::size_t k,
                                                      std::size_t d, bool useKahan) noexcept {
    const std::int32_t lbl = labelsBase[i];
    if (lbl < 0 || std::cmp_greater_equal(lbl, k)) {
      return;
    }
    const auto row = static_cast<std::size_t>(lbl);
    const T *xRow = xBase + (i * d);
    T *sumRow = m_partialSums.data() + (((slot * k) + row) * d);
    std::int32_t *cslab = m_partialCounts.data() + (slot * k);
    if (useKahan) {
      T *compRow = m_partialComps.data() + (((slot * k) + row) * d);
      math::detail::kahanAddRow<T>(xRow, d, sumRow, compRow);
    } else {
      for (std::size_t t = 0; t < d; ++t) {
        sumRow[t] += xRow[t];
      }
    }
    cslab[row] += 1;
  }

  /**
   * @brief Compute per-X column variance and per-row squared norms in a single fan-out.
   *
   * Both quantities are one-shot O(n*d) passes over @p X. Computing them in the same parallel
   * section halves the dispatch overhead at the run() head and keeps caches warm. Returns the
   * mean column variance (used to scale @c tol per sklearn convention); writes per-row norms
   * into @c m_xNormsSq.
   */
  [[nodiscard]] T computeXStatistics(const NDArray<T, 2, Layout::Contig> &X, math::Pool pool) {
    const std::size_t n = X.dim(0);
    const std::size_t d = X.dim(1);
    if (n == 0 || d == 0) {
      return T{0};
    }
    const T *xData = X.data();

    if (m_varSum.dim(0) != d) {
      m_varSum = NDArray<T, 1>({d});
      m_varSumSq = NDArray<T, 1>({d});
    }
    T *colSum = m_varSum.data();
    T *colSumSq = m_varSumSq.data();

    const std::size_t workers = pool.workerCount();
    const bool willParallelize = workers > 1;
    if (willParallelize) {
      const std::size_t partialSize = workers * d;
      if (m_varPartialSum.dim(0) != partialSize) {
        m_varPartialSum = NDArray<T, 1>({partialSize});
        m_varPartialSumSq = NDArray<T, 1>({partialSize});
      }
      T *partialSum = m_varPartialSum.data();
      T *partialSumSq = m_varPartialSumSq.data();
      for (std::size_t e = 0; e < partialSize; ++e) {
        partialSum[e] = T{0};
        partialSumSq[e] = T{0};
      }
      pool.parallelForExactBlocksWithSlot<citor::HintsDefaults>(
          std::size_t{0}, n, workers,
          [&](std::size_t lo, std::size_t hi, std::size_t slot) noexcept {
            T *localSum = partialSum + (slot * d);
            T *localSumSq = partialSumSq + (slot * d);
            for (std::size_t i = lo; i < hi; ++i) {
              const T *row = xData + (i * d);
              math::detail::columnwiseAccumSumSq<T>(row, d, localSum, localSumSq);
              m_xNormsSq(i) = math::detail::sqNormRow<T, Layout::Contig>(X, i);
            }
          });
      for (std::size_t t = 0; t < d; ++t) {
        T s = T{0};
        T ss = T{0};
        for (std::size_t w = 0; w < workers; ++w) {
          s += partialSum[(w * d) + t];
          ss += partialSumSq[(w * d) + t];
        }
        colSum[t] = s;
        colSumSq[t] = ss;
      }
    } else {
      for (std::size_t t = 0; t < d; ++t) {
        colSum[t] = T{0};
        colSumSq[t] = T{0};
      }
      for (std::size_t i = 0; i < n; ++i) {
        const T *row = xData + (i * d);
        math::detail::columnwiseAccumSumSq<T>(row, d, colSum, colSumSq);
        m_xNormsSq(i) = math::detail::sqNormRow<T, Layout::Contig>(X, i);
      }
    }

    const auto nInv = static_cast<T>(1) / static_cast<T>(n);
    T acc = T{0};
    for (std::size_t t = 0; t < d; ++t) {
      const T mean = colSum[t] * nInv;
      acc += (colSumSq[t] * nInv) - (mean * mean);
    }
    return acc / static_cast<T>(d);
  }

  [[nodiscard]] double inertiaKahan(std::size_t n, math::Pool pool) {
    const T *minDist = m_minDistSq.data();
    auto sumRange = [minDist](std::size_t lo, std::size_t hi) noexcept {
      double sum = 0.0;
      double comp = 0.0;
      for (std::size_t i = lo; i < hi; ++i) {
        const auto addend = static_cast<double>(minDist[i]);
        const double y = addend - comp;
        const double t = sum + y;
        comp = (t - sum) - y;
        sum = t;
      }
      return sum;
    };
    return pool.parallelReduce<citor::HintsDefaults>(
        std::size_t{0}, n, 0.0, sumRange,
        [](double lhs, double rhs) noexcept { return lhs + rhs; });
  }

  void ensureShape(std::size_t n, std::size_t d, std::size_t k, std::size_t workerCount) {
    const bool shapeChanged = (n != m_n) || (d != m_d) || (k != m_k);
    const bool workerChanged = (workerCount != m_workerCount);
    if (!shapeChanged && !workerChanged) {
      return;
    }

    const bool needsChunk = d > math::defaults::pairwiseArgminMaxD;
    const std::size_t chunkCap = math::pairwiseArgminChunkRows;
    const std::size_t blocks = workerCount == 0 ? std::size_t{1} : workerCount;
    const bool directHamerlyScratch =
        (d * k >= kHamerlyMinDirectScanDims) && (blocks <= kHamerlyDirectWorkerCap);
    const bool hamerlyScratch = ((d > detail::kDirectArgminMaxD) || directHamerlyScratch) &&
                                (k <= kHamerlyMaxK) && (k >= 2);
    const std::size_t partialBlocks = hamerlyScratch ? hamerlyScatterBlocks(n, blocks) : blocks;

    if (shapeChanged) {
      m_centroidsOld = NDArray<T, 2, Layout::Contig>({k, d});
      m_cSqNorms = NDArray<T, 1>({k});
      m_sums = NDArray<T, 2, Layout::Contig>({k, d});
      m_counts = NDArray<std::int32_t, 1>({k});
      m_minDistSq = NDArray<T, 1>({n});
      m_xNormsSq = NDArray<T, 1>({n});
      m_shiftSq = NDArray<T, 1>({k});
      m_foldComp = NDArray<T, 1>({k * d});
      // Hamerly bound scratch: per-point upper/lower Euclidean bounds and per-cluster sqrt
      // shifts. Seeded by the first iteration's full scan and maintained by the bounds-aware
      // reassignment on subsequent iterations.
      m_u = NDArray<T, 1>({n});
      m_l = NDArray<T, 1>({n});
      m_shiftEuclidean = NDArray<T, 1>({k});
      m_halfDistToNearestOther = NDArray<T, 1>({k});
      // Elkan's @c n * k bound matrix and the per-pair centroid-distance matrix are only
      // touched when @c k > @ref kHamerlyMaxK (Hamerly handles smaller k). Skip the n*k
      // allocation entirely at small k -- the n*k slab dominates per-call
      // alloc cost when KMeans is constructed fresh per binding call.
      const bool elkanCanFire = (k > kHamerlyMaxK) && (k <= kElkanMaxK) && (n * k <= kElkanNKLimit);
      if (elkanCanFire) {
        m_elkanBounds = NDArray<T, 2, Layout::Contig>({n, k});
        m_centerDist = NDArray<T, 2, Layout::Contig>({k, k});
      } else {
        m_elkanBounds = NDArray<T, 2, Layout::Contig>({0, 0});
        m_centerDist = NDArray<T, 2, Layout::Contig>({0, 0});
      }
      // Packed-B sizing: the fused fast path at d<=pairwiseArgminMaxD uses the flat
      // panel-per-centroid layout (ceil(k/Nr)*Nr*d); the chunked fallback uses the tiled
      // (jcIdx, pcIdx) layout that @c gemmRunPrepacked expects and is what supports d > kKc
      // and k > kNc without envelope asserts.
      const std::size_t packedBSize = needsChunk
                                          ? math::detail::packedBScratchSizeFloatsTiled<T>(k, d)
                                          : math::detail::packedBScratchSizeFloats(k, d);
      const std::size_t packedNormsSize = math::detail::packedCSqNormsScratchSizeFloats(k);
      m_packedB = NDArray<T, 1>({packedBSize == 0 ? std::size_t{1} : packedBSize});
      m_packedCSqNorms = NDArray<T, 1>({packedNormsSize == 0 ? std::size_t{1} : packedNormsSize});
      // Per-worker distance tile for the chunked path: one chunkCap*k slab per worker so
      // the chunk fan-out runs without touching a shared tile.
      const std::size_t distRows = needsChunk ? (blocks * chunkCap) : std::size_t{1};
      const std::size_t safeK = (k == 0) ? std::size_t{1} : k;
      const std::size_t distCols = needsChunk ? safeK : std::size_t{1};
      m_distsChunk = NDArray<T, 2, Layout::Contig>({distRows, distCols});
    } else if (workerChanged) {
      // Only the per-worker slabs depend on workerCount; resize them if d triggered needsChunk.
      if (needsChunk) {
        const std::size_t distRows = blocks * chunkCap;
        const std::size_t distCols = (k == 0 ? std::size_t{1} : k);
        m_distsChunk = NDArray<T, 2, Layout::Contig>({distRows, distCols});
      }
    }

    // Per-block scratch sizing for scatter-and-fold. Hamerly may use oversubscribed deterministic
    // slots so its row ranges fold in a stable order across repeated threaded fits.
    m_partialSums = NDArray<T, 1>({partialBlocks * k * d});
    m_partialComps = NDArray<T, 1>({partialBlocks * k * d});
    m_partialCounts = NDArray<std::int32_t, 1>({partialBlocks * k});

    // Gemm A-pack arena sized to @c blocks * kMc * kKc so @c gemmRunPrepacked's per-worker
    // slice indexing stays in-bounds on every fan-out path.
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

  [[nodiscard]] bool assignmentUsesFusedArgmin(const NDArray<T, 2, Layout::Contig> &X,
                                               const NDArray<T, 2, Layout::Contig> &C) noexcept {
#ifdef CLUSTERING_USE_AVX2
    if constexpr (std::is_same_v<T, float>) {
      const std::size_t d = X.dim(1);
      return X.template isAligned<32>() && C.template isAligned<32>() &&
             d > detail::kDirectArgminMaxD && d <= math::defaults::pairwiseArgminMaxD;
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

  /**
   * @brief Pack the centroid matrix into the tiled `(jcIdx, pcIdx)` layout matching
   *        @c gemmRunPrepacked's walk.
   *
   * Supports arbitrary @c d and @c k by splitting into `kKc<T>` and `kNc<T>` tiles
   * respectively; the flat single-tile packB the fused path uses at small d cannot represent
   * @c d > kKc<T> or @c k > kNc<T>.
   */
  void packCentroidsTiled(const NDArray<T, 2, Layout::Contig> &centroids) noexcept {
    constexpr std::size_t kNr = math::detail::kKernelNr<T>;
    constexpr std::size_t kKcVal = math::detail::kKc<T>;
    constexpr std::size_t kNcVal = math::detail::kNc<T>;
    const std::size_t k = centroids.dim(0);
    const std::size_t d = centroids.dim(1);
    const auto cTransposed = centroids.t();
    const auto cDesc = ::clustering::detail::describeMatrix(cTransposed);
    T *bp = m_packedB.data();
    std::size_t jcBase = 0;
    for (std::size_t jc = 0; jc < k; jc += kNcVal) {
      const std::size_t nc = (jc + kNcVal <= k) ? kNcVal : (k - jc);
      const std::size_t roundedNc = ((nc + kNr - 1) / kNr) * kNr;
      std::size_t pcOffInJc = 0;
      for (std::size_t pc = 0; pc < d; pc += kKcVal) {
        const std::size_t kc = (pc + kKcVal <= d) ? kKcVal : (d - pc);
        math::detail::packB<T>(cDesc, pc, kc, jc, nc, bp + jcBase + pcOffInJc);
        pcOffInJc += kc * roundedNc;
      }
      jcBase += d * roundedNc;
    }
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

    packCentroidsTiled(centroids);

    constexpr std::size_t kMcVal = math::detail::kMc<T>;
    constexpr std::size_t kKcVal = math::detail::kKc<T>;
    const std::size_t chunkCap = math::pairwiseArgminChunkRows;
    const std::size_t numChunks = (n + chunkCap - 1) / chunkCap;
    const T *bp = m_packedB.data();
    T *apArena = m_gemmApArena.data();
    T *distsBase = m_distsChunk.data();
    const T *cNormsBase = m_cSqNorms.data();
    T *minDistBase = m_minDistSq.data();
    std::int32_t *labelsBase = labels.data();
    const T *xBase = X.data();
    // Hamerly bound seeding happens inline in the argmin post-pass; the cost is a handful of
    // extra comparisons per row plus two @c sqrt calls, far below a second pass over @p X.
    T *uBase = m_u.data();
    T *lBase = m_l.data();
    // Elkan bound seeding lights up only when the scratch matrix was sized for this shape;
    // at shapes past @ref kElkanNKLimit or @ref kElkanMaxK the pointer stays null and the
    // per-row loop skips the per-cluster bound stores.
    T *elkanBoundsBase = m_elkanBounds.dim(0) == n ? m_elkanBounds.data() : nullptr;

    auto runOneChunk = [&](std::size_t chunkIdx) noexcept {
      const std::size_t iBase = chunkIdx * chunkCap;
      const std::size_t chunkRows = (iBase + chunkCap <= n) ? chunkCap : (n - iBase);
      const std::size_t w = math::Pool::workerIndex();
      T *distsChunk = distsBase + (w * chunkCap * k);
      T *apSlice = apArena + (w * kMcVal * kKcVal);

      auto xChunk = NDArray<T, 2, Layout::Contig>::borrow(const_cast<T *>(xBase) + (iBase * d),
                                                          {chunkRows, d});
      auto distsView = NDArray<T, 2>::borrow(distsChunk, {chunkRows, k});
      const auto xDesc = ::clustering::detail::describeMatrix(xChunk);
      auto distsDesc = ::clustering::detail::describeMatrixMut(distsView);
      // Serial GEMM inside the chunk; outer fan-out already owns parallelism.
      math::detail::gemmRunPrepacked<T>(xDesc, bp, d, k, distsDesc, T{-2}, T{0}, apSlice,
                                        math::Pool{});

      const T *xNormsChunk = m_xNormsSq.data() + iBase;
      for (std::size_t i = 0; i < chunkRows; ++i) {
        const T xn = xNormsChunk[i];
        const T *row = distsChunk + (i * k);
        T *elkanRow = elkanBoundsBase != nullptr ? elkanBoundsBase + ((iBase + i) * k) : nullptr;
        T bestVal = std::numeric_limits<T>::infinity();
        T secondVal = std::numeric_limits<T>::infinity();
        std::int32_t bestIdx = 0;
        for (std::size_t j = 0; j < k; ++j) {
          T v = row[j] + xn + cNormsBase[j];
          if (v < T{0}) {
            v = T{0};
          }
          if (elkanRow != nullptr) {
            elkanRow[j] = std::sqrt(v);
          }
          if (v < bestVal) {
            secondVal = bestVal;
            bestVal = v;
            bestIdx = static_cast<std::int32_t>(j);
          } else if (v < secondVal) {
            secondVal = v;
          }
        }
        minDistBase[iBase + i] = bestVal;
        labelsBase[iBase + i] = bestIdx;
        uBase[iBase + i] = std::sqrt(bestVal);
        lBase[iBase + i] = std::sqrt(secondVal);
      }
    };

    pool.parallelForBlocks(std::size_t{0}, numChunks, std::size_t{0},
                           [&](std::size_t lo, std::size_t hi) {
                             for (std::size_t c = lo; c < hi; ++c) {
                               runOneChunk(c);
                             }
                           });
  }

  /**
   * @brief Fused assignment + per-cluster scatter.
   *
   * One parallel-for per Lloyd iteration: each slot owns a contiguous range of mTiles or
   * chunks; the worker body runs the appropriate argmin kernel for its slice and immediately
   * scatters the resulting rows into its private slab. Folding assignment and scatter into one
   * barrier avoids the second fan-out and the round-trip through a separate partial-sum scatter.
   *
   * Only invoked from the first Lloyd iteration (or from no-prune shapes); the Hamerly and
   * Elkan paths run their own fused per-row body since they conditionally update labels.
   */
  void runAssignmentAndScatter(const NDArray<T, 2, Layout::Contig> &X,
                               const NDArray<T, 2, Layout::Contig> &centroids,
                               NDArray<std::int32_t, 1> &labels, std::size_t k, bool useKahan,
                               math::Pool pool) {
    const std::size_t n = X.dim(0);
    const std::size_t d = X.dim(1);
    const std::size_t workers = pool.workerCount();
    if (n == 0 || k == 0 || d == 0) {
      return;
    }

    preZeroPartialSlabs(useKahan, workers, k, d);

#ifdef CLUSTERING_USE_AVX2
    const bool aligned32 = X.template isAligned<32>() && centroids.template isAligned<32>();
#else
    const bool aligned32 = false;
#endif
    const bool useDirect = aligned32 && d <= detail::kDirectArgminMaxD;
    const bool useFused = aligned32 && !useDirect && d <= math::defaults::pairwiseArgminMaxD;
    const bool useChunked = !useDirect && !useFused;

#ifdef CLUSTERING_USE_AVX2
    if constexpr (std::is_same_v<T, float>) {
      if (useFused) {
        math::detail::packCentroidsForFusedArgminF32(centroids, k, d, m_packedB.data());
        math::detail::packCSqNorms<float>(m_cSqNorms.data(), k, m_packedCSqNorms.data());
      }
    }
#endif
    if (useChunked) {
      packCentroidsTiled(centroids);
    }

#ifdef CLUSTERING_USE_AVX2
    if constexpr (std::is_same_v<T, float>) {
      if (useDirect) {
        constexpr std::size_t kMr8 = 8;
        constexpr std::size_t kMr16 = 16;
        const bool useWideTile = workers > 1;
        const std::size_t mTiles =
            useWideTile ? ((n + kMr16 - 1) / kMr16) : ((n + kMr8 - 1) / kMr8);
        pool.parallelForExactBlocksWithSlot<citor::HintsDefaults>(
            std::size_t{0}, mTiles, workers,
            [&](std::size_t lo, std::size_t hi, std::size_t slot) noexcept {
              if (useWideTile) {
                assignScatterDirect16Tiles(X, centroids, labels, k, useKahan, lo, hi, slot);
              } else {
                assignScatterDirectTiles(X, centroids, labels, k, useKahan, lo, hi, slot);
              }
            });
        foldPartialSlabs(useKahan, workers, k, d);
        return;
      }
      if (useFused) {
        constexpr std::size_t kMr = math::detail::kKernelMr<float>;
        const std::size_t mTiles = (n + kMr - 1) / kMr;
        pool.parallelForExactBlocksWithSlot<citor::HintsDefaults>(
            std::size_t{0}, mTiles, workers,
            [&](std::size_t lo, std::size_t hi, std::size_t slot) noexcept {
              assignScatterFusedTiles(X, labels, k, useKahan, lo, hi, slot);
            });
        foldPartialSlabs(useKahan, workers, k, d);
        return;
      }
    }
#endif

    // Chunked path (d > pairwiseArgminMaxD or T == double): fan out over chunks.
    const std::size_t chunkCap = math::pairwiseArgminChunkRows;
    const std::size_t numChunks = (n + chunkCap - 1) / chunkCap;
    pool.parallelForExactBlocksWithSlot<citor::HintsDefaults>(
        std::size_t{0}, numChunks, workers,
        [&](std::size_t chunkLo, std::size_t chunkHi, std::size_t slot) noexcept {
          assignScatterChunkRange(X, labels, k, useKahan, chunkLo, chunkHi, slot);
        });

    foldPartialSlabs(useKahan, workers, k, d);
  }

  /// Chunked GEMM assignment + scatter over chunks `[chunkLo, chunkHi)` into slot @p slot's
  /// slab and distance/A-pack slices; reads the tiled centroid pack in @c m_packedB, which
  /// the caller has already refreshed. The argmin post-pass seeds the Hamerly bounds (and
  /// Elkan's, when the bound matrix is sized for this shape) inline.
  void assignScatterChunkRange(const NDArray<T, 2, Layout::Contig> &X,
                               NDArray<std::int32_t, 1> &labels, std::size_t k, bool useKahan,
                               std::size_t chunkLo, std::size_t chunkHi,
                               std::size_t slot) noexcept {
    constexpr std::size_t kMcVal = math::detail::kMc<T>;
    constexpr std::size_t kKcVal = math::detail::kKc<T>;
    const std::size_t n = X.dim(0);
    const std::size_t d = X.dim(1);
    const std::size_t chunkCap = math::pairwiseArgminChunkRows;
    const T *xBase = X.data();
    std::int32_t *labelsBase = labels.data();
    const T *bp = m_packedB.data();
    const T *cNormsBase = m_cSqNorms.data();
    T *minDistBase = m_minDistSq.data();
    T *uBase = m_u.data();
    T *lBase = m_l.data();
    T *elkanBoundsBase = m_elkanBounds.dim(0) == n ? m_elkanBounds.data() : nullptr;
    T *distsChunk = m_distsChunk.data() + (slot * chunkCap * k);
    T *apSlice = m_gemmApArena.data() + (slot * kMcVal * kKcVal);

    for (std::size_t c = chunkLo; c < chunkHi; ++c) {
      const std::size_t iBase = c * chunkCap;
      const std::size_t chunkRows = (iBase + chunkCap <= n) ? chunkCap : (n - iBase);

      auto xChunk = NDArray<T, 2, Layout::Contig>::borrow(const_cast<T *>(xBase) + (iBase * d),
                                                          {chunkRows, d});
      auto distsView = NDArray<T, 2>::borrow(distsChunk, {chunkRows, k});
      const auto xDesc = ::clustering::detail::describeMatrix(xChunk);
      auto distsDesc = ::clustering::detail::describeMatrixMut(distsView);
      // Serial GEMM inside the chunk; the outer fan-out already owns parallelism.
      math::detail::gemmRunPrepacked<T>(xDesc, bp, d, k, distsDesc, T{-2}, T{0}, apSlice,
                                        math::Pool{});

      const T *xNormsChunk = m_xNormsSq.data() + iBase;
      for (std::size_t i = 0; i < chunkRows; ++i) {
        const T xn = xNormsChunk[i];
        const T *row = distsChunk + (i * k);
        T *elkanRow = elkanBoundsBase != nullptr ? elkanBoundsBase + ((iBase + i) * k) : nullptr;
        T bestVal = std::numeric_limits<T>::infinity();
        T secondVal = std::numeric_limits<T>::infinity();
        std::int32_t bestIdx = 0;
        for (std::size_t j = 0; j < k; ++j) {
          T v = row[j] + xn + cNormsBase[j];
          if (v < T{0}) {
            v = T{0};
          }
          if (elkanRow != nullptr) {
            elkanRow[j] = std::sqrt(v);
          }
          if (v < bestVal) {
            secondVal = bestVal;
            bestVal = v;
            bestIdx = static_cast<std::int32_t>(j);
          } else if (v < secondVal) {
            secondVal = v;
          }
        }
        minDistBase[iBase + i] = bestVal;
        labelsBase[iBase + i] = bestIdx;
        uBase[iBase + i] = std::sqrt(bestVal);
        lBase[iBase + i] = std::sqrt(secondVal);
        scatterRowToSlab(xBase, labelsBase, iBase + i, slot, k, d, useKahan);
      }
    }
  }

#ifdef CLUSTERING_USE_AVX2
  /// Direct small-d argmin + scatter over M-tiles `[tileLo, tileHi)` into slot @p slot's slab.
  void assignScatterDirectTiles(const NDArray<T, 2, Layout::Contig> &X,
                                const NDArray<T, 2, Layout::Contig> &centroids,
                                NDArray<std::int32_t, 1> &labels, std::size_t k, bool useKahan,
                                std::size_t tileLo, std::size_t tileHi, std::size_t slot) noexcept {
    constexpr std::size_t kMr = 8;
    const std::size_t n = X.dim(0);
    const std::size_t d = X.dim(1);
    const T *xBase = X.data();
    const std::int32_t *labelsBase = labels.data();
    for (std::size_t t = tileLo; t < tileHi; ++t) {
      math::detail::argminDirectMTileF32(X, centroids, labels, m_minDistSq, t, n, k, d);
      const std::size_t iBase = t * kMr;
      const std::size_t mc = (iBase + kMr <= n) ? kMr : (n - iBase);
      for (std::size_t r = 0; r < mc; ++r) {
        scatterRowToSlab(xBase, labelsBase, iBase + r, slot, k, d, useKahan);
      }
    }
  }

  /// Direct 16-row argmin + scatter over M-tiles `[tileLo, tileHi)` into slot @p slot's slab.
  void assignScatterDirect16Tiles(const NDArray<T, 2, Layout::Contig> &X,
                                  const NDArray<T, 2, Layout::Contig> &centroids,
                                  NDArray<std::int32_t, 1> &labels, std::size_t k, bool useKahan,
                                  std::size_t tileLo, std::size_t tileHi,
                                  std::size_t slot) noexcept {
    constexpr std::size_t kMr = 16;
    const std::size_t n = X.dim(0);
    const std::size_t d = X.dim(1);
    const T *xBase = X.data();
    const std::int32_t *labelsBase = labels.data();
    for (std::size_t t = tileLo; t < tileHi; ++t) {
      math::detail::argminDirectM16TileF32(X, centroids, labels, m_minDistSq, t, n, k, d);
      const std::size_t iBase = t * kMr;
      const std::size_t mc = (iBase + kMr <= n) ? kMr : (n - iBase);
      for (std::size_t r = 0; r < mc; ++r) {
        scatterRowToSlab(xBase, labelsBase, iBase + r, slot, k, d, useKahan);
      }
    }
  }

  /// Fused argmin-GEMM + scatter over M-tiles `[tileLo, tileHi)`; reads the packed centroid
  /// panels in @c m_packedB / @c m_packedCSqNorms, which the caller has already refreshed.
  void assignScatterFusedTiles(const NDArray<T, 2, Layout::Contig> &X,
                               NDArray<std::int32_t, 1> &labels, std::size_t k, bool useKahan,
                               std::size_t tileLo, std::size_t tileHi, std::size_t slot) noexcept {
    constexpr std::size_t kMr = math::detail::kKernelMr<float>;
    const std::size_t n = X.dim(0);
    const std::size_t d = X.dim(1);
    const T *xBase = X.data();
    const std::int32_t *labelsBase = labels.data();
    const float *bpacked = m_packedB.data();
    const float *normsPacked = m_packedCSqNorms.data();
    for (std::size_t t = tileLo; t < tileHi; ++t) {
      math::detail::argminFusedMTileF32(X, bpacked, normsPacked, labels, m_minDistSq, t, n, k, d);
      const std::size_t iBase = t * kMr;
      const std::size_t mc = (iBase + kMr <= n) ? kMr : (n - iBase);
      for (std::size_t r = 0; r < mc; ++r) {
        scatterRowToSlab(xBase, labelsBase, iBase + r, slot, k, d, useKahan);
      }
    }
  }

  /**
   * @brief Plex-driven Lloyd loop for the direct, fused, and chunked-Hamerly assignment
   *        families.
   *
   * One @c runPlex dispatch drives every iteration: each phase is one assignment+scatter
   * pass over slot-static work ranges (kernel M-tiles for direct/fused, GEMM chunks above
   * the fused @c d cap), and the serial glue (fold, reseed, mean step, shift, convergence
   * test) runs in the pre-phase hook on the producer with happens-before to every worker's
   * phase body. Hamerly phases derive their row range from the slot's unit range, so every
   * row stays with one slot across phases. Convergence stops the plex's cancellation token;
   * the phase the hook precedes still fires once (the producer observes the token at the
   * next boundary), so the body no-ops behind @c stopPhases while the plex drains.
   *
   * @param iter      Incremented once per completed iteration, exactly as the fallback loop.
   * @param converged Set when the Kahan-summed shift falls at or below @p shiftSqThreshold.
   */
  void runPlexLoop(const NDArray<T, 2, Layout::Contig> &X, NDArray<T, 2, Layout::Contig> &centroids,
                   NDArray<std::int32_t, 1> &labels, std::size_t k, std::size_t maxIter,
                   T shiftSqThreshold, bool useKahan, bool hamerlyEligible, math::Pool pool,
                   std::size_t &iter, bool &converged) {
    const std::size_t n = X.dim(0);
    const std::size_t d = X.dim(1);
    const std::size_t workers = pool.workerCount();
    const bool useDirect = d <= detail::kDirectArgminMaxD;
    const bool useChunked = d > math::defaults::pairwiseArgminMaxD;
    // Partition work units, not rows, so a kernel tile or GEMM chunk never straddles two
    // slots.
    const std::size_t unit = useChunked
                                 ? math::pairwiseArgminChunkRows
                                 : (useDirect ? std::size_t{16} : math::detail::kKernelMr<float>);
    const std::size_t units = (n + unit - 1) / unit;

    HamerlyShiftTop2 top2{};
    bool stopPhases = false;
    auto plexTok = citor::CancellationToken::makeOwned();

    auto packFusedCentroids = [&]() noexcept {
      math::detail::packCentroidsForFusedArgminF32(centroids, k, d, m_packedB.data());
      math::detail::packCSqNorms<float>(m_cSqNorms.data(), k, m_packedCSqNorms.data());
    };

    // Fold + serial glue for the iteration whose scatter just finished; identical to the
    // fallback loop's tail. Callers inside the plex pass the serial pool because the
    // workers are plex-resident and cannot pick up a nested dispatch.
    auto iterationGlue = [&](math::Pool gluePool) {
      foldPartialSlabs(useKahan, workers, k, d);
      (void)::clustering::kmeans::detail::reseedEmptyClusters<T>(X, centroids, m_sums, m_counts,
                                                                 m_minDistSq);
      finalizeMeans(centroids);
      refreshCentroidSqNorms(centroids);
      math::centroidShift<T>(m_centroidsOld, centroids, m_shiftSq, gluePool);
      const T totalShift = ::clustering::kmeans::detail::totalShiftSqKahan<T>(m_shiftSq);
      ++iter;
      if (totalShift <= shiftSqThreshold) {
        converged = true;
      }
    };

    auto prePhase = [&](std::size_t phaseIdx) {
      if (phaseIdx == 0) {
        std::memcpy(m_centroidsOld.data(), centroids.data(), k * d * sizeof(T));
        preZeroPartialSlabs(useKahan, workers, k, d);
        if (useChunked) {
          packCentroidsTiled(centroids);
        } else if (!useDirect) {
          packFusedCentroids();
        }
        return;
      }
      iterationGlue(math::Pool{});
      if (converged) {
        stopPhases = true;
        plexTok.request_stop();
        return;
      }
      std::memcpy(m_centroidsOld.data(), centroids.data(), k * d * sizeof(T));
      preZeroPartialSlabs(useKahan, workers, k, d);
      if (hamerlyEligible) {
        top2 = prepareHamerlyGeometry(centroids, k, d);
      } else if (useChunked) {
        packCentroidsTiled(centroids);
      } else if (!useDirect) {
        packFusedCentroids();
      }
    };

    auto phase = [&](std::size_t phaseIdx, std::uint32_t slot, std::size_t lo, std::size_t hi,
                     void * /*tlsArena*/ = nullptr) noexcept {
      if (stopPhases) {
        return;
      }
      const auto s = static_cast<std::size_t>(slot);
      if (hamerlyEligible && phaseIdx > 0) {
        hamerlyAssignScatterRange(X, centroids, labels, k, useKahan, top2, std::min(lo * unit, n),
                                  std::min(hi * unit, n), s);
        return;
      }
      if (useChunked) {
        // The chunk body seeds the Hamerly bounds inline in its argmin post-pass.
        assignScatterChunkRange(X, labels, k, useKahan, lo, hi, s);
        return;
      }
      if (useDirect) {
        assignScatterDirect16Tiles(X, centroids, labels, k, useKahan, lo, hi, s);
      } else {
        assignScatterFusedTiles(X, labels, k, useKahan, lo, hi, s);
      }
      if (hamerlyEligible && phaseIdx == 0) {
        seedHamerlyBoundsFromMinDistRange(labels, k, d, std::min(lo * unit, n),
                                          std::min(hi * unit, n));
      }
    };

    pool.parallelRunPlex<citor::HintsDefaults>(maxIter, units, std::move(phase),
                                               std::move(prePhase), plexTok);

    if (!converged) {
      // No pre-phase hook follows the last phase; its glue runs here, with the caller's
      // pool restored now that the plex has drained.
      iterationGlue(pool);
    }
  }
#endif

  void preZeroPartialSlabs(bool useKahan, std::size_t numBlocks, std::size_t k,
                           std::size_t d) noexcept {
    T *partialSums = m_partialSums.data();
    std::int32_t *partialCounts = m_partialCounts.data();

    for (std::size_t c = 0; c < k; ++c) {
      m_counts(c) = 0;
      for (std::size_t t = 0; t < d; ++t) {
        m_sums(c, t) = T{0};
      }
    }
    if (useKahan) {
      T *foldComp = m_foldComp.data();
      for (std::size_t e = 0; e < k * d; ++e) {
        foldComp[e] = T{0};
      }
      T *partialComps = m_partialComps.data();
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
    } else {
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
    }
  }

  void foldPartialSlabs(bool useKahan, std::size_t numBlocks, std::size_t k,
                        std::size_t d) noexcept {
    const T *partialSums = m_partialSums.data();
    const std::int32_t *partialCounts = m_partialCounts.data();
    if (useKahan) {
      const T *partialComps = m_partialComps.data();
      T *foldComp = m_foldComp.data();
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
    } else {
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
  }

  [[nodiscard]] static std::size_t hamerlyScatterBlocks(std::size_t n,
                                                        std::size_t workers) noexcept {
    if (workers <= 1 || n == 0) {
      return std::max<std::size_t>(workers, std::size_t{1});
    }
    constexpr std::size_t kMinRowsPerBlock = 256;
    const std::size_t byRows = std::max<std::size_t>(1, n / kMinRowsPerBlock);
    const std::size_t blocks = std::min(workers * 8, byRows);
    return std::max(blocks, workers);
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
        m_minDistSq(i) = math::detail::sqEuclideanRowPtr<T>(xRow, cRow, d);
      }
    };

    pool.parallelForBlocks(std::size_t{0}, n, std::size_t{0},
                           [&](std::size_t lo, std::size_t hi) { runRowRange(lo, hi); });
  }

  /// Seed per-row Hamerly bounds from the assignment pass' stored minimum distances.
  void seedHamerlyBoundsFromMinDistRange(const NDArray<std::int32_t, 1> &labels, std::size_t k,
                                         std::size_t d, std::size_t lo, std::size_t hi) noexcept {
    // Fused decomposed distances can round below the direct squared-distance sum; inflate the
    // Euclidean upper bound so Hamerly's prune stays conservative.
    const T slackScale = static_cast<T>(8) * std::numeric_limits<T>::epsilon() * static_cast<T>(d);
    for (std::size_t i = lo; i < hi; ++i) {
      const std::int32_t lbl = labels(i);
      if (lbl < 0 || std::cmp_greater_equal(lbl, k)) {
        m_minDistSq(i) = T{0};
        m_u(i) = std::numeric_limits<T>::infinity();
        m_l(i) = T{0};
        continue;
      }
      T tightSq = m_minDistSq(i);
      if (tightSq < T{0}) {
        tightSq = T{0};
        m_minDistSq(i) = T{0};
      }
      m_minDistSq(i) = tightSq;
      const T u = std::sqrt(tightSq);
      m_u(i) = u + ((u + T{1}) * slackScale);
      m_l(i) = T{0};
    }
  }

  void seedHamerlyBoundsFromAssignedMinDist(const NDArray<std::int32_t, 1> &labels, std::size_t k,
                                            std::size_t d, math::Pool pool) noexcept {
    const std::size_t n = labels.dim(0);
    if (n == 0 || d == 0 || k == 0) {
      return;
    }

    pool.parallelForBlocks(std::size_t{0}, n, std::size_t{0}, [&](std::size_t lo, std::size_t hi) {
      seedHamerlyBoundsFromMinDistRange(labels, k, d, lo, hi);
    });
  }

  /**
   * @brief Maximum @c k supported by the Hamerly per-row scan's stack-allocated distance buffer.
   *
   * Kept modest to bound stack usage; at larger @c k Elkan-style per-centroid bounds beat
   * Hamerly's single lower bound so we'd switch strategies rather than grow the buffer.
   */
  static constexpr std::size_t kHamerlyMaxK = 64;

  /**
   * @brief Minimum @c d * @c k dense-scan volume for Hamerly bounds on the direct small-D
   *        path.
   *
   * A skipped row saves about `d * k` dims of kernel work against a constant bound check,
   * so the scan must be wide enough for the upkeep to pay for itself; below the floor the
   * dense tile kernel recomputes faster than the bounds can prune.
   */
  static constexpr std::size_t kHamerlyMinDirectScanDims = 128;

  /**
   * @brief Widest pool for which Hamerly bounds stay on below the direct-path boundary.
   *
   * Bound pruning trades per-row branches for skipped kernel work, which only wins while
   * the assignment is compute-bound. Wide pools push the low-@c d iteration latency-bound,
   * where the branchy per-row walk loses to the dense tile kernel's throughput even at
   * high skip rates.
   */
  static constexpr std::size_t kHamerlyDirectWorkerCap = 4;

  /**
   * @brief Maximum @c k for which Elkan's @c n * k bound matrix is allowed to fit in memory.
   *
   * Pairs with @ref kElkanNKLimit to cap the scratch footprint across the shape envelope;
   * anything above this falls back to the bound-free chunked assignment every iteration.
   */
  static constexpr std::size_t kElkanMaxK = 4096;

  /**
   * @brief Envelope cap on @c n * k for Elkan eligibility, in elements.
   *
   * At `sizeof(T)` = 4 this caps the bound matrix at @c kElkanNKLimit * 4 bytes (128 MB by
   * default). Shapes above this threshold skip Elkan and keep paying the full chunked
   * assignment each iteration.
   */
  static constexpr std::size_t kElkanNKLimit = std::size_t{32} << 20;

  /**
   * @brief Per-iteration Hamerly geometry: Euclidean shifts, their top-2, and per-cluster
   *        half-distance to the nearest other centroid.
   *
   * The half-distance feeds the Lemma 1 shortcut: when `u(x)` for a sample assigned to
   * cluster @c c clears `halfDist[c]`, triangle inequality pins the sample in @c c because
   * any other centroid is at least `2 * halfDist[c]` away. Fills @c m_shiftEuclidean and
   * @c m_halfDistToNearestOther from @c m_shiftSq and @p centroids; `O(k^2 * d)`, negligible
   * next to the per-sample scan at `k <= kHamerlyMaxK`.
   */
  [[nodiscard]] HamerlyShiftTop2
  prepareHamerlyGeometry(const NDArray<T, 2, Layout::Contig> &centroids, std::size_t k,
                         std::size_t d) noexcept {
    HamerlyShiftTop2 top2{};
    const T *cData = centroids.data();
    T *shiftData = m_shiftEuclidean.data();
    for (std::size_t c = 0; c < k; ++c) {
      const T s = std::sqrt(m_shiftSq(c));
      shiftData[c] = s;
      if (s > top2.sMax) {
        top2.s2Max = top2.sMax;
        top2.sMax = s;
        top2.argMax = c;
      } else if (s > top2.s2Max) {
        top2.s2Max = s;
      }
    }

    T *halfDistData = m_halfDistToNearestOther.data();
    for (std::size_t c = 0; c < k; ++c) {
      T nearestSq = std::numeric_limits<T>::infinity();
      const T *caRow = cData + (c * d);
      for (std::size_t cp = 0; cp < k; ++cp) {
        if (cp == c) {
          continue;
        }
        const T dsq = math::detail::sqEuclideanRowPtr<T>(caRow, cData + (cp * d), d);
        if (dsq < nearestSq) {
          nearestSq = dsq;
        }
      }
      halfDistData[c] = T{0.5} * std::sqrt(nearestSq);
    }
    return top2;
  }

  /// Hamerly bounded assignment + scatter for rows `[lo, hi)` into slot @p slot's slab.
  /// Reads the geometry @ref prepareHamerlyGeometry left in @c m_shiftEuclidean and
  /// @c m_halfDistToNearestOther plus the top-2 shifts in @p top2.
  void hamerlyAssignScatterRange(const NDArray<T, 2, Layout::Contig> &X,
                                 const NDArray<T, 2, Layout::Contig> &centroids,
                                 NDArray<std::int32_t, 1> &labels, std::size_t k, bool useKahan,
                                 const HamerlyShiftTop2 &top2, std::size_t lo, std::size_t hi,
                                 std::size_t slot) noexcept {
    const std::size_t d = X.dim(1);
    const T *xData = X.data();
    const T *cData = centroids.data();
    T *uData = m_u.data();
    T *lData = m_l.data();
    T *minDistData = m_minDistSq.data();
    std::int32_t *labelsData = labels.data();
    const T *shiftData = m_shiftEuclidean.data();
    const T *halfDistData = m_halfDistToNearestOther.data();
    T *slabSum = m_partialSums.data() + (slot * k * d);
    T *slabComp = m_partialComps.data() + (slot * k * d);
    std::int32_t *slabCnt = m_partialCounts.data() + (slot * k);

    std::array<T, kHamerlyMaxK> distBuf{};
    for (std::size_t i = lo; i < hi; ++i) {
      const std::int32_t a = labelsData[i];
      if (a < 0 || std::cmp_greater_equal(a, k)) {
        continue;
      }
      const auto au = static_cast<std::size_t>(a);
      T ui = uData[i] + shiftData[au];
      T li = lData[i] - ((au == top2.argMax) ? top2.s2Max : top2.sMax);
      std::int32_t bestLabel = a;
      bool labelDecided = false;

      if (ui <= li || ui <= halfDistData[au]) {
        uData[i] = ui;
        lData[i] = li;
        labelDecided = true;
      }

      if (!labelDecided) {
        const T *xi = xData + (i * d);
        const T *caRow = cData + (au * d);
        const T tightSq = math::detail::sqEuclideanRowPtr<T>(xi, caRow, d);
        ui = std::sqrt(tightSq);

        if (ui <= li) {
          uData[i] = ui;
          lData[i] = li;
          minDistData[i] = tightSq;
          labelDecided = true;
        } else {
          detail::sqEuclideanRowToBatch<T>(xi, cData, k, d, distBuf.data());
          T best = std::numeric_limits<T>::infinity();
          T second = std::numeric_limits<T>::infinity();
          std::int32_t bestIdx = 0;
          for (std::size_t j = 0; j < k; ++j) {
            const T v = distBuf[j];
            if (v < best) {
              second = best;
              best = v;
              bestIdx = static_cast<std::int32_t>(j);
            } else if (v < second) {
              second = v;
            }
          }
          bestLabel = bestIdx;
          labelsData[i] = bestIdx;
          minDistData[i] = best;
          uData[i] = std::sqrt(best);
          lData[i] = std::sqrt(second);
        }
      }

      if (bestLabel < 0 || std::cmp_greater_equal(bestLabel, k)) {
        continue;
      }
      const auto row = static_cast<std::size_t>(bestLabel);
      const T *xRow = xData + (i * d);
      T *sumRow = slabSum + (row * d);
      if (useKahan) {
        T *compRow = slabComp + (row * d);
        math::detail::kahanAddRow<T>(xRow, d, sumRow, compRow);
      } else {
        for (std::size_t t = 0; t < d; ++t) {
          sumRow[t] += xRow[t];
        }
      }
      slabCnt[row] += 1;
    }
  }

  /**
   * @brief Hamerly assignment fused with per-slot scatter into @c m_partialSums / @c
   * m_partialCounts.
   *
   * Same per-row body as @ref runHamerlyAssignment, but the outer fan-out is one
   * @c parallelForBlocks over row ranges keyed to the worker slot, and after each row's label
   * is decided the row is scattered into the slot's slab. Pre-zeroes slabs and folds at the end
   * so the assignment and scatter share a single fan-out.
   */
  void runHamerlyAssignmentAndScatter(const NDArray<T, 2, Layout::Contig> &X,
                                      const NDArray<T, 2, Layout::Contig> &centroids,
                                      NDArray<std::int32_t, 1> &labels, std::size_t k,
                                      bool useKahan, math::Pool pool) noexcept {
    const std::size_t n = X.dim(0);
    const std::size_t d = X.dim(1);
    if (n == 0 || d == 0 || k == 0 || k > kHamerlyMaxK) {
      return;
    }
    const std::size_t workers = pool.workerCount();
    const std::size_t blocks = hamerlyScatterBlocks(n, workers);
    preZeroPartialSlabs(useKahan, blocks, k, d);

    const HamerlyShiftTop2 top2 = prepareHamerlyGeometry(centroids, k, d);

    pool.parallelForExactBlocksWithSlot<citor::HintsDefaults>(
        std::size_t{0}, n, blocks, [&](std::size_t lo, std::size_t hi, std::size_t slot) noexcept {
          hamerlyAssignScatterRange(X, centroids, labels, k, useKahan, top2, lo, hi, slot);
        });

    foldPartialSlabs(useKahan, blocks, k, d);
  }

  /**
   * @brief Elkan assignment fused with per-slot scatter.
   *
   * Same row body as @ref runElkanAssignment, with scatter appended after the per-row label is
   * decided. Pre-zeroes slabs and folds at the end.
   */
  void runElkanAssignmentAndScatter(const NDArray<T, 2, Layout::Contig> &X,
                                    const NDArray<T, 2, Layout::Contig> &centroids,
                                    NDArray<std::int32_t, 1> &labels, std::size_t k, bool useKahan,
                                    math::Pool pool) noexcept {
    const std::size_t n = X.dim(0);
    const std::size_t d = X.dim(1);
    if (n == 0 || d == 0 || k == 0 || m_elkanBounds.dim(0) != n || m_elkanBounds.dim(1) != k) {
      return;
    }
    const std::size_t workers = pool.workerCount();
    preZeroPartialSlabs(useKahan, workers, k, d);

    const T *xData = X.data();
    const T *cData = centroids.data();
    T *uData = m_u.data();
    T *boundsData = m_elkanBounds.data();
    T *minDistData = m_minDistSq.data();
    std::int32_t *labelsData = labels.data();

    T *shiftData = m_shiftEuclidean.data();
    for (std::size_t c = 0; c < k; ++c) {
      shiftData[c] = std::sqrt(m_shiftSq(c));
    }

    T *centerDistData = m_centerDist.data();
    T *halfDistData = m_halfDistToNearestOther.data();
    for (std::size_t c = 0; c < k; ++c) {
      centerDistData[(c * k) + c] = T{0};
      T nearest = std::numeric_limits<T>::infinity();
      for (std::size_t cp = 0; cp < k; ++cp) {
        if (cp == c) {
          continue;
        }
        T dist;
        if (cp > c) {
          const T dsq = math::detail::sqEuclideanRowPtr<T>(cData + (c * d), cData + (cp * d), d);
          dist = std::sqrt(dsq);
          centerDistData[(c * k) + cp] = dist;
          centerDistData[(cp * k) + c] = dist;
        } else {
          dist = centerDistData[(c * k) + cp];
        }
        if (dist < nearest) {
          nearest = dist;
        }
      }
      halfDistData[c] = T{0.5} * nearest;
    }

    T *partialSums = m_partialSums.data();
    T *partialComps = m_partialComps.data();
    std::int32_t *partialCounts = m_partialCounts.data();

    pool.parallelForBlocks<citor::HintsDefaults>(
        std::size_t{0}, n, std::size_t{0}, [&](std::size_t lo, std::size_t hi) noexcept {
          const std::size_t slot = math::Pool::workerIndex();
          T *slabSum = partialSums + (slot * k * d);
          T *slabComp = partialComps + (slot * k * d);
          std::int32_t *slabCnt = partialCounts + (slot * k);
          for (std::size_t i = lo; i < hi; ++i) {
            std::int32_t a = labelsData[i];
            if (a < 0 || std::cmp_greater_equal(a, k)) {
              continue;
            }
            auto au = static_cast<std::size_t>(a);
            T u = uData[i] + shiftData[au];
            T *lRow = boundsData + (i * k);
            for (std::size_t c = 0; c < k; ++c) {
              T lnew = lRow[c] - shiftData[c];
              if (lnew < T{0}) {
                lnew = T{0};
              }
              lRow[c] = lnew;
            }

            if (u <= halfDistData[au]) {
              uData[i] = u;
            } else {
              bool uTight = false;
              const T *xi = xData + (i * d);
              for (std::size_t c = 0; c < k; ++c) {
                if (c == au) {
                  continue;
                }
                const T lc = lRow[c];
                const T half = T{0.5} * centerDistData[(au * k) + c];
                if (u <= lc || u <= half) {
                  continue;
                }
                if (!uTight) {
                  const T tightSq = math::detail::sqEuclideanRowPtr<T>(xi, cData + (au * d), d);
                  u = std::sqrt(tightSq);
                  minDistData[i] = tightSq;
                  uTight = true;
                  if (u <= lc || u <= half) {
                    continue;
                  }
                }
                const T dSq = math::detail::sqEuclideanRowPtr<T>(xi, cData + (c * d), d);
                const T dEuc = std::sqrt(dSq);
                lRow[c] = dEuc;
                if (dEuc < u) {
                  au = c;
                  a = static_cast<std::int32_t>(c);
                  u = dEuc;
                  minDistData[i] = dSq;
                }
              }
              uData[i] = u;
              labelsData[i] = a;
            }

            const auto row = static_cast<std::size_t>(a);
            const T *xRow = xData + (i * d);
            T *sumRow = slabSum + (row * d);
            if (useKahan) {
              T *compRow = slabComp + (row * d);
              math::detail::kahanAddRow<T>(xRow, d, sumRow, compRow);
            } else {
              for (std::size_t t = 0; t < d; ++t) {
                sumRow[t] += xRow[t];
              }
            }
            slabCnt[row] += 1;
          }
        });

    foldPartialSlabs(useKahan, workers, k, d);
  }

  /**
   * @brief Hamerly bounds-aware assignment for iterations beyond the first.
   *
   * Updates @c m_u and @c m_l against the Euclidean centroid shifts the caller has already
   * computed into @c m_shiftSq, then for each point applies the `u <= l` prune; points that
   * clear the prune have their @c u tightened to the exact distance to the current assigned
   * centroid and rechecked, and only those still above the lower bound fall into the full
   * @c k-distance scan. With @p refreshAssignedMinDist, pruned rows also refresh
   * @c m_minDistSq for final inertia.
   */
  void runHamerlyAssignment(const NDArray<T, 2, Layout::Contig> &X,
                            const NDArray<T, 2, Layout::Contig> &centroids,
                            NDArray<std::int32_t, 1> &labels, math::Pool pool,
                            bool refreshAssignedMinDist = false) noexcept {
    const std::size_t n = X.dim(0);
    const std::size_t d = X.dim(1);
    const std::size_t k = centroids.dim(0);
    if (n == 0 || d == 0 || k == 0 || k > kHamerlyMaxK) {
      return;
    }
    const T *xData = X.data();
    const T *cData = centroids.data();
    T *uData = m_u.data();
    T *lData = m_l.data();
    T *minDistData = m_minDistSq.data();
    std::int32_t *labelsData = labels.data();

    // The second-largest shift is the amount we subtract from `l(x)` when x's assigned
    // cluster is the one with the largest shift -- otherwise the largest shift is the loose
    // bound donor for every non-assigned cluster.
    const HamerlyShiftTop2 top2 = prepareHamerlyGeometry(centroids, k, d);
    const T *shiftData = m_shiftEuclidean.data();
    const T *halfDistData = m_halfDistToNearestOther.data();

    auto processRange = [&](std::size_t lo, std::size_t hi) noexcept {
      std::array<T, kHamerlyMaxK> distBuf{};
      for (std::size_t i = lo; i < hi; ++i) {
        const std::int32_t a = labelsData[i];
        if (a < 0 || std::cmp_greater_equal(a, k)) {
          continue;
        }
        const auto au = static_cast<std::size_t>(a);
        T ui = uData[i] + shiftData[au];
        T li = lData[i] - ((au == top2.argMax) ? top2.s2Max : top2.sMax);

        const bool assignedByLower = ui <= li;
        const bool assignedByCenter = !assignedByLower && ui <= halfDistData[au];
        if (assignedByLower || assignedByCenter) {
          if (refreshAssignedMinDist) {
            const T *xi = xData + (i * d);
            const T *caRow = cData + (au * d);
            const T tightSq = math::detail::sqEuclideanRowPtr<T>(xi, caRow, d);
            minDistData[i] = tightSq;
            ui = std::sqrt(tightSq);
          }
          uData[i] = ui;
          lData[i] = li;
          continue;
        }

        const T *xi = xData + (i * d);
        const T *caRow = cData + (au * d);
        const T tightSq = math::detail::sqEuclideanRowPtr<T>(xi, caRow, d);
        ui = std::sqrt(tightSq);

        if (ui <= li) {
          uData[i] = ui;
          lData[i] = li;
          minDistData[i] = tightSq;
          continue;
        }

        detail::sqEuclideanRowToBatch<T>(xi, cData, k, d, distBuf.data());
        T best = std::numeric_limits<T>::infinity();
        T second = std::numeric_limits<T>::infinity();
        std::int32_t bestIdx = 0;
        for (std::size_t j = 0; j < k; ++j) {
          const T v = distBuf[j];
          if (v < best) {
            second = best;
            best = v;
            bestIdx = static_cast<std::int32_t>(j);
          } else if (v < second) {
            second = v;
          }
        }
        labelsData[i] = bestIdx;
        minDistData[i] = best;
        uData[i] = std::sqrt(best);
        lData[i] = std::sqrt(second);
      }
    };

    pool.parallelForBlocks(std::size_t{0}, n, std::size_t{0},
                           [&](std::size_t lo, std::size_t hi) { processRange(lo, hi); });
  }

  /**
   * @brief Elkan bounds-aware assignment (k lower bounds per sample).
   *
   * At shapes above Hamerly's @c k cap this maintains `m_elkanBounds(i, c)` as a lower bound
   * on `||x_i - c||`. Per iteration the bounds are refreshed against the centroid shifts, the
   * pairwise centroid matrix is recomputed, and each sample runs the classic three-gate prune:
   * Lemma 1 shortcut on the upper bound, per-cluster lower-bound gate, and the
   * @c 0.5 * ||c_a - c|| center-midpoint gate. The tight distance to @c c_a is lazily computed
   * only when the first gate of some cluster @c c requires it.
   */
  void runElkanAssignment(const NDArray<T, 2, Layout::Contig> &X,
                          const NDArray<T, 2, Layout::Contig> &centroids,
                          NDArray<std::int32_t, 1> &labels, math::Pool pool) noexcept {
    const std::size_t n = X.dim(0);
    const std::size_t d = X.dim(1);
    const std::size_t k = centroids.dim(0);
    if (n == 0 || d == 0 || k == 0 || m_elkanBounds.dim(0) != n || m_elkanBounds.dim(1) != k) {
      return;
    }
    const T *xData = X.data();
    const T *cData = centroids.data();
    T *uData = m_u.data();
    T *boundsData = m_elkanBounds.data();
    T *minDistData = m_minDistSq.data();
    std::int32_t *labelsData = labels.data();

    // Per-cluster shift (Euclidean) used to update all bounds once at the top of the pass.
    T *shiftData = m_shiftEuclidean.data();
    for (std::size_t c = 0; c < k; ++c) {
      shiftData[c] = std::sqrt(m_shiftSq(c));
    }

    // Pairwise centroid distances. Symmetric; fill upper triangle and mirror. `O(k^2 * d)`,
    // amortized against the @c n * k inner scan below.
    T *centerDistData = m_centerDist.data();
    T *halfDistData = m_halfDistToNearestOther.data();
    for (std::size_t c = 0; c < k; ++c) {
      centerDistData[(c * k) + c] = T{0};
      T nearest = std::numeric_limits<T>::infinity();
      for (std::size_t cp = 0; cp < k; ++cp) {
        if (cp == c) {
          continue;
        }
        T dist;
        if (cp > c) {
          const T dsq = math::detail::sqEuclideanRowPtr<T>(cData + (c * d), cData + (cp * d), d);
          dist = std::sqrt(dsq);
          centerDistData[(c * k) + cp] = dist;
          centerDistData[(cp * k) + c] = dist;
        } else {
          dist = centerDistData[(c * k) + cp];
        }
        if (dist < nearest) {
          nearest = dist;
        }
      }
      halfDistData[c] = T{0.5} * nearest;
    }

    auto processRange = [&](std::size_t lo, std::size_t hi) noexcept {
      for (std::size_t i = lo; i < hi; ++i) {
        std::int32_t a = labelsData[i];
        if (a < 0 || std::cmp_greater_equal(a, k)) {
          continue;
        }
        auto au = static_cast<std::size_t>(a);
        T u = uData[i] + shiftData[au];
        T *lRow = boundsData + (i * k);
        // Bound-shift pass for this sample: looser lower bounds against all clusters. Done
        // inline so the per-sample walk touches the row exactly once.
        for (std::size_t c = 0; c < k; ++c) {
          T lnew = lRow[c] - shiftData[c];
          if (lnew < T{0}) {
            lnew = T{0};
          }
          lRow[c] = lnew;
        }

        if (u <= halfDistData[au]) {
          uData[i] = u;
          continue;
        }

        bool uTight = false;
        const T *xi = xData + (i * d);
        for (std::size_t c = 0; c < k; ++c) {
          if (c == au) {
            continue;
          }
          const T lc = lRow[c];
          const T half = T{0.5} * centerDistData[(au * k) + c];
          if (u <= lc || u <= half) {
            continue;
          }
          if (!uTight) {
            const T tightSq = math::detail::sqEuclideanRowPtr<T>(xi, cData + (au * d), d);
            u = std::sqrt(tightSq);
            minDistData[i] = tightSq;
            uTight = true;
            if (u <= lc || u <= half) {
              continue;
            }
          }
          const T dSq = math::detail::sqEuclideanRowPtr<T>(xi, cData + (c * d), d);
          const T dEuc = std::sqrt(dSq);
          lRow[c] = dEuc;
          if (dEuc < u) {
            au = c;
            a = static_cast<std::int32_t>(c);
            u = dEuc;
            minDistData[i] = dSq;
          }
        }
        uData[i] = u;
        labelsData[i] = a;
      }
    };

    if (pool.shouldParallelize(n, 64, 2)) {
      pool.parallelForBlocks(std::size_t{0}, n, std::size_t{0},
                             [&](std::size_t lo, std::size_t hi) { processRange(lo, hi); });
    } else {
      processRange(0, n);
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
  NDArray<T, 1> m_xNormsSq;
  NDArray<T, 1> m_varSum;
  NDArray<T, 1> m_varSumSq;
  NDArray<T, 1> m_varPartialSum;
  NDArray<T, 1> m_varPartialSumSq;
  /// Per-point Hamerly upper bound on ||x - c(a(x))||, Euclidean (not squared). Seeded after
  /// the first iteration's full assignment; updated in-place each subsequent iteration.
  NDArray<T, 1> m_u;
  /// Per-point Hamerly lower bound on ||x - c(j != a(x))||, Euclidean. Same seeding/update.
  NDArray<T, 1> m_l;
  /// Per-cluster Euclidean centroid shift for the current iteration, sqrt of m_shiftSq.
  NDArray<T, 1> m_shiftEuclidean;
  /// Per-cluster half-distance to the nearest other centroid. Populated each Hamerly/Elkan
  /// call and consulted by the Lemma 1 shortcut to skip the per-sample tight-distance recompute
  /// when the upper bound already clears the inter-cluster midpoint.
  NDArray<T, 1> m_halfDistToNearestOther;
  /// Elkan's `n x k` per-sample lower-bound matrix (Euclidean, not squared). `m_elkanBounds(i, c)`
  /// is a lower bound on `||x_i - c||`; updated after each centroid shift and refined on the
  /// per-sample scan when the bound is consulted.
  NDArray<T, 2, Layout::Contig> m_elkanBounds;
  /// Elkan's pairwise centroid-distance matrix (Euclidean). `m_centerDist(c, c')` = ||c - c'||,
  /// populated each Elkan call and consulted for the 0.5-times bound shortcut.
  NDArray<T, 2, Layout::Contig> m_centerDist;

  std::size_t m_n = 0;
  std::size_t m_d = 0;
  std::size_t m_k = 0;
  std::size_t m_workerCount = 0;

  // X-shape-stable cache: the mean column variance and m_xNormsSq depend only on X. The best-of
  // harness calls run() n_init times with the same X; recomputing per call is wasted.
  // Invalidated when (data ptr, n, d) changes; first call after a miss recomputes and caches.
  const T *m_xstatsCachedXData = nullptr;
  std::size_t m_xstatsCachedN = 0;
  std::size_t m_xstatsCachedD = 0;
  T m_xstatsCachedMeanVar{0};
};

} // namespace clustering::kmeans
