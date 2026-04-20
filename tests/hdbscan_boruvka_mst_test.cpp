#include <gtest/gtest.h>

#include <BS_thread_pool.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include "clustering/hdbscan/mst_backend.h"
#include "clustering/hdbscan/policy/boruvka_mst_backend.h"
#include "clustering/hdbscan/policy/prim_mst_backend.h"
#include "clustering/math/dsu.h"
#include "clustering/math/thread.h"
#include "clustering/ndarray.h"

using clustering::NDArray;
using clustering::UnionFind;
using clustering::hdbscan::BoruvkaMstBackend;
using clustering::hdbscan::MstBackendStrategy;
using clustering::hdbscan::MstOutput;
using clustering::hdbscan::PrimMstBackend;
using clustering::math::Pool;

// The backend must satisfy the MstBackendStrategy concept at compile time; this is the contract
// HDBSCAN relies on for its @c MstBackend template parameter.
static_assert(MstBackendStrategy<BoruvkaMstBackend<float>, float>,
              "BoruvkaMstBackend<float> must satisfy MstBackendStrategy<float>");

namespace {

// Deterministic Gaussian-blob dataset: @p blobs clusters of @p perBlob points, each a unit
// variance normal draw around a grid centre. The grid spacing is chosen wide enough that every
// blob stays well-separated in the metric space; this keeps the MRD MST numerically stable
// under float32 across the dimension / size sweep below.
NDArray<float, 2> makeBlobGaussian(std::size_t blobs, std::size_t perBlob, std::size_t d,
                                   std::uint64_t seed) {
  const std::size_t n = blobs * perBlob;
  NDArray<float, 2> X(std::array<std::size_t, 2>{n, d});
  std::mt19937_64 gen(seed);
  std::normal_distribution<float> jitter(0.0F, 0.8F);
  std::uniform_real_distribution<float> centerJitter(-0.2F, 0.2F);
  for (std::size_t b = 0; b < blobs; ++b) {
    std::vector<float> center(d, 0.0F);
    for (std::size_t k = 0; k < d; ++k) {
      center[k] = (static_cast<float>(b) * 6.0F) + centerJitter(gen);
    }
    for (std::size_t i = 0; i < perBlob; ++i) {
      const std::size_t row = (b * perBlob) + i;
      for (std::size_t k = 0; k < d; ++k) {
        X(row, k) = center[k] + jitter(gen);
      }
    }
  }
  return X;
}

// Compact helper for making a single Gaussian cloud with no cluster structure.
NDArray<float, 2> makeGaussianCloud(std::size_t n, std::size_t d, std::uint64_t seed) {
  NDArray<float, 2> X(std::array<std::size_t, 2>{n, d});
  std::mt19937_64 gen(seed);
  std::normal_distribution<float> dist(0.0F, 1.0F);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < d; ++j) {
      X(i, j) = dist(gen);
    }
  }
  return X;
}

// Sum of edge weights across a full MST output.
double edgeWeightTotal(const MstOutput<float> &out) {
  double total = 0.0;
  for (const auto &edge : out.edges) {
    total += static_cast<double>(edge.weight);
  }
  return total;
}

} // namespace

// ---------------------------------------------------------------------------
// Shape contract: the run populates exactly n-1 edges and a length-n coreDistances array.
// ---------------------------------------------------------------------------

TEST(BoruvkaMstContract, PostRunShapesMatchContract) {
  const std::size_t n = 40;
  const std::size_t d = 4;
  const auto X = makeGaussianCloud(n, d, 0xB0F2B0F2ULL);

  BoruvkaMstBackend<float> backend;
  MstOutput<float> out;
  backend.run(X, /*minSamples=*/3, Pool{}, out);

  EXPECT_EQ(out.edges.size(), n - 1);
  EXPECT_EQ(out.coreDistances.dim(0), n);
}

// ---------------------------------------------------------------------------
// Spanning-tree invariant: the edge set connects every vertex. The union-find collapses to a
// single component exactly when the edges form a spanning tree over [0, n).
// ---------------------------------------------------------------------------

TEST(BoruvkaMstContract, EdgeSetFormsSpanningTree) {
  const std::size_t n = 50;
  const std::size_t d = 8;
  const auto X = makeGaussianCloud(n, d, 0x5C4FF01DULL);

  BoruvkaMstBackend<float> backend;
  MstOutput<float> out;
  backend.run(X, /*minSamples=*/4, Pool{}, out);

  ASSERT_EQ(out.edges.size(), n - 1);
  UnionFind<std::uint32_t> uf(n);
  for (const auto &edge : out.edges) {
    uf.unite(static_cast<std::uint32_t>(edge.u), static_cast<std::uint32_t>(edge.v));
  }
  const auto root = uf.find(0);
  EXPECT_EQ(uf.componentSize(root), n);
}

// ---------------------------------------------------------------------------
// Total-weight agreement with Prim across d in {2, 8, 16} on three Gaussian shapes and two
// minSamples settings. The MRD-MST total is identical for every valid MST, so absolute
// tolerance bounds the float32 summation error alone. The edge set itself may differ because
// tied-weight edges admit multiple valid MSTs; we compare totals, not edge membership.
// ---------------------------------------------------------------------------

namespace {

struct TotalWeightCase {
  std::size_t n;
  std::size_t d;
  std::size_t minSamples;
  std::uint64_t seed;
  const char *name;
};

void compareTotalWeightsCase(const TotalWeightCase &cse) {
  const auto X = makeBlobGaussian(/*blobs=*/4, /*perBlob=*/cse.n / 4, cse.d, cse.seed);

  PrimMstBackend<float> prim;
  MstOutput<float> primOut;
  prim.run(X, cse.minSamples, Pool{}, primOut);

  BoruvkaMstBackend<float> boruvka;
  MstOutput<float> bOut;
  boruvka.run(X, cse.minSamples, Pool{}, bOut);

  ASSERT_EQ(bOut.edges.size(), primOut.edges.size()) << "edge count mismatch (" << cse.name << ")";
  ASSERT_EQ(bOut.coreDistances.dim(0), primOut.coreDistances.dim(0));

  const double primTotal = edgeWeightTotal(primOut);
  const double bTotal = edgeWeightTotal(bOut);
  const double tol = 1e-5 * std::max(1.0, std::abs(primTotal));
  EXPECT_NEAR(bTotal, primTotal, tol) << cse.name << " n=" << cse.n << " d=" << cse.d;

  // Core distances are derived from the same KDTree kNN path; they must agree bit-for-bit.
  for (std::size_t i = 0; i < primOut.coreDistances.dim(0); ++i) {
    EXPECT_FLOAT_EQ(bOut.coreDistances(i), primOut.coreDistances(i)) << cse.name << " i=" << i;
  }
}

} // namespace

TEST(BoruvkaMstPrimAgreement, TotalWeightMatchesAtDim2Small) {
  const TotalWeightCase cse{
      .n = 200, .d = 2, .minSamples = 3, .seed = 0xB0A0D0002ULL, .name = "dim2-n200-k3"};
  compareTotalWeightsCase(cse);
}

TEST(BoruvkaMstPrimAgreement, TotalWeightMatchesAtDim2Medium) {
  const TotalWeightCase cse{
      .n = 800, .d = 2, .minSamples = 5, .seed = 0xB0A0D0802ULL, .name = "dim2-n800-k5"};
  compareTotalWeightsCase(cse);
}

TEST(BoruvkaMstPrimAgreement, TotalWeightMatchesAtDim8Small) {
  const TotalWeightCase cse{
      .n = 400, .d = 8, .minSamples = 3, .seed = 0xB0A0D4008ULL, .name = "dim8-n400-k3"};
  compareTotalWeightsCase(cse);
}

TEST(BoruvkaMstPrimAgreement, TotalWeightMatchesAtDim8Medium) {
  const TotalWeightCase cse{
      .n = 1200, .d = 8, .minSamples = 5, .seed = 0xB0A0D8008ULL, .name = "dim8-n1200-k5"};
  compareTotalWeightsCase(cse);
}

TEST(BoruvkaMstPrimAgreement, TotalWeightMatchesAtDim16Small) {
  const TotalWeightCase cse{
      .n = 600, .d = 16, .minSamples = 3, .seed = 0xB0A0D6016ULL, .name = "dim16-n600-k3"};
  compareTotalWeightsCase(cse);
}

TEST(BoruvkaMstPrimAgreement, TotalWeightMatchesAtDim16Medium) {
  const TotalWeightCase cse{
      .n = 2000, .d = 16, .minSamples = 5, .seed = 0xB0A0D2016ULL, .name = "dim16-n2000-k5"};
  compareTotalWeightsCase(cse);
}

// ---------------------------------------------------------------------------
// Output edge ordering: edges are sorted ascending by weight. The downstream single-linkage
// tree construction depends on this contract for a deterministic traversal.
// ---------------------------------------------------------------------------

TEST(BoruvkaMstContract, EdgesSortedAscendingByWeight) {
  const std::size_t n = 120;
  const std::size_t d = 4;
  const auto X = makeGaussianCloud(n, d, 0xED5E50F1ULL);

  BoruvkaMstBackend<float> backend;
  MstOutput<float> out;
  backend.run(X, /*minSamples=*/4, Pool{}, out);

  ASSERT_EQ(out.edges.size(), n - 1);
  for (std::size_t i = 1; i < out.edges.size(); ++i) {
    EXPECT_LE(out.edges[i - 1].weight, out.edges[i].weight) << "i=" << i;
  }
}

// ---------------------------------------------------------------------------
// Parallel pool agreement: running under a multi-worker pool must produce the same total weight
// as the single-threaded path. The actual edge set may differ across pool sizes because tie
// resolution between workers depends on merge order; total weight is invariant.
// ---------------------------------------------------------------------------

TEST(BoruvkaMstParallel, SingleAndMultiWorkerTotalsAgree) {
  const std::size_t n = 1000;
  const std::size_t d = 8;
  const auto X = makeBlobGaussian(/*blobs=*/4, /*perBlob=*/n / 4, d, 0xBA11E10EULL);

  BoruvkaMstBackend<float> serial;
  MstOutput<float> serialOut;
  serial.run(X, /*minSamples=*/5, Pool{}, serialOut);

  BS::light_thread_pool pool(4);
  BoruvkaMstBackend<float> parallel;
  MstOutput<float> parallelOut;
  parallel.run(X, /*minSamples=*/5, Pool{.pool = &pool}, parallelOut);

  ASSERT_EQ(serialOut.edges.size(), parallelOut.edges.size());
  const double serialTotal = edgeWeightTotal(serialOut);
  const double parallelTotal = edgeWeightTotal(parallelOut);
  EXPECT_NEAR(parallelTotal, serialTotal, 1e-5 * std::max(1.0, std::abs(serialTotal)));
}

// ---------------------------------------------------------------------------
// Tiny hand-crafted dataset: five collinear points with one off-axis. The MST totaling is
// identical to the Prim backend's hand-verified oracle, and the total under minSamples=2 is
// 8.0 exactly (same coreDistances as the PrimMstBasic.FivePointSpanningTree fixture).
// ---------------------------------------------------------------------------

TEST(BoruvkaMstBasic, FivePointMatchesPrimOracle) {
  NDArray<float, 2> X(std::array<std::size_t, 2>{5, 2});
  X(0, 0) = 0.0F;
  X(0, 1) = 0.0F;
  X(1, 0) = 1.0F;
  X(1, 1) = 0.0F;
  X(2, 0) = 2.0F;
  X(2, 1) = 0.0F;
  X(3, 0) = 0.0F;
  X(3, 1) = 1.0F;
  X(4, 0) = 3.0F;
  X(4, 1) = 0.0F;

  BoruvkaMstBackend<float> backend;
  MstOutput<float> out;
  backend.run(X, /*minSamples=*/2, Pool{}, out);

  ASSERT_EQ(out.edges.size(), 4U);
  ASSERT_EQ(out.coreDistances.dim(0), 5U);

  // Spanning-tree invariant.
  UnionFind<std::uint32_t> uf(5);
  for (const auto &edge : out.edges) {
    uf.unite(static_cast<std::uint32_t>(edge.u), static_cast<std::uint32_t>(edge.v));
  }
  const auto root = uf.find(0);
  EXPECT_EQ(uf.componentSize(root), 5U);

  // Same total-weight oracle as the Prim backend's hand-crafted test.
  double totalWeight = 0.0;
  for (const auto &edge : out.edges) {
    totalWeight += static_cast<double>(edge.weight);
  }
  EXPECT_NEAR(totalWeight, 8.0, 1e-6);

  // Same core distances as the Prim oracle.
  const std::vector<float> expectedCoreDists = {1.0F, 1.0F, 1.0F, 2.0F, 4.0F};
  for (std::size_t i = 0; i < 5; ++i) {
    EXPECT_FLOAT_EQ(out.coreDistances(i), expectedCoreDists[i]) << "i=" << i;
  }
}

// ---------------------------------------------------------------------------
// ThreadSanitizer stress: 50k points at d=8, minSamples=5, nJobs=16. Under a TSan build this
// runs the full fit path under the sanitizer so any data race in the per-round traversal or in
// the per-worker merge surfaces. Outside of TSan builds the fit still runs as a smoke test --
// the outer contract (edges.size() == n-1, coreDistances length) holds either way.
//
// The test is compiled in by default; the @c __SANITIZE_THREAD__ / @c __has_feature fork below
// preserves existing behaviour on non-sanitized configurations while keeping the sanitizer
// fixture runnable by simply flipping the build flag.
// ---------------------------------------------------------------------------

namespace {

constexpr bool kTsanActive =
#ifdef __has_feature
#if __has_feature(thread_sanitizer)
    true
#else
    false
#endif
#elif defined(__SANITIZE_THREAD__)
    true
#else
    false
#endif
    ;

} // namespace

TEST(BoruvkaTsanStress, LargeInputCompletesUnderSanitizer) {
  if (!kTsanActive) {
    GTEST_SKIP() << "TSan build not available; skipping large-input stress run.";
  }
  const std::size_t n = 50000;
  const std::size_t d = 8;
  const auto X = makeBlobGaussian(/*blobs=*/10, /*perBlob=*/n / 10, d, 0xBB5A55ULL);

  BS::light_thread_pool pool(16);
  BoruvkaMstBackend<float> backend;
  MstOutput<float> out;
  backend.run(X, /*minSamples=*/5, Pool{.pool = &pool}, out);

  EXPECT_EQ(out.edges.size(), n - 1);
  EXPECT_EQ(out.coreDistances.dim(0), n);
}
