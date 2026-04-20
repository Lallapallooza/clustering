#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include "clustering/hdbscan/mst_backend.h"
#include "clustering/hdbscan/policy/prim_mst_backend.h"
#include "clustering/math/dsu.h"
#include "clustering/math/equality.h"
#include "clustering/math/pairwise.h"
#include "clustering/math/thread.h"
#include "clustering/ndarray.h"

using clustering::NDArray;
using clustering::UnionFind;
using clustering::hdbscan::MstBackendStrategy;
using clustering::hdbscan::MstOutput;
using clustering::hdbscan::PrimMstBackend;
using clustering::math::allClose;
using clustering::math::pairwiseSqEuclidean;
using clustering::math::Pool;

// The backend must satisfy the MstBackendStrategy concept at compile time; this is the contract
// HDBSCAN relies on for its @c MstBackend template parameter.
static_assert(MstBackendStrategy<PrimMstBackend<float>, float>,
              "PrimMstBackend<float> must satisfy MstBackendStrategy<float>");

namespace {

// Scalar reference: straightforward double-loop squared Euclidean over X[i] and X[j]. Used as the
// oracle for both the self-aliased pairwise check and the end-to-end MST numeric comparison.
NDArray<float, 2> scalarSqEuclidean(const NDArray<float, 2> &X) {
  const std::size_t n = X.dim(0);
  const std::size_t d = X.dim(1);
  NDArray<float, 2> out(std::array<std::size_t, 2>{n, n});
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      float sum = 0.0F;
      for (std::size_t k = 0; k < d; ++k) {
        const float diff = X(i, k) - X(j, k);
        sum += diff * diff;
      }
      out(i, j) = sum;
    }
  }
  return out;
}

// Deterministic Gaussian point cloud for the self-aliasing shape sweep.
NDArray<float, 2> makeGaussianPoints(std::size_t n, std::size_t d, std::uint64_t seed) {
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

} // namespace

// ---------------------------------------------------------------------------
// Self-aliased pairwiseSqEuclidean(X, X, out) matches the scalar oracle at three shapes. This
// pins the invariant the Prim backend depends on: feeding the same buffer as both operands to
// the pairwise kernel yields a correct symmetric distance matrix.
//
// Tolerance matches the GEMM-path contract elsewhere in the test suite: absolute 1e-4 at the
// GEMM workload sizes (n * n * d >= 100000) accommodates the ||x||^2 + ||y||^2 - 2 x . y
// cancellation residue proportional to d; at the SIMD-path size (64, 2) the residue is zero.
// ---------------------------------------------------------------------------

TEST(PrimMstA3, SelfAliasedPairwiseMatchesScalarN64D2) {
  const auto X = makeGaussianPoints(64, 2, 0xA3A3A301ULL);
  NDArray<float, 2> got(std::array<std::size_t, 2>{64, 64});
  pairwiseSqEuclidean(X, X, got, Pool{});
  const auto expect = scalarSqEuclidean(X);
  EXPECT_TRUE(allClose(expect, got, 1e-5F, 1e-5F));
}

TEST(PrimMstA3, SelfAliasedPairwiseMatchesScalarN512D8) {
  const auto X = makeGaussianPoints(512, 8, 0xA3A3A308ULL);
  NDArray<float, 2> got(std::array<std::size_t, 2>{512, 512});
  pairwiseSqEuclidean(X, X, got, Pool{});
  const auto expect = scalarSqEuclidean(X);
  EXPECT_TRUE(allClose(expect, got, 1e-4F, 1e-4F));
}

TEST(PrimMstA3, SelfAliasedPairwiseMatchesScalarN256D32) {
  const auto X = makeGaussianPoints(256, 32, 0xA3A3A320ULL);
  NDArray<float, 2> got(std::array<std::size_t, 2>{256, 256});
  pairwiseSqEuclidean(X, X, got, Pool{});
  const auto expect = scalarSqEuclidean(X);
  EXPECT_TRUE(allClose(expect, got, 1e-4F, 1e-4F));
}

TEST(PrimMstA3, SelfAliasedPairwiseIsSymmetricAndSmallOnDiagonal) {
  // The direct property of self-aliasing: the output is symmetric (out(i,j) == out(j,i)) and the
  // diagonal is small (out(i,i) is approximately zero, modulo the GEMM cancellation residue).
  // Any bug introduced by sharing the same buffer for both operands would manifest as an
  // asymmetry or a non-zero off-diagonal pattern on the diagonal.
  const std::size_t n = 128;
  const std::size_t d = 16;
  const auto X = makeGaussianPoints(n, d, 0xA3F1A3F1ULL);
  NDArray<float, 2> got(std::array<std::size_t, 2>{n, n});
  pairwiseSqEuclidean(X, X, got, Pool{});
  for (std::size_t i = 0; i < n; ++i) {
    EXPECT_NEAR(got(i, i), 0.0F, 1e-4F) << "i=" << i;
    for (std::size_t j = i + 1; j < n; ++j) {
      EXPECT_FLOAT_EQ(got(i, j), got(j, i)) << "asymmetry at (i=" << i << ", j=" << j << ")";
    }
  }
}

// ---------------------------------------------------------------------------
// Hand-crafted 5-point 2-D dataset with minSamples = 2: expected MST has exactly 4 edges and
// every vertex is reachable from every other through the edge set.
// ---------------------------------------------------------------------------

TEST(PrimMstBasic, FivePointSpanningTree) {
  // Points:
  //   P0 (0, 0), P1 (1, 0), P2 (2, 0), P3 (0, 1), P4 (3, 0)
  // Squared pair distances yield coreDists = [1, 1, 1, 2, 4] at minSamples = 2; the MRD MST that
  // results has expected total weight 1 + 1 + 2 + 4 = 8.
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

  PrimMstBackend<float> backend;
  MstOutput<float> out;
  backend.run(X, /*minSamples=*/2, Pool{}, out);

  ASSERT_EQ(out.edges.size(), 4u);
  ASSERT_EQ(out.coreDistances.dim(0), 5u);

  // Spanning-tree invariant: union-find over the edge set should collapse to a single component.
  UnionFind<std::uint32_t> uf(5);
  for (const auto &e : out.edges) {
    uf.unite(static_cast<std::uint32_t>(e.u), static_cast<std::uint32_t>(e.v));
  }
  const std::uint32_t root = uf.find(0);
  EXPECT_EQ(uf.componentSize(root), 5u);

  // Total weight agrees with the hand computation.
  float totalWeight = 0.0F;
  for (const auto &e : out.edges) {
    totalWeight += e.weight;
  }
  EXPECT_FLOAT_EQ(totalWeight, 8.0F);

  // Core distances agree with the hand computation within float epsilon.
  const std::vector<float> expectedCoreDists = {1.0F, 1.0F, 1.0F, 2.0F, 4.0F};
  for (std::size_t i = 0; i < 5; ++i) {
    EXPECT_FLOAT_EQ(out.coreDistances(i), expectedCoreDists[i]) << "i=" << i;
  }
}

// ---------------------------------------------------------------------------
// Reference-dataset MST edge weights agree with an offline scalar reference within 1e-5 absolute
// tolerance. The oracle builds the MRD matrix from the scalar squared-Euclidean table, then
// extracts the MST via Kruskal on a sorted edge list -- a completely independent MST algorithm so
// agreement checks the backend's numeric contract rather than its Prim-specific control flow.
// ---------------------------------------------------------------------------

namespace {

// Scalar-reference oracle producing the total MST weight under MRD weights built from the scalar
// pairwise-squared-distance table and the minSamples-th squared distance as core-distance. Uses
// Kruskal with union-find so it is algorithmically orthogonal to Prim.
float kruskalMrdTotalWeight(const NDArray<float, 2> &X, std::size_t minSamples) {
  const std::size_t n = X.dim(0);
  const auto sqD = scalarSqEuclidean(X);

  // Core distances from a sorted list of self-excluded squared distances per point.
  std::vector<float> coreDist(n, 0.0F);
  for (std::size_t i = 0; i < n; ++i) {
    std::vector<float> row;
    row.reserve(n - 1);
    for (std::size_t j = 0; j < n; ++j) {
      if (i == j) {
        continue;
      }
      row.push_back(sqD(i, j));
    }
    std::sort(row.begin(), row.end());
    coreDist[i] = row[minSamples - 1];
  }

  // Upper-triangular MRD edge list.
  struct Edge {
    float w;
    std::uint32_t u;
    std::uint32_t v;
  };
  std::vector<Edge> edges;
  edges.reserve(n * (n - 1) / 2);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = i + 1; j < n; ++j) {
      float w = sqD(i, j);
      w = std::max(coreDist[i], w);
      w = std::max(coreDist[j], w);
      edges.push_back({w, static_cast<std::uint32_t>(i), static_cast<std::uint32_t>(j)});
    }
  }
  std::sort(edges.begin(), edges.end(), [](const Edge &a, const Edge &b) { return a.w < b.w; });

  UnionFind<std::uint32_t> uf(n);
  float total = 0.0F;
  std::size_t taken = 0;
  for (const auto &e : edges) {
    if (uf.unite(e.u, e.v)) {
      total += e.w;
      ++taken;
      if (taken + 1 == n) {
        break;
      }
    }
  }
  return total;
}

} // namespace

TEST(PrimMstReference, TotalWeightAgreesWithKruskalOracle) {
  // Synthetic 40-point Gaussian in d = 2; small enough for the scalar Kruskal oracle to run in a
  // test but large enough that the MRD-MST has a non-trivial structure.
  const std::size_t n = 40;
  const std::size_t d = 2;
  const auto X = makeGaussianPoints(n, d, 0xBEEFCAFEULL);

  PrimMstBackend<float> backend;
  MstOutput<float> out;
  backend.run(X, /*minSamples=*/5, Pool{}, out);

  ASSERT_EQ(out.edges.size(), n - 1);

  float primTotal = 0.0F;
  for (const auto &e : out.edges) {
    primTotal += e.weight;
  }
  const float oracleTotal = kruskalMrdTotalWeight(X, /*minSamples=*/5);
  EXPECT_NEAR(primTotal, oracleTotal, 1e-5F * std::max(1.0F, std::abs(oracleTotal)));

  // Spanning-tree invariant on the edges.
  UnionFind<std::uint32_t> uf(n);
  for (const auto &e : out.edges) {
    uf.unite(static_cast<std::uint32_t>(e.u), static_cast<std::uint32_t>(e.v));
  }
  const std::uint32_t root = uf.find(0);
  EXPECT_EQ(uf.componentSize(root), n);
}

// ---------------------------------------------------------------------------
// Out-of-budget assertion: a shape whose MRD matrix exceeds kPrimMrdMatrixByteBudget fires
// CLUSTERING_ALWAYS_ASSERT before allocating. Death test: size chosen so n*n*sizeof(float)
// strictly exceeds the budget and no in-range n satisfies the comparison.
// ---------------------------------------------------------------------------

TEST(PrimMstBudgetDeath, OverBudgetShapeAborts) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  // 9000 * 9000 * 4 = 324 MiB > 256 MiB budget.
  constexpr std::size_t n = 9000;
  const NDArray<float, 2> X(std::array<std::size_t, 2>{n, 2});
  PrimMstBackend<float> backend;
  MstOutput<float> out;
  EXPECT_DEATH(
      { backend.run(X, /*minSamples=*/2, Pool{}, out); },
      "always-assert failed: n <= kNsqBudget / n");
}

// ---------------------------------------------------------------------------
// Post-run shape contract: edges.size() == n - 1 and coreDistances length is n.
// ---------------------------------------------------------------------------

TEST(PrimMstContract, PostRunShapesMatchContract) {
  // Small Gaussian to keep the test fast while still exercising the main pipeline.
  const std::size_t n = 30;
  const auto X = makeGaussianPoints(n, 4, 0x7E57CA5EULL);

  PrimMstBackend<float> backend;
  MstOutput<float> out;
  backend.run(X, /*minSamples=*/3, Pool{}, out);

  EXPECT_EQ(out.edges.size(), n - 1);
  EXPECT_EQ(out.coreDistances.dim(0), n);
}
