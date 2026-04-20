#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <random>
#include <unordered_map>
#include <vector>

#include "clustering/hdbscan/mst_backend.h"
#include "clustering/hdbscan/policy/nn_descent_mst_backend.h"
#include "clustering/hdbscan/policy/prim_mst_backend.h"
#include "clustering/math/dsu.h"
#include "clustering/math/thread.h"
#include "clustering/ndarray.h"

using clustering::NDArray;
using clustering::UnionFind;
using clustering::hdbscan::MstBackendStrategy;
using clustering::hdbscan::MstEdge;
using clustering::hdbscan::MstOutput;
using clustering::hdbscan::NnDescentMstBackend;
using clustering::hdbscan::NnDescentMstConfig;
using clustering::hdbscan::PrimMstBackend;
using clustering::math::Pool;

// The backend must satisfy the MstBackendStrategy concept; compile-time gate mirrors the
// Prim-backend contract HDBSCAN's MstBackend template parameter relies on.
static_assert(MstBackendStrategy<NnDescentMstBackend<float>, float>,
              "NnDescentMstBackend<float> must satisfy MstBackendStrategy<float>");

namespace {

// Deterministic Gaussian point cloud; seed controls shape so each test has stable state.
NDArray<float, 2> makeGaussian(std::size_t n, std::size_t d, std::uint64_t seed) {
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

// Build a spatially-separated multi-blob dataset that defeats kNN-graph connectivity at small k:
// c blobs each containing blobSize points, centers laid out far apart along the first axis so the
// gap between clusters exceeds any in-cluster distance. At minSamples small and kExtra small the
// kNN-graph never bridges the gap, which is exactly the adversarial case the fallback targets.
NDArray<float, 2> makeSeparatedBlobs(std::size_t c, std::size_t blobSize, std::size_t d,
                                     std::uint64_t seed, float centerSpacing) {
  const std::size_t n = c * blobSize;
  NDArray<float, 2> X(std::array<std::size_t, 2>{n, d});
  std::mt19937_64 gen(seed);
  std::normal_distribution<float> dist(0.0F, 0.3F);
  for (std::size_t b = 0; b < c; ++b) {
    const float cx = static_cast<float>(b) * centerSpacing;
    for (std::size_t p = 0; p < blobSize; ++p) {
      const std::size_t row = (b * blobSize) + p;
      for (std::size_t j = 0; j < d; ++j) {
        const float jitter = dist(gen);
        X(row, j) = (j == 0 ? cx + jitter : jitter);
      }
    }
  }
  return X;
}

// Count connected components of an undirected graph induced by a set of MST edges over n nodes.
// Used to verify the spanning-tree contract end-to-end and to sanity-check the adversarial
// fallback: a true spanning tree has exactly one component.
std::size_t countGraphComponents(std::size_t n, const std::vector<MstEdge<float>> &edges) {
  UnionFind<std::uint32_t> uf(n);
  for (const auto &e : edges) {
    uf.unite(static_cast<std::uint32_t>(e.u), static_cast<std::uint32_t>(e.v));
  }
  return uf.countComponents();
}

// Cut an MST into k clusters by removing the top k-1 highest-weight edges and running
// union-find over the remainder. Mirrors single-linkage clustering at a fixed cluster count;
// chosen as the ARI comparator because it is agnostic to absolute edge weights - both backends
// produce MRD-MSTs of the same data so relative edge orderings align for the edges that matter.
std::vector<std::int32_t> mstToLabels(std::size_t n, std::vector<MstEdge<float>> edges,
                                      std::size_t k) {
  // Sort descending by weight; the first k-1 edges are the cuts that produce k clusters.
  std::sort(edges.begin(), edges.end(),
            [](const MstEdge<float> &a, const MstEdge<float> &b) { return a.weight > b.weight; });
  UnionFind<std::uint32_t> uf(n);
  // Skip the first k-1 edges; unite through the rest.
  for (std::size_t i = k - 1; i < edges.size(); ++i) {
    uf.unite(static_cast<std::uint32_t>(edges[i].u), static_cast<std::uint32_t>(edges[i].v));
  }
  // Canonicalize root -> contiguous label id so both backends produce labels comparable by ARI
  // rather than by root-id coincidence.
  std::vector<std::int32_t> labels(n, -1);
  std::unordered_map<std::uint32_t, std::int32_t> rootToLabel;
  std::int32_t nextLabel = 0;
  for (std::uint32_t i = 0; i < static_cast<std::uint32_t>(n); ++i) {
    const std::uint32_t r = uf.find(i);
    auto it = rootToLabel.find(r);
    if (it == rootToLabel.end()) {
      it = rootToLabel.emplace(r, nextLabel++).first;
    }
    labels[i] = it->second;
  }
  return labels;
}

// Adjusted Rand Index between two integer label arrays of the same length. Uses the contingency-
// table form with combinations-of-2 (nC2); returns 1.0 for identical partitions (up to label
// permutation) and 0.0 in expectation for independent random partitions.
double adjustedRandIndex(const std::vector<std::int32_t> &a, const std::vector<std::int32_t> &b) {
  const std::size_t n = a.size();
  if (b.size() != n) {
    return 0.0;
  }
  // Map labels to contiguous ids.
  std::unordered_map<std::int32_t, std::int32_t> mapA;
  std::unordered_map<std::int32_t, std::int32_t> mapB;
  for (std::size_t i = 0; i < n; ++i) {
    mapA.try_emplace(a[i], static_cast<std::int32_t>(mapA.size()));
    mapB.try_emplace(b[i], static_cast<std::int32_t>(mapB.size()));
  }
  const std::size_t ka = mapA.size();
  const std::size_t kb = mapB.size();
  std::vector<std::vector<std::uint64_t>> ct(ka, std::vector<std::uint64_t>(kb, 0));
  std::vector<std::uint64_t> rowSums(ka, 0);
  std::vector<std::uint64_t> colSums(kb, 0);
  for (std::size_t i = 0; i < n; ++i) {
    const auto ai = static_cast<std::size_t>(mapA[a[i]]);
    const auto bi = static_cast<std::size_t>(mapB[b[i]]);
    ++ct[ai][bi];
    ++rowSums[ai];
    ++colSums[bi];
  }
  auto choose2 = [](std::uint64_t v) {
    return v < 2 ? 0.0 : static_cast<double>(v) * static_cast<double>(v - 1) / 2.0;
  };
  double sumCt = 0.0;
  for (std::size_t i = 0; i < ka; ++i) {
    for (std::size_t j = 0; j < kb; ++j) {
      sumCt += choose2(ct[i][j]);
    }
  }
  double sumA = 0.0;
  for (std::size_t i = 0; i < ka; ++i) {
    sumA += choose2(rowSums[i]);
  }
  double sumB = 0.0;
  for (std::size_t j = 0; j < kb; ++j) {
    sumB += choose2(colSums[j]);
  }
  const double totalPairs = choose2(static_cast<std::uint64_t>(n));
  if (totalPairs == 0.0) {
    return 1.0;
  }
  const double expected = sumA * sumB / totalPairs;
  const double maxIndex = 0.5 * (sumA + sumB);
  if (maxIndex == expected) {
    return 1.0;
  }
  return (sumCt - expected) / (maxIndex - expected);
}

} // namespace

// ---------------------------------------------------------------------------
// Post-run shape contract: edges.size() == n - 1 and coreDistances length is n. Applies on every
// successful fit regardless of backend parameters.
// ---------------------------------------------------------------------------

TEST(NnDescentMstContract, PostRunShapesMatchContract) {
  const std::size_t n = 200;
  const std::size_t d = 16;
  const auto X = makeGaussian(n, d, 0x7E57CA5E01ULL);

  NnDescentMstBackend<float> backend;
  MstOutput<float> out;
  backend.run(X, /*minSamples=*/5, Pool{}, out);

  EXPECT_EQ(out.edges.size(), n - 1);
  EXPECT_EQ(out.coreDistances.dim(0), n);
  EXPECT_EQ(countGraphComponents(n, out.edges), 1u);
}

// ---------------------------------------------------------------------------
// End-to-end at the target workload shape: the backend runs to completion, produces a spanning
// tree, and the core-distance array is sized to n. Label-quality checks at this size are the
// ARI-vs-Prim criterion at smaller shapes; this test pins the ability to run at scale.
// ---------------------------------------------------------------------------

TEST(NnDescentMstEndToEnd, CompletesAtModerateNHighD) {
  // Backend completes on a high-d input and produces a spanning tree with the contracted shape.
  // Full-scale integration at the rfc's target ceiling (N=250k, d up to 300) is exercised by the
  // pybench hdbscan recipe, which compares head-to-head against sklearn.cluster.HDBSCAN; ctest
  // keeps the shape small enough for per-commit runs.
  const std::size_t n = 2500;
  const std::size_t d = 128;
  const auto X = makeGaussian(n, d, 0xA133A133ULL);

  NnDescentMstBackend<float> backend;
  MstOutput<float> out;
  backend.run(X, /*minSamples=*/5, Pool{}, out);

  ASSERT_EQ(out.edges.size(), n - 1);
  ASSERT_EQ(out.coreDistances.dim(0), n);
  EXPECT_EQ(countGraphComponents(n, out.edges), 1u);
}

// ---------------------------------------------------------------------------
// ARI vs Prim at d in {64, 128, 300}. On a shape small enough that Prim is tractable (the
// (n x n) MRD matrix at float sits well within the Prim budget), the approximate backend's
// single-linkage-k partition should align with Prim's at ARI >= 0.98. Labels come from a
// single-linkage cut at a fixed cluster count (5) as a reasonable number of coarse clusters
// on a Gaussian input; the metric compares partitions induced by the MST structure itself.
// ---------------------------------------------------------------------------

namespace {

void runAriVsPrim(std::size_t d, std::uint64_t seed, std::size_t minSamples) {
  const std::size_t n = 1500;
  const auto X = makeGaussian(n, d, seed);

  PrimMstBackend<float> prim;
  MstOutput<float> primOut;
  prim.run(X, minSamples, Pool{}, primOut);
  ASSERT_EQ(primOut.edges.size(), n - 1);
  EXPECT_EQ(countGraphComponents(n, primOut.edges), 1u);

  NnDescentMstBackend<float> nnd;
  MstOutput<float> nndOut;
  nnd.run(X, minSamples, Pool{}, nndOut);
  ASSERT_EQ(nndOut.edges.size(), n - 1);
  EXPECT_EQ(countGraphComponents(n, nndOut.edges), 1u);

  const std::size_t kClusters = 5;
  const auto primLabels = mstToLabels(n, primOut.edges, kClusters);
  const auto nndLabels = mstToLabels(n, nndOut.edges, kClusters);
  const double ari = adjustedRandIndex(primLabels, nndLabels);
  EXPECT_GE(ari, 0.98) << "d=" << d << " ARI=" << ari;
}

} // namespace

TEST(NnDescentMstAri, MatchesPrimAtD64) { runAriVsPrim(64, 0xA700D064ULL, /*minSamples=*/5); }

TEST(NnDescentMstAri, MatchesPrimAtD128) { runAriVsPrim(128, 0xA700D128ULL, /*minSamples=*/5); }

TEST(NnDescentMstAri, MatchesPrimAtD300) { runAriVsPrim(300, 0xA700D300ULL, /*minSamples=*/5); }

// ---------------------------------------------------------------------------
// Adversarial disconnected input at c in {2, 3, 5}: a multi-blob dataset with centers far enough
// apart that the kNN graph at small k cannot bridge the gaps. The backend's fallback must
// enumerate inter-component bridges and insert exactly c - 1 of them so the final output is a
// single spanning tree with n - 1 edges.
//
// The test forces the fallback path by shrinking kExtra to 0 and using a small minSamples so the
// per-point k = minSamples only covers in-blob neighbours. The center spacing is chosen so any
// cross-blob squared distance exceeds every in-blob distance by a comfortable margin.
// ---------------------------------------------------------------------------

namespace {

void runAdversarialDisconnected(std::size_t c) {
  const std::size_t blobSize = 50;
  const std::size_t d = 8;
  const std::size_t n = c * blobSize;
  const auto X = makeSeparatedBlobs(c, blobSize, d, 0xADBC0FEEULL, /*centerSpacing=*/50.0F);

  // Shrink kExtra to 0 and pick a small minSamples so the kNN graph cannot span the blob gap.
  NnDescentMstConfig cfg;
  cfg.kExtra = 0;
  cfg.seed = 0xADBC0FEEULL;
  NnDescentMstBackend<float> backend(cfg);
  MstOutput<float> out;
  backend.run(X, /*minSamples=*/4, Pool{}, out);

  // The spanning-tree invariant must hold regardless of how many components the kNN graph had:
  // the fallback inserts c - 1 bridges, producing a single spanning tree with n - 1 edges.
  ASSERT_EQ(out.edges.size(), n - 1);
  ASSERT_EQ(out.coreDistances.dim(0), n);
  EXPECT_EQ(countGraphComponents(n, out.edges), 1u);
}

} // namespace

TEST(NnDescentMstFallback, SpansTwoComponents) { runAdversarialDisconnected(2); }

TEST(NnDescentMstFallback, SpansThreeComponents) { runAdversarialDisconnected(3); }

TEST(NnDescentMstFallback, SpansFiveComponents) { runAdversarialDisconnected(5); }

// ---------------------------------------------------------------------------
// Reuse across repeat runs: a single backend instance fitting the same data twice must produce a
// spanning tree both times. The internal optional<NnDescentIndex> warm-starts when (pointer,
// shape, k) match; the MST pipeline does not depend on warm-start behaviour for correctness but
// this test pins the "call run twice on the same data" path the downstream HDBSCAN class hits
// under hyperparameter sweeps.
// ---------------------------------------------------------------------------

TEST(NnDescentMstReuse, RepeatRunsProduceSpanningTrees) {
  const std::size_t n = 200;
  const std::size_t d = 16;
  const auto X = makeGaussian(n, d, 0xABC1234ULL);

  NnDescentMstBackend<float> backend;
  MstOutput<float> out1;
  backend.run(X, /*minSamples=*/5, Pool{}, out1);
  ASSERT_EQ(out1.edges.size(), n - 1);
  ASSERT_EQ(countGraphComponents(n, out1.edges), 1u);

  MstOutput<float> out2;
  backend.run(X, /*minSamples=*/5, Pool{}, out2);
  ASSERT_EQ(out2.edges.size(), n - 1);
  ASSERT_EQ(countGraphComponents(n, out2.edges), 1u);
}
