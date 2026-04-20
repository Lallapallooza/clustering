#include <gtest/gtest.h>

#include <cstddef>
#include <vector>

#include "clustering/hdbscan.h"
#include "clustering/math/thread.h"
#include "clustering/ndarray.h"

using clustering::HDBSCAN;
using clustering::NDArray;
using clustering::hdbscan::ClusterSelectionMethod;
using clustering::hdbscan::MstBackendStrategy;
using clustering::hdbscan::MstEdge;
using clustering::hdbscan::MstOutput;

namespace {

// Minimal concept-satisfying backend: default-constructible with a run(X, minSamples, pool, out)
// signature and empty body. Used to instantiate HDBSCAN in tests without building the real Prim
// backend.
template <class T> struct MockBackend {
  void run(const NDArray<T, 2> & /*X*/, std::size_t /*minSamples*/, clustering::math::Pool /*pool*/,
           MstOutput<T> & /*out*/) {}
};

// Deliberately malformed: no run() method at all. Must fail the concept.
template <class T> struct WrongBackendNoRun {
  static_assert(std::is_same_v<T, T>);
};

// Deliberately malformed: run() has the wrong signature.
template <class T> struct WrongBackendWrongSignature {
  // Signature mismatch: takes `int` instead of the expected NDArray/pool/out arguments.
  void run(int /*x*/) {}
};

using Hdb = clustering::HDBSCAN<float, MockBackend<float>>;

} // namespace

// ---------------------------------------------------------------------------
// Type defaults: MstEdge and MstOutput land at well-defined empty values.
// ---------------------------------------------------------------------------

TEST(HdbscanTypes, MstEdgeDefaults) {
  const MstEdge<float> e;
  EXPECT_EQ(e.u, 0);
  EXPECT_EQ(e.v, 0);
  EXPECT_EQ(e.weight, 0.0F);
}

TEST(HdbscanTypes, MstOutputDefaults) {
  const MstOutput<float> o;
  EXPECT_TRUE(o.edges.empty());
  EXPECT_EQ(o.coreDistances.dim(0), 0u);
}

// ---------------------------------------------------------------------------
// Concept: MstBackendStrategy accepts conforming types and rejects non-conforming.
// ---------------------------------------------------------------------------

static_assert(MstBackendStrategy<MockBackend<float>, float>,
              "MockBackend<float> must satisfy MstBackendStrategy<float>");
static_assert(!MstBackendStrategy<WrongBackendNoRun<float>, float>,
              "Type lacking run() must not satisfy the concept");
static_assert(!MstBackendStrategy<WrongBackendWrongSignature<float>, float>,
              "Type with wrong run() signature must not satisfy the concept");

// HDBSCAN<double, MockBackend<double>> must not compile because of the static_assert in the class
// body. A negative compile test requires a dedicated compile-fail harness; we document the
// invariant here and verify float-only at the type-parameter level through the instantiation
// tests below.

// ---------------------------------------------------------------------------
// Construction: valid parameters pass; invalid parameters assert.
// ---------------------------------------------------------------------------

TEST(HdbscanConstruction, ValidArgsConstruct) {
  const Hdb h(5);
  EXPECT_EQ(h.nClusters(), 0u);
  EXPECT_EQ(h.labels().dim(0), 0u);
  EXPECT_EQ(h.outlierScores().dim(0), 0u);
  EXPECT_TRUE(h.condensedTree().empty());
}

TEST(HdbscanConstruction, MethodAndMinSamplesPropagate) {
  // Construct with explicit method and minSamples; the invariants don't observably surface until
  // fit, but the ctor must accept and store them without firing an assertion.
  const Hdb h(5, 3, ClusterSelectionMethod::kLeaf);
  EXPECT_EQ(h.nClusters(), 0u);
}

TEST(HdbscanConstructionDeath, MinClusterSizeBelowTwoAborts) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_DEATH({ const Hdb h(1); }, "always-assert failed: minClusterSize >= 2");
  EXPECT_DEATH({ const Hdb h(0); }, "always-assert failed: minClusterSize >= 2");
}

// ---------------------------------------------------------------------------
// reset() returns the instance to fresh-constructed state.
// ---------------------------------------------------------------------------

TEST(HdbscanReset, RestoresEmptyState) {
  Hdb h(3);
  NDArray<float, 2> X({8, 2});
  for (std::size_t i = 0; i < 8; ++i) {
    X(i, 0) = static_cast<float>(i);
    X(i, 1) = static_cast<float>(i);
  }
  // Calling run() here just exercises the precondition path and leaves the result accessors
  // empty.
  h.run(X);
  h.reset();
  EXPECT_EQ(h.nClusters(), 0u);
  EXPECT_EQ(h.labels().dim(0), 0u);
  EXPECT_EQ(h.outlierScores().dim(0), 0u);
  EXPECT_TRUE(h.condensedTree().empty());
}

// ---------------------------------------------------------------------------
// Fit-entry preconditions: every precondition fires the always-assert before any work begins.
// ---------------------------------------------------------------------------

TEST(HdbscanFitDeath, MinSamplesGeOrEqualsNAborts) {
  // Resolved minSamples >= N: pass minSamples = 10 against an n = 5 dataset.
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  Hdb h(3, /*minSamples=*/10);
  const NDArray<float, 2> X({5, 2});
  EXPECT_DEATH(h.run(X), "always-assert failed: effectiveMinSamples < n");
}

TEST(HdbscanFitDeath, NLessThanMinClusterSizeAborts) {
  // N = 2, minClusterSize = 3: the N >= minClusterSize precondition fires.
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  Hdb h(3, /*minSamples=*/1);
  const NDArray<float, 2> X({2, 2});
  EXPECT_DEATH(h.run(X), "always-assert failed: n >= m_minClusterSize");
}

TEST(HdbscanFit, SentinelResolvedMinSamplesFromMinClusterSize) {
  // Sentinel resolution: minSamples = 0 at construction means "resolve to minClusterSize at fit
  // time." Since minClusterSize >= 2 (enforced in the ctor), the resolved value is always >= 2,
  // which in turn satisfies effectiveMinSamples >= 1. This test records that the sentinel path
  // cannot itself produce a resolved minSamples of 0 -- the construction gate already rules it
  // out.
  Hdb h(2, /*minSamples=*/0);
  const NDArray<float, 2> X({3, 2});
  // Must not die: sentinel resolves to minClusterSize == 2, which is valid (< n = 3).
  h.run(X);
  SUCCEED();
}
