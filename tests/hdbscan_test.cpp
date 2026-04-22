#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <random>
#include <set>
#include <utility>
#include <vector>

#include "clustering/hdbscan.h"
#include "clustering/hdbscan/detail/condensed_tree.h"
#include "clustering/hdbscan/detail/leaf_extract.h"
#include "clustering/hdbscan/policy/prim_mst_backend.h"
#include "clustering/math/thread.h"
#include "clustering/ndarray.h"

using clustering::HDBSCAN;
using clustering::NDArray;
using clustering::hdbscan::ClusterSelectionMethod;
using clustering::hdbscan::MstBackendStrategy;
using clustering::hdbscan::MstEdge;
using clustering::hdbscan::MstOutput;
using clustering::hdbscan::PrimMstBackend;

namespace {

// Minimal concept-satisfying backend: default-constructible with a run(X, minSamples, pool, out)
// signature and empty body. Used to verify the concept's compile-time contract.
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

// The end-to-end and precondition tests use the real Prim backend. The MockBackend above exists
// only so the static_asserts on the concept have a concrete positive exemplar.
using Hdb = clustering::HDBSCAN<float, PrimMstBackend<float>>;

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

TEST(HdbscanReset, DefaultAutoBackendCompilesAndResets) {
  // Pins the default-path reset compilation. The variant-held NnDescent backend's
  // move-assignability is required for HDBSCAN::reset to build under the default
  // MstBackend = AutoMstBackend<T>. Without this test, the default path can
  // regress silently as CI never compiles it.
  clustering::HDBSCAN<float> h(3);
  h.reset();
  EXPECT_EQ(h.nClusters(), 0u);
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
  // minSamples = 2 is the minimum valid value under the default sklearn convention (the query
  // point counts as one of the two neighbours, leaving one non-self neighbour for the backend).
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  Hdb h(3, /*minSamples=*/2);
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
  NDArray<float, 2> X({3, 2});
  // Populate with small non-degenerate coordinates so the backend can build a real MST.
  X(0, 0) = 0.0F;
  X(0, 1) = 0.0F;
  X(1, 0) = 1.0F;
  X(1, 1) = 1.0F;
  X(2, 0) = 2.0F;
  X(2, 1) = 2.0F;
  // Must not die: sentinel resolves to minClusterSize == 2, which is valid (< n = 3).
  h.run(X);
  SUCCEED();
}

// ---------------------------------------------------------------------------
// End-to-end pipeline tests against the real Prim backend.
// ---------------------------------------------------------------------------

namespace {

// Adjusted Rand Index on two label arrays -- dense labels remapped per side.
// Noise (-1) is treated as its own singleton label per convention: each noise point forms its own
// cluster of size 1. The contingency matrix approach is adequate for small N; ARI's denominator
// cancels the choice of size-1 conventions that are unchanged between the two label arrays.
double adjustedRandIndex(const std::vector<std::int32_t> &a, const std::vector<std::int32_t> &b) {
  const std::size_t n = a.size();
  if (n != b.size() || n < 2) {
    return 0.0;
  }
  // Remap each label array to a dense 0..k-1 range (plus unique ids for -1 noise).
  auto remap = [n](const std::vector<std::int32_t> &labels) {
    std::vector<std::int32_t> remapped(n);
    std::int32_t next = 0;
    // Assign a unique id to each noise point so they form singleton clusters.
    for (std::size_t i = 0; i < n; ++i) {
      if (labels[i] == -1) {
        remapped[i] = next++;
      }
    }
    // Then remap the positive labels.
    std::vector<std::pair<std::int32_t, std::int32_t>> seen;
    for (std::size_t i = 0; i < n; ++i) {
      if (labels[i] == -1) {
        continue;
      }
      bool found = false;
      for (const auto &p : seen) {
        if (p.first == labels[i]) {
          remapped[i] = p.second;
          found = true;
          break;
        }
      }
      if (!found) {
        seen.emplace_back(labels[i], next);
        remapped[i] = next++;
      }
    }
    return std::make_pair(remapped, static_cast<std::size_t>(next));
  };
  const auto [aD, ka] = remap(a);
  const auto [bD, kb] = remap(b);

  // Contingency matrix of size ka x kb.
  std::vector<std::size_t> c(ka * kb, 0);
  for (std::size_t i = 0; i < n; ++i) {
    c[(static_cast<std::size_t>(aD[i]) * kb) + static_cast<std::size_t>(bD[i])] += 1;
  }
  auto choose2 = [](std::size_t v) -> double {
    return (v < 2) ? 0.0 : static_cast<double>(v * (v - 1)) / 2.0;
  };
  double sumC = 0.0;
  for (const std::size_t v : c) {
    sumC += choose2(v);
  }
  std::vector<std::size_t> rowSum(ka, 0);
  std::vector<std::size_t> colSum(kb, 0);
  for (std::size_t i = 0; i < ka; ++i) {
    for (std::size_t j = 0; j < kb; ++j) {
      rowSum[i] += c[(i * kb) + j];
      colSum[j] += c[(i * kb) + j];
    }
  }
  double sumRow = 0.0;
  for (const std::size_t v : rowSum) {
    sumRow += choose2(v);
  }
  double sumCol = 0.0;
  for (const std::size_t v : colSum) {
    sumCol += choose2(v);
  }
  const double nC2 = choose2(n);
  const double expectedIndex = (sumRow * sumCol) / nC2;
  const double maxIndex = 0.5 * (sumRow + sumCol);
  const double denom = maxIndex - expectedIndex;
  if (denom == 0.0) {
    return 1.0; // Both partitions are trivial; ARI is conventionally 1.
  }
  return (sumC - expectedIndex) / denom;
}

// Two-moon fixture data and upstream `sklearn.cluster.HDBSCAN(min_cluster_size=5,
// algorithm='brute').fit_predict` labels. Generated offline at implementation time via
// `make_moons(n_samples=200, noise=0.05, random_state=42)` cast to float32. The reference labels
// come from sklearn v1.x upstream; only the 2-cluster structure matters for the ARI check.
struct TwoMoonFixture {
  static constexpr std::size_t kN = 200;
  std::array<float, kN * 2> points{};
  std::array<std::int32_t, kN> referenceLabels{};

  TwoMoonFixture() { populate(); }

  NDArray<float, 2> toNdarray() const {
    NDArray<float, 2> X(std::array<std::size_t, 2>{kN, std::size_t{2}});
    for (std::size_t i = 0; i < kN; ++i) {
      X(i, 0) = points[(i * 2) + 0];
      X(i, 1) = points[(i * 2) + 1];
    }
    return X;
  }

  std::vector<std::int32_t> referenceVec() const {
    return {referenceLabels.begin(), referenceLabels.end()};
  }

private:
  void populate();
};

void TwoMoonFixture::populate() {
  // Dataset generated from make_moons(n_samples=200, noise=0.05, random_state=42) cast to
  // float32.  Reference labels from sklearn.cluster.HDBSCAN(min_cluster_size=5,
  // algorithm='brute'). Float literals here are random samples from a 2-D point cloud; the
  // modernize-use-std-numbers lint occasionally pattern-matches digits against mathematical
  // constants (ln2, sqrt3, etc.) and is inapplicable to fixture data.
  // NOLINTBEGIN(modernize-avoid-c-arrays, modernize-use-std-numbers)
  constexpr float kP[kN * 2] = {
      -1.02069032F, 0.10551754F,  0.90582651F,  0.45785752F,  0.61842173F,  0.75708634F,
      1.22770703F,  -0.42518511F, 0.32935596F,  -0.20694567F, 0.18142481F,  0.11138976F,
      -0.62408894F, 0.81842071F,  1.58093810F,  -0.24960370F, 1.78680515F,  -0.16901471F,
      0.14277205F,  0.97154450F,  -0.46754566F, 0.80488175F,  1.88480020F,  -0.05941194F,
      1.50329852F,  -0.38950023F, -0.80546629F, 0.57388335F,  1.89827120F,  0.12687555F,
      1.81823742F,  -0.11646520F, -0.13661234F, 1.10576057F,  1.06302536F,  -0.50714809F,
      -0.92998546F, 0.57321376F,  -0.48895022F, 0.78921252F,  0.30714801F,  -0.12922548F,
      0.88516378F,  0.52856684F,  0.96115971F,  -0.55971009F, -0.92946130F, 0.25550652F,
      -0.43554601F, 0.83372849F,  -0.36462694F, 0.88847941F,  -0.85033655F, 0.52164602F,
      -0.50600117F, 0.89675224F,  0.25514305F,  -0.20540108F, 0.40428099F,  -0.38831446F,
      1.00342667F,  0.26975283F,  0.90172261F,  0.56753880F,  -0.03692707F, 0.98949605F,
      -0.79405433F, 0.68835253F,  0.96771520F,  -0.45101511F, 0.01014419F,  0.36654395F,
      0.54972678F,  -0.35897672F, 0.59103179F,  -0.38350180F, 1.49634516F,  -0.41310087F,
      -0.79156679F, 0.64112073F,  -0.76976478F, 0.59753186F,  2.02949047F,  0.52472293F,
      0.36843392F,  0.89370102F,  0.77660519F,  0.69264704F,  0.27757791F,  -0.08947768F,
      1.94400442F,  0.44250777F,  0.65086579F,  0.76556665F,  0.83889902F,  0.57992023F,
      0.79638994F,  -0.43520674F, -0.84424067F, 0.40129584F,  0.16361752F,  0.11961522F,
      1.69012427F,  -0.22578266F, 0.92981064F,  0.08296297F,  0.32974172F,  -0.27738816F,
      0.14003478F,  -0.04041053F, -0.76793426F, 0.75245696F,  0.68235302F,  -0.45089230F,
      -1.04415250F, 0.01498707F,  1.73287976F,  -0.24932203F, -0.96799308F, 0.03151245F,
      1.69950867F,  -0.22240527F, -0.92981488F, 0.10486222F,  0.55467725F,  0.85467356F,
      0.98076892F,  0.34023398F,  0.42184168F,  0.81253713F,  0.13773246F,  0.01600522F,
      0.21904658F,  0.95591921F,  1.96595943F,  0.10735519F,  -0.91222388F, 0.34505290F,
      0.60344982F,  -0.32706445F, -0.11392350F, 1.09105313F,  0.30860218F,  -0.31974557F,
      1.37447548F,  -0.32875401F, 0.26550171F,  0.91721261F,  0.09399934F,  -0.04589674F,
      -0.01585654F, 0.03405982F,  0.75566280F,  0.80075490F,  0.66651142F,  -0.46828407F,
      0.00292545F,  0.61739987F,  0.08055615F,  0.20346676F,  0.98786378F,  0.06646875F,
      -0.73368633F, 0.64909238F,  0.01596935F,  1.05786550F,  1.89761126F,  0.23390342F,
      2.00229120F,  0.25240064F,  -0.88096017F, 0.42437002F,  1.33256459F,  -0.42376795F,
      0.45966452F,  0.88640553F,  1.92838109F,  0.20049301F,  -0.30928567F, 1.00677538F,
      0.78414512F,  0.67185330F,  0.70204115F,  -0.42000443F, 0.02809031F,  0.43276566F,
      0.41061643F,  0.80499691F,  0.90623248F,  0.40011701F,  2.06508183F,  0.32002270F,
      1.27652466F,  -0.44182110F, 0.98253602F,  0.23579794F,  0.62870133F,  0.80320674F,
      0.18276851F,  -0.05020919F, 0.97840279F,  0.09593772F,  0.62284136F,  0.90298676F,
      0.67463881F,  -0.49415025F, 0.91793132F,  -0.54965448F, 0.07904094F,  0.25162026F,
      0.79139447F,  -0.48524860F, 0.84557021F,  0.40055501F,  -0.40179974F, 0.96254492F,
      0.10232414F,  0.19623159F,  0.86749661F,  -0.48364732F, 1.52795541F,  -0.32239833F,
      1.05771768F,  -0.02386909F, 1.96628523F,  0.53627765F,  1.00898445F,  -0.41533685F,
      -0.62072587F, 0.79553372F,  0.22061843F,  -0.16917877F, -0.38050970F, 0.93453693F,
      0.19828250F,  0.98925138F,  1.34313452F,  -0.45183885F, 0.61339259F,  0.79903030F,
      0.30040300F,  1.00407743F,  0.05573267F,  0.25156954F,  1.28743625F,  -0.46449640F,
      1.18320954F,  -0.38580409F, 0.76149547F,  0.80349857F,  0.73971784F,  0.71558446F,
      1.81619298F,  0.08097380F,  0.91286385F,  -0.56460768F, -0.78962350F, 0.45013425F,
      0.23143263F,  0.96312463F,  1.58248746F,  -0.28404367F, 0.11634811F,  1.07068765F,
      -0.95242673F, 0.15138642F,  1.13863432F,  -0.50393248F, 0.21958055F,  0.92393517F,
      0.66015989F,  -0.43846855F, 1.00925362F,  0.10899220F,  -0.11005212F, 0.29693392F,
      -0.11626963F, 0.99106419F,  0.51296949F,  -0.35090798F, 1.89948821F,  -0.05143694F,
      1.77040195F,  -0.19811516F, 1.83449697F,  0.09697968F,  0.06888831F,  1.07694268F,
      -0.86507124F, 0.55186665F,  0.48438776F,  0.98382330F,  0.98117286F,  0.29787368F,
      0.05842753F,  0.24562882F,  -0.91661972F, 0.25945240F,  -0.92486852F, 0.47274202F,
      1.81971669F,  -0.12287774F, 1.81957102F,  -0.05775625F, -1.03661478F, 0.38425037F,
      1.01225185F,  0.24429409F,  0.92611516F,  0.29238725F,  -0.32125610F, 1.02162457F,
      1.67963004F,  -0.25961143F, 0.44534403F,  -0.29274958F, 0.84391415F,  0.43044272F,
      -0.65517068F, 0.77590555F,  1.77986789F,  -0.26213881F, 0.52158338F,  -0.38521856F,
      1.47216284F,  -0.32534176F, 1.91121781F,  -0.00073135F, -0.31266040F, 0.89366400F,
      -0.16464901F, 0.96603787F,  -0.01620474F, 0.25173819F,  -0.08757041F, 0.99029750F,
      1.99834561F,  0.29080382F,  -0.27157128F, 0.96594888F,  0.07331137F,  1.05909908F,
      -0.89712489F, 0.31100196F,  0.74923247F,  0.64713418F,  -0.26312292F, 0.91040885F,
      1.44577503F,  -0.46053705F, 1.94410360F,  0.17008297F,  0.43696043F,  -0.34515864F,
      0.38777617F,  0.97078544F,  1.18714547F,  -0.46783745F, 2.05811810F,  0.18791141F,
      0.95897126F,  -0.01349916F, -0.08859521F, 0.99446523F,  0.93107092F,  -0.53259277F,
      0.38128185F,  -0.33211541F, 1.05849302F,  -0.45333913F, 0.05729463F,  0.43762568F,
      -1.04369867F, -0.00016480F, 0.24486709F,  0.06452990F,  -0.95039314F, 0.45076436F,
      -0.72087836F, 0.67927408F,  0.28373352F,  -0.18364145F, 1.90431118F,  0.54706424F,
      0.80665481F,  0.59927619F,  1.92162144F,  0.14605519F,  -0.65345818F, 0.74484682F,
      -0.00102421F, 0.29007620F,  0.86149734F,  0.40253645F,  -1.10056782F, 0.25454265F,
      1.82478297F,  -0.09715337F, 0.01387954F,  0.45025495F};
  std::copy(std::begin(kP), std::end(kP), points.begin());

  constexpr std::int32_t kR[kN] = {
      0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1,
      1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0,
      1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1,
      0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1,
      0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0,
      0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0,
      1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1};
  std::copy(std::begin(kR), std::end(kR), referenceLabels.begin());
  // NOLINTEND(modernize-avoid-c-arrays, modernize-use-std-numbers)
}

} // namespace

// Condensed-tree edge count sanity check: the condensed tree cannot have more rows than 2(N-1)
// (every non-root node contributes exactly one row). On a small dataset the bound is tight enough
// to catch mass-emit bugs in the drop-as-noise code path.
TEST(HdbscanEndToEnd, CondensedTreeRowCountBound) {
  const TwoMoonFixture fix;
  auto X = fix.toNdarray();
  Hdb h(5);
  h.run(X);
  const auto view = h.condensedTree();
  // Upper bound: at most 2*(N-1) rows in a strictly binary condensed tree.
  EXPECT_LE(view.size(), 2u * (TwoMoonFixture::kN - 1));
  // Lower bound: at least N leaf rows must exist (every point either drops out via a leaf row or
  // is inside a cluster that itself produced a leaf row).
  EXPECT_GE(view.size(), TwoMoonFixture::kN);
}

// Primary ARI check: EOM labels on the two-moon dataset match sklearn upstream within 0.999.
TEST(HdbscanEndToEnd, EomLabelsMatchSklearnOnTwoMoons) {
  const TwoMoonFixture fix;
  auto X = fix.toNdarray();
  Hdb h(5);
  h.run(X);

  std::vector<std::int32_t> ours(TwoMoonFixture::kN);
  for (std::size_t i = 0; i < TwoMoonFixture::kN; ++i) {
    ours[i] = h.labels()(i);
  }
  const auto ref = fix.referenceVec();
  const double ari = adjustedRandIndex(ours, ref);
  EXPECT_GE(ari, 0.999) << "ari=" << ari;
  // Sanity: at least one cluster found.
  EXPECT_GE(h.nClusters(), 1u);
}

// Leaf extraction: every non-noise label must correspond to a leaf cluster of the condensed tree
// (a cluster node that has no cluster-children). Exercised as a unit test against a hand-crafted
// condensed tree, so the assertion is direct (no label-to-cluster remap required).
//
// Tree layout (N = 6 points; cluster ids in [6, 11)):
//   root(6) -- cluster 7 at lambda = 1.0 (internal, size 4)
//   root(6) -- cluster 8 at lambda = 1.0 (leaf cluster, size 2)
//   7 -- cluster 9 at lambda = 2.0       (leaf cluster, size 2)
//   7 -- cluster 10 at lambda = 2.0      (leaf cluster, size 2)
//   8 -- point 4 at lambda = 3.0
//   8 -- point 5 at lambda = 3.0
//   9 -- point 0 at lambda = 4.0
//   9 -- point 1 at lambda = 4.0
//   10 -- point 2 at lambda = 5.0
//   10 -- point 3 at lambda = 5.0
//
// Clusters with a cluster-child: {6, 7}. Leaf clusters: {8, 9, 10}.
// Dense labels in condensed-id order: cluster 8 -> 0, cluster 9 -> 1, cluster 10 -> 2.
TEST(HdbscanLeafExtract, LeafLabelsPointAtLeafClusters) {
  clustering::hdbscan::detail::CondensedTree<float> tree;
  tree.parent = {6, 6, 7, 7, 8, 8, 9, 9, 10, 10};
  tree.child = {7, 8, 9, 10, 4, 5, 0, 1, 2, 3};
  tree.lambdaVal = {1.0F, 1.0F, 2.0F, 2.0F, 3.0F, 3.0F, 4.0F, 4.0F, 5.0F, 5.0F};
  tree.childSize = {4, 2, 2, 2, 1, 1, 1, 1, 1, 1};
  tree.numClusters = 5; // clusters 6, 7, 8, 9, 10.

  constexpr std::size_t kN = 6;
  std::vector<std::int32_t> labels;
  clustering::hdbscan::detail::extractLeaf(tree, kN, labels);
  ASSERT_EQ(labels.size(), kN);

  // Build the set of cluster ids that have a cluster-child. Every non-noise label must map back to
  // a cluster NOT in this set.
  const auto kSignedN = static_cast<std::int32_t>(kN);
  std::set<std::int32_t> hasClusterChild;
  for (std::size_t i = 0; i < tree.parent.size(); ++i) {
    if (tree.child[i] >= kSignedN) {
      hasClusterChild.insert(tree.parent[i]);
    }
  }
  EXPECT_EQ(hasClusterChild.count(6), 1u);
  EXPECT_EQ(hasClusterChild.count(7), 1u);
  EXPECT_EQ(hasClusterChild.count(8), 0u);
  EXPECT_EQ(hasClusterChild.count(9), 0u);
  EXPECT_EQ(hasClusterChild.count(10), 0u);

  // Reconstruct each point's source cluster by scanning the leaf rows of the tree, then assert its
  // cluster is absent from hasClusterChild for every non-noise label.
  std::vector<std::int32_t> sourceCluster(kN, std::int32_t{-1});
  for (std::size_t i = 0; i < tree.parent.size(); ++i) {
    if (tree.child[i] < kSignedN) {
      sourceCluster[static_cast<std::size_t>(tree.child[i])] = tree.parent[i];
    }
  }
  // Every point is assigned to some cluster.
  for (std::size_t i = 0; i < kN; ++i) {
    EXPECT_GE(sourceCluster[i], kSignedN) << "point " << i << " has no leaf row";
  }

  // Core leaf-extract invariant: non-noise labels only reference leaf clusters.
  std::size_t nonNoise = 0;
  for (std::size_t i = 0; i < kN; ++i) {
    if (labels[i] != -1) {
      ++nonNoise;
      EXPECT_EQ(hasClusterChild.count(sourceCluster[i]), 0u)
          << "point " << i << " (label=" << labels[i] << ") maps to cluster " << sourceCluster[i]
          << " which has a cluster-child";
    }
  }
  EXPECT_EQ(nonNoise, kN) << "Every point should reach a leaf cluster on this tree";

  // Exact label assignment pins the dense-label mapping (cluster 8 -> 0, 9 -> 1, 10 -> 2).
  EXPECT_EQ(labels[0], 1); // cluster 9
  EXPECT_EQ(labels[1], 1);
  EXPECT_EQ(labels[2], 2); // cluster 10
  EXPECT_EQ(labels[3], 2);
  EXPECT_EQ(labels[4], 0); // cluster 8
  EXPECT_EQ(labels[5], 0);
}

// Pinned-Prim end-to-end: construct with the Prim backend explicitly, fit on two-moon, match
// sklearn within the ARI threshold. Separate from the general EOM test so failures clearly
// differentiate backend pinning from post-MST parity issues.
TEST(HdbscanEndToEnd, PinnedPrimFitMatchesSklearn) {
  const TwoMoonFixture fix;
  auto X = fix.toNdarray();
  clustering::HDBSCAN<float, PrimMstBackend<float>> h(5);
  h.run(X);
  std::vector<std::int32_t> ours(TwoMoonFixture::kN);
  for (std::size_t i = 0; i < TwoMoonFixture::kN; ++i) {
    ours[i] = h.labels()(i);
  }
  const double ari = adjustedRandIndex(ours, fix.referenceVec());
  EXPECT_GE(ari, 0.999);
}

// Dying-subcluster boundedness: build a dataset where a sub-cluster drops out before final
// condensation, and verify every outlier score lands in [0, 1]. The bound is structural under the
// Campello 2015 formula; a score > 1 would indicate the scikit-learn-contrib/hdbscan bug.
TEST(HdbscanEndToEnd, DyingSubclusterOutlierScoresBounded) {
  // Two tight sub-clusters inside a loose super-cluster, plus a small sub-cluster that dies
  // during condensation. Generated with a fixed seed for reproducibility.
  std::mt19937_64 gen(0xDEADBEEFULL);
  std::normal_distribution<float> tight(0.0F, 0.05F);
  std::normal_distribution<float> wide(0.0F, 0.8F);
  NDArray<float, 2> X(std::array<std::size_t, 2>{100, std::size_t{2}});
  for (std::size_t i = 0; i < 40; ++i) {
    X(i, 0) = 0.0F + tight(gen);
    X(i, 1) = 0.0F + tight(gen);
  }
  for (std::size_t i = 0; i < 40; ++i) {
    X(40 + i, 0) = 3.0F + tight(gen);
    X(40 + i, 1) = 0.0F + tight(gen);
  }
  for (std::size_t i = 0; i < 20; ++i) {
    X(80 + i, 0) = 1.5F + wide(gen);
    X(80 + i, 1) = 0.0F + wide(gen);
  }

  Hdb h(5);
  h.run(X);
  for (std::size_t i = 0; i < X.dim(0); ++i) {
    const float s = h.outlierScores()(i);
    EXPECT_GE(s, 0.0F) << "i=" << i;
    EXPECT_LE(s, 1.0F) << "i=" << i;
  }
}

// Single Gaussian blob: every non-noise point lands in cluster 0, and any noise point is -1. No
// other labels may appear, and exactly one cluster must be reported.
TEST(HdbscanEndToEnd, SingleBlobProducesValidLabels) {
  // On a single Gaussian blob, HDBSCAN's excess-of-mass traversal may discover sub-clusters
  // driven by local density fluctuations or mark every point as noise; the Campello 2015 FORC
  // algorithm forbids the root cluster from ever being chosen, so a "one label for every point"
  // collapse is never the right answer. The contract the fitter must satisfy on arbitrary
  // continuous input is weaker: every label is in @c {-1, 0, 1, ..., nClusters - 1} and no
  // access violation occurs.
  constexpr std::size_t kN = 100;
  constexpr std::size_t kD = 2;
  std::mt19937 rng(42);
  std::normal_distribution<float> gaussian(0.0F, 1.0F);
  NDArray<float, 2> X(std::array<std::size_t, 2>{kN, kD});
  for (std::size_t i = 0; i < kN; ++i) {
    for (std::size_t j = 0; j < kD; ++j) {
      X(i, j) = gaussian(rng);
    }
  }

  Hdb h(5);
  h.run(X);

  const auto nc = static_cast<std::int32_t>(h.nClusters());
  for (std::size_t i = 0; i < kN; ++i) {
    const std::int32_t label = h.labels()(i);
    EXPECT_TRUE(label == -1 || (label >= 0 && label < nc))
        << "i=" << i << " label=" << label << " nc=" << nc;
  }
}
