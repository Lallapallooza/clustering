#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <random>
#include <unordered_map>
#include <vector>

#include "clustering/hdbscan.h"
#include "clustering/hdbscan/mst_backend.h"
#include "clustering/hdbscan/mst_output.h"
#include "clustering/hdbscan/policy/auto_mst_backend.h"
#include "clustering/hdbscan/policy/prim_mst_backend.h"
#include "clustering/math/thread.h"
#include "clustering/ndarray.h"

using clustering::HDBSCAN;
using clustering::NDArray;
using clustering::hdbscan::AutoMstBackend;
using clustering::hdbscan::MstBackendStrategy;
using clustering::hdbscan::MstOutput;
using clustering::hdbscan::PrimMstBackend;
using clustering::math::Pool;

// Concept satisfaction: the dispatcher must itself satisfy @c MstBackendStrategy so
// @c HDBSCAN<T, AutoMstBackend<T>> compiles and the default-template-argument path is reachable.
static_assert(MstBackendStrategy<AutoMstBackend<float>, float>,
              "AutoMstBackend<float> must satisfy MstBackendStrategy<float>");

namespace {

// Dense Gaussian point cloud with a caller-chosen seed; used for all non-two-moon shapes below.
// Correctness of the dispatcher is about variant arm selection and pipeline shape, not about the
// underlying distribution, so a single source of random shape suffices.
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

// Adjusted Rand Index on two label arrays. Matches the formulation used in
// @c hdbscan_test.cpp and @c hdbscan_nn_descent_mst_test.cpp: contingency-table plus
// combinations-of-2; returns 1.0 for identical partitions up to label permutation.
double adjustedRandIndex(const std::vector<std::int32_t> &a, const std::vector<std::int32_t> &b) {
  const std::size_t n = a.size();
  if (b.size() != n || n < 2) {
    return 0.0;
  }
  // Treat noise labels (-1) as singleton clusters so ARI is well-defined on HDBSCAN output.
  auto remap = [n](const std::vector<std::int32_t> &labels) {
    std::vector<std::int32_t> remapped(n);
    std::int32_t next = 0;
    for (std::size_t i = 0; i < n; ++i) {
      if (labels[i] == -1) {
        remapped[i] = next++;
      }
    }
    std::unordered_map<std::int32_t, std::int32_t> seen;
    for (std::size_t i = 0; i < n; ++i) {
      if (labels[i] == -1) {
        continue;
      }
      auto it = seen.find(labels[i]);
      if (it == seen.end()) {
        it = seen.emplace(labels[i], next++).first;
      }
      remapped[i] = it->second;
    }
    return std::make_pair(remapped, static_cast<std::size_t>(next));
  };
  const auto [aD, ka] = remap(a);
  const auto [bD, kb] = remap(b);
  std::vector<std::uint64_t> ct(ka * kb, 0);
  std::vector<std::uint64_t> rowSums(ka, 0);
  std::vector<std::uint64_t> colSums(kb, 0);
  for (std::size_t i = 0; i < n; ++i) {
    const auto ai = static_cast<std::size_t>(aD[i]);
    const auto bi = static_cast<std::size_t>(bD[i]);
    ++ct[(ai * kb) + bi];
    ++rowSums[ai];
    ++colSums[bi];
  }
  auto choose2 = [](std::uint64_t v) {
    return v < 2 ? 0.0 : static_cast<double>(v) * static_cast<double>(v - 1) / 2.0;
  };
  double sumCt = 0.0;
  for (const std::uint64_t v : ct) {
    sumCt += choose2(v);
  }
  double sumA = 0.0;
  for (const std::uint64_t v : rowSums) {
    sumA += choose2(v);
  }
  double sumB = 0.0;
  for (const std::uint64_t v : colSums) {
    sumB += choose2(v);
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

// Two-moon reference fixture.  Points and frozen reference labels come from
// sklearn.datasets.make_moons(n_samples=200, noise=0.05, random_state=42) cast to float32 plus
// sklearn.cluster.HDBSCAN(min_cluster_size=5, algorithm='brute').fit_predict. The arrays below
// are a verbatim copy of @c hdbscan_test.cpp's @c TwoMoonFixture; the auto-backend test keeps
// its own copy so it is self-contained against a frozen reference.
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
  // Float literals here are random samples from a 2-D point cloud; the modernize-use-std-numbers
  // lint occasionally pattern-matches digits against mathematical constants (ln2, sqrt3, etc.) and
  // is inapplicable to fixture data.
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

// Indices of the variant arms in @ref AutoMstBackend's declared order. Mirrors the declaration so
// tests can confirm selection by reading @c heldIndex().
constexpr std::size_t kBoruvkaArm = 0;
constexpr std::size_t kPrimArm = 1;
constexpr std::size_t kNnDescentArm = 2;

} // namespace

// ---------------------------------------------------------------------------
// End-to-end: the default HDBSCAN path (AutoMstBackend) matches the sklearn reference.
// ---------------------------------------------------------------------------

TEST(HdbscanAutoBackend, DefaultFitMatchesSklearnOnTwoMoons) {
  const TwoMoonFixture fix;
  auto X = fix.toNdarray();

  // Default @c HDBSCAN<float> -- no explicit MstBackend -- so the dispatcher's arm is picked
  // automatically. On a 200-point 2-D input the dispatcher selects the Prim arm.
  HDBSCAN<float> h(5);
  h.run(X);

  std::vector<std::int32_t> ours(TwoMoonFixture::kN);
  for (std::size_t i = 0; i < TwoMoonFixture::kN; ++i) {
    ours[i] = h.labels()(i);
  }
  const double ari = adjustedRandIndex(ours, fix.referenceVec());
  EXPECT_GE(ari, 0.99) << "ari=" << ari;
  EXPECT_GE(h.nClusters(), 1u);
}

// ---------------------------------------------------------------------------
// Shape contract post-run: the dispatcher's output satisfies the frozen MstOutput contract.
// ---------------------------------------------------------------------------

TEST(HdbscanAutoBackend, MstOutputShapeAfterRun) {
  const std::size_t n = 400;
  const std::size_t d = 4;
  auto X = makeGaussian(n, d, 0xA55EULL);

  AutoMstBackend<float> backend;
  MstOutput<float> out;
  backend.run(X, /*minSamples=*/5, Pool{}, out);

  EXPECT_EQ(out.edges.size(), n - 1);
  EXPECT_EQ(out.coreDistances.dim(0), n);
}

// ---------------------------------------------------------------------------
// Explicit-backend agreement: pinning each of Prim / Borůvka / NN-Descent reproduces the
// default-backend labels at a shape the dispatcher would route to that backend.
// ---------------------------------------------------------------------------

TEST(HdbscanAutoBackend, ExplicitPrimMatchesDefaultOnTwoMoons) {
  // Two-moon shape sits below primNThreshold so the dispatcher picks Prim. Pinning Prim
  // explicitly must therefore reproduce the default-backend labels.
  const TwoMoonFixture fix;
  auto X = fix.toNdarray();

  HDBSCAN<float> hDefault(5);
  hDefault.run(X);

  HDBSCAN<float, PrimMstBackend<float>> hPinned(5);
  hPinned.run(X);

  std::vector<std::int32_t> defaultLabels(TwoMoonFixture::kN);
  std::vector<std::int32_t> pinnedLabels(TwoMoonFixture::kN);
  for (std::size_t i = 0; i < TwoMoonFixture::kN; ++i) {
    defaultLabels[i] = hDefault.labels()(i);
    pinnedLabels[i] = hPinned.labels()(i);
  }
  const double ari = adjustedRandIndex(defaultLabels, pinnedLabels);
  EXPECT_GE(ari, 0.99) << "ari=" << ari;
}

// Dispatch-selection tests use @c peekArm, which resolves the variant arm for a shape without
// running the full MST pipeline. Pinned-backend vs default-backend parity at realistic shapes
// is covered by the pybench hdbscan recipe, not by ctest: those runs are too slow to carry in
// every build.
//
// ---------------------------------------------------------------------------
// Dispatch selection: specific (N, d) points map to the expected variant arm.
// ---------------------------------------------------------------------------

TEST(HdbscanAutoBackend, DispatcherPicksPrimOnSmallLowD) {
  AutoMstBackend<float> backend;
  EXPECT_EQ(backend.peekArm(/*n=*/1000, /*d=*/2), kPrimArm);
}

TEST(HdbscanAutoBackend, DispatcherPicksBoruvkaOnLargeNLowD) {
  AutoMstBackend<float> backend;
  EXPECT_EQ(backend.peekArm(/*n=*/AutoMstBackend<float>::primNThreshold + 1000, /*d=*/8),
            kBoruvkaArm);
}

TEST(HdbscanAutoBackend, DispatcherPicksNnDescentOnLargeNHighD) {
  AutoMstBackend<float> backend;
  EXPECT_EQ(backend.peekArm(/*n=*/AutoMstBackend<float>::primNThreshold + 1000,
                            /*d=*/AutoMstBackend<float>::boruvkaDimCeil + 10),
            kNnDescentArm);
}

// ---------------------------------------------------------------------------
// Threshold boundaries: N just below / at / above primNThreshold, d at / just above boruvkaDimCeil.
// Each boundary point must resolve to exactly one backend (no gap).
// ---------------------------------------------------------------------------

TEST(HdbscanAutoBackend, DispatcherBoundaryPrimThreshold) {
  AutoMstBackend<float> backend;
  EXPECT_EQ(backend.peekArm(AutoMstBackend<float>::primNThreshold - 1, /*d=*/4), kPrimArm);
  EXPECT_EQ(backend.peekArm(AutoMstBackend<float>::primNThreshold, /*d=*/4), kBoruvkaArm);
}

TEST(HdbscanAutoBackend, DispatcherBoundaryBoruvkaDimCeil) {
  const std::size_t n = AutoMstBackend<float>::primNThreshold + 100;
  AutoMstBackend<float> backend;
  EXPECT_EQ(backend.peekArm(n, AutoMstBackend<float>::boruvkaDimCeil), kBoruvkaArm);
  EXPECT_EQ(backend.peekArm(n, AutoMstBackend<float>::boruvkaDimCeil + 1), kNnDescentArm);
}

// ---------------------------------------------------------------------------
// Dispatch totality: a grid across the threshold boundaries covers the full (N, d) dispatch
// surface. Every grid point routes to exactly one backend; no gap leaves a variant-monostate
// arm. The grid straddles both thresholds in each direction.
// ---------------------------------------------------------------------------

TEST(HdbscanAutoBackend, DispatcherTotalityGrid) {
  struct Case {
    std::size_t n;
    std::size_t d;
    std::size_t expectedArm;
    const char *label;
  };
  const std::size_t ptNe = AutoMstBackend<float>::primNThreshold;
  const std::size_t ptNlo = ptNe - 1;
  const std::size_t dCe = AutoMstBackend<float>::boruvkaDimCeil;
  const std::size_t dAbove = dCe + 1;
  const std::array<Case, 8> cases = {{
      {.n = 1000, .d = 2, .expectedArm = kPrimArm, .label = "N=1000,d=2"},
      {.n = 1000, .d = dCe, .expectedArm = kPrimArm, .label = "N=1000,d=dCe"},
      {.n = ptNlo, .d = 2, .expectedArm = kPrimArm, .label = "N=pt-1,d=2"},
      {.n = ptNlo, .d = dCe, .expectedArm = kPrimArm, .label = "N=pt-1,d=dCe"},
      {.n = ptNlo, .d = dAbove, .expectedArm = kPrimArm, .label = "N=pt-1,d=dCe+1"},
      {.n = ptNe, .d = 2, .expectedArm = kBoruvkaArm, .label = "N=pt,d=2"},
      {.n = ptNe, .d = dCe, .expectedArm = kBoruvkaArm, .label = "N=pt,d=dCe"},
      {.n = ptNe, .d = dAbove, .expectedArm = kNnDescentArm, .label = "N=pt,d=dCe+1"},
  }};
  AutoMstBackend<float> backend;
  for (const auto &c : cases) {
    EXPECT_EQ(backend.peekArm(c.n, c.d), c.expectedArm) << "case " << c.label;
  }
}

// ---------------------------------------------------------------------------
// Shape-tuple staleness: repeated @c peekArm calls at the same @c (N, d) reuse the held arm; a
// boundary-crossing call re-emplaces the variant.
// ---------------------------------------------------------------------------

TEST(HdbscanAutoBackend, ShapeStalenessReemplacementAcrossBoundary) {
  AutoMstBackend<float> backend;
  EXPECT_EQ(backend.peekArm(/*n=*/1000, /*d=*/4), kPrimArm);
  EXPECT_EQ(backend.peekArm(AutoMstBackend<float>::primNThreshold + 200, /*d=*/4), kBoruvkaArm);
  EXPECT_EQ(backend.peekArm(AutoMstBackend<float>::primNThreshold + 200,
                            AutoMstBackend<float>::boruvkaDimCeil + 5),
            kNnDescentArm);
  EXPECT_EQ(backend.peekArm(/*n=*/1000, /*d=*/4), kPrimArm);
}
