#include <benchmark/benchmark.h>

#include <BS_thread_pool.hpp>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <random>
#include <vector>

#include "clustering/dbscan.h"
#include "clustering/index/brute_force_pairwise.h"
#include "clustering/index/kdtree.h"
#include "clustering/math/thread.h"
#include "clustering/ndarray.h"

using clustering::BruteForcePairwise;
using clustering::DBSCAN;
using clustering::KDTree;
using clustering::KDTreeDistanceType;
using clustering::NDArray;

namespace {

// Approximate pybench's @c DatasetSpec(centers=20, cluster_std=3.0, center_box=(-150,150)) so
// the cells we measure line up with the Python-side recipe. Full bit-exactness against
// @c sklearn.make_blobs is not worth the dep: the clustering structure (20 isotropic blobs in
// a 300-unit cube) is what drives DBSCAN's work profile.
NDArray<float, 2> makeBlobs(std::size_t n, std::size_t d, std::uint64_t seed) {
  constexpr std::size_t kCenters = 20;
  constexpr float kStd = 3.0F;
  constexpr float kBox = 150.0F;

  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<float> center_dist(-kBox, kBox);
  std::normal_distribution<float> jitter(0.0F, kStd);

  std::vector<float> centers(kCenters * d);
  for (std::size_t c = 0; c < kCenters; ++c) {
    for (std::size_t j = 0; j < d; ++j) {
      centers[(c * d) + j] = center_dist(rng);
    }
  }

  NDArray<float, 2> x({n, d});
  std::uniform_int_distribution<std::size_t> pick_center(0, kCenters - 1);
  for (std::size_t i = 0; i < n; ++i) {
    const std::size_t c = pick_center(rng);
    for (std::size_t j = 0; j < d; ++j) {
      x[i][j] = centers[(c * d) + j] + jitter(rng);
    }
  }
  return x;
}

// vMF-like fixture for the d >= 16 brute-force path. pybench's @c make_vmf draws from
// @c scipy.stats.vonmises_fisher with @c kappa=20; we cannot pull scipy into a C++ benchmark, so
// approximate the same shape: pick @c centers unit-vector means uniformly on the sphere, then for
// each sample jitter the chosen mean by an isotropic Gaussian and renormalise. The resulting cloud
// has the same essential property the brute-force path reacts to -- everything sits on the unit
// sphere with a small per-cluster angular spread -- which puts the eps-neighbour density at the
// same ~0.2% the Python benchmark sees, so the kernel work per cell is comparable.
NDArray<float, 2> makeVmfLikeData(std::size_t n, std::size_t d, std::size_t centers,
                                  std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::normal_distribution<float> normal(0.0F, 1.0F);

  std::vector<float> mus(centers * d);
  for (std::size_t c = 0; c < centers; ++c) {
    float norm = 0.0F;
    for (std::size_t j = 0; j < d; ++j) {
      const float v = normal(rng);
      mus[(c * d) + j] = v;
      norm += v * v;
    }
    norm = std::sqrt(norm);
    for (std::size_t j = 0; j < d; ++j) {
      mus[(c * d) + j] /= norm;
    }
  }

  // 1/sqrt(kappa) where kappa=20 sets the per-coordinate jitter scale. A direct vMF draw would
  // sample a different conditional density, but for d=32 with these centres the resulting
  // unit-sphere cloud has the right per-row neighbour count for the eps the recipe picks.
  const float jitterScale = 1.0F / std::sqrt(20.0F);
  std::uniform_int_distribution<std::size_t> pickCenter(0, centers - 1);

  NDArray<float, 2> X({n, d});
  std::vector<float> rowBuf(d);
  for (std::size_t i = 0; i < n; ++i) {
    const std::size_t c = pickCenter(rng);
    float norm = 0.0F;
    for (std::size_t j = 0; j < d; ++j) {
      const float v = mus[(c * d) + j] + (jitterScale * normal(rng));
      rowBuf[j] = v;
      norm += v * v;
    }
    const float invNorm = 1.0F / std::sqrt(norm);
    for (std::size_t j = 0; j < d; ++j) {
      X[i][j] = rowBuf[j] * invNorm;
    }
  }
  return X;
}

// Pinned @c eps values derived from pybench's knee policy on this fixture shape. Baked in so
// the benchmark does not depend on @c kneed; the numbers track the values the Python recipe
// reports inside an order-of-magnitude window at each shape.
float pinnedEpsFor(std::size_t n, std::size_t d) {
  if (d == 2) {
    if (n <= 1500) {
      return 4.03F;
    }
    if (n <= 15000) {
      return 2.21F;
    }
    return 1.78F;
  }
  // d == 8
  if (n <= 1500) {
    return 10.9F;
  }
  if (n <= 15000) {
    return 9.2F;
  }
  return 7.6F;
}

// Full @c DBSCAN::run end-to-end: index construction, adjacency sweep, core marking and BFS
// expansion. Shape arguments are @c (n, d, n_jobs) so the matrix covers the KDTree-path cells
// pybench exercises (@c d < 16); higher-d cells route through @c BruteForcePairwise, which is
// out of this fixture's scope.
void BM_DBSCANRun(benchmark::State &state) {
  const auto n = static_cast<std::size_t>(state.range(0));
  const auto d = static_cast<std::size_t>(state.range(1));
  const auto nJobs = static_cast<std::size_t>(state.range(2));
  const float eps = pinnedEpsFor(n, d);

  auto data = makeBlobs(n, d, /*seed=*/42);
  DBSCAN<float> algo(eps, /*minPts=*/5, nJobs);
  // Warm the pool: the first run pays the thread-spawn tax while subsequent runs reuse. The
  // fixture measures amortized run time, not cold-start.
  algo.run(data);

  for (auto _ : state) {
    algo.run(data);
    benchmark::DoNotOptimize(algo.labels().data());
    benchmark::ClobberMemory();
  }
  state.counters["nClusters"] = static_cast<double>(algo.nClusters());
  state.counters["eps"] = eps;
}

// Adjacency-sweep only: isolates the range-index's radius query so regressions in the tree
// walk / leaf brute force surface without BFS + core-marking noise. Also reports the adjacency
// degree distribution so callers can see how much raw distance work the sweep did.
template <class Index> void BM_RangeQuery(benchmark::State &state) {
  const auto n = static_cast<std::size_t>(state.range(0));
  const auto d = static_cast<std::size_t>(state.range(1));
  const auto nJobs = static_cast<std::size_t>(state.range(2));
  const float eps = pinnedEpsFor(n, d);

  auto data = makeBlobs(n, d, /*seed=*/42);
  Index idx(data);

  std::optional<BS::light_thread_pool> workerPool;
  if (nJobs > 1) {
    workerPool.emplace(nJobs);
  }
  const clustering::math::Pool pool{workerPool.has_value() ? &*workerPool : nullptr};

  std::size_t lastTotal = 0;
  std::size_t lastMax = 0;
  for (auto _ : state) {
    auto adj = idx.query(eps, pool);
    std::size_t total = 0;
    std::size_t rowMax = 0;
    for (const auto &row : adj) {
      total += row.size();
      if (row.size() > rowMax) {
        rowMax = row.size();
      }
    }
    lastTotal = total;
    lastMax = rowMax;
    benchmark::DoNotOptimize(adj);
    benchmark::ClobberMemory();
  }
  state.counters["edges"] = static_cast<double>(lastTotal);
  state.counters["meanDeg"] = static_cast<double>(lastTotal) / static_cast<double>(n);
  state.counters["maxDeg"] = static_cast<double>(lastMax);
  state.counters["eps"] = eps;
}

// Perf-target fixture: matches the pybench @c (d=32, n=10000, n_jobs=1) cell that exercises the
// brute-force adjacency path. Routes through @c BruteForcePairwise (d >= 16 brute-force floor)
// and the fused AVX2 threshold microkernel. Reports both the full @c DBSCAN::run cost and the
// isolated adjacency sweep so optimization deltas attribute cleanly.
void BM_DBSCANRun_VMF_d32_n10k_j1(benchmark::State &state) {
  constexpr std::size_t kN = 10000;
  constexpr std::size_t kD = 32;
  constexpr std::size_t kCenters = 20;
  constexpr float kEps = 0.98F;

  auto data = makeVmfLikeData(kN, kD, kCenters, /*seed=*/42);
  DBSCAN<float> algo(kEps, /*minPts=*/5, /*nJobs=*/1);
  algo.run(data);

  for (auto _ : state) {
    algo.run(data);
    benchmark::DoNotOptimize(algo.labels().data());
    benchmark::ClobberMemory();
  }
  state.counters["nClusters"] = static_cast<double>(algo.nClusters());
  state.counters["eps"] = kEps;
}

void BM_BruteForceQuery_VMF_d32_n10k_j1(benchmark::State &state) {
  constexpr std::size_t kN = 10000;
  constexpr std::size_t kD = 32;
  constexpr std::size_t kCenters = 20;
  constexpr float kEps = 0.98F;

  auto data = makeVmfLikeData(kN, kD, kCenters, /*seed=*/42);
  const BruteForcePairwise<float> idx(data);
  const clustering::math::Pool pool{nullptr};

  std::size_t lastTotal = 0;
  for (auto _ : state) {
    auto adj = idx.query(kEps, pool);
    std::size_t total = 0;
    for (const auto &row : adj) {
      total += row.size();
    }
    lastTotal = total;
    benchmark::DoNotOptimize(adj);
    benchmark::ClobberMemory();
  }
  state.counters["edges"] = static_cast<double>(lastTotal);
  state.counters["meanDeg"] = static_cast<double>(lastTotal) / static_cast<double>(kN);
  state.counters["eps"] = kEps;
}

} // namespace

// Shape matrix: the @c (n, d) cells pybench's DBSCAN recipe exercises on the KDTree path
// (@c d < 16), plus @c n_jobs sweep to cover serial, mid, and full fan-out. The 100k / 16-jobs
// cell is the one the perf slice targets; the 1k / 1-job and 10k / 4-jobs cells guard the
// small-shape tail that could regress on pool overhead or reorder build cost.
BENCHMARK(BM_DBSCANRun)
    ->ArgsProduct({
        {1000, 10000, 100000},
        {2, 8},
        {1, 4, 16},
    })
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_RangeQuery<KDTree<float, KDTreeDistanceType::kEucledian>>)
    ->ArgsProduct({
        {1000, 10000, 100000},
        {2, 8},
        {1, 4, 16},
    })
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_DBSCANRun_VMF_d32_n10k_j1)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK(BM_BruteForceQuery_VMF_d32_n10k_j1)->Unit(benchmark::kMillisecond)->UseRealTime();

// NOLINTNEXTLINE(misc-const-correctness,modernize-avoid-c-arrays)
BENCHMARK_MAIN();
