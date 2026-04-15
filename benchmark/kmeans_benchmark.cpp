#include <benchmark/benchmark.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <random>

#include "clustering/kmeans.h"
#include "clustering/kmeans/detail/dispatch.h"
#include "clustering/ndarray.h"

using clustering::KMeans;
using clustering::NDArray;

namespace {

// Seeded deterministically so every benchmark iteration fits against identical data. Benchmarks
// must not tax the RNG for reproducibility across runs.
NDArray<float, 2> makeBlobs(std::size_t n, std::size_t d, std::size_t k, std::uint32_t seed) {
  NDArray<float, 2> X({n, d});
  std::mt19937 gen(seed);
  std::uniform_int_distribution<std::size_t> pickCluster(0, k - 1);
  std::normal_distribution<float> noise(0.0F, 0.5F);

  for (std::size_t i = 0; i < n; ++i) {
    const std::size_t c = pickCluster(gen);
    for (std::size_t t = 0; t < d; ++t) {
      // Spread centers along axis 0 so sigma << inter-center distance keeps the blobs cleanly
      // separable.
      const float center = (t == 0) ? static_cast<float>(c * 50U) : 0.0F;
      X(i, t) = center + noise(gen);
    }
  }
  return X;
}

void runKMeans(benchmark::State &state, std::size_t n, std::size_t d, std::size_t k,
               std::optional<clustering::kmeans::detail::Algorithm> forced) {
  auto X = makeBlobs(n, d, k, /*seed=*/1234U);

  // Fresh fitter per iteration so scratch allocation is amortized into the measured cost, the
  // same way a Python caller pays it per fit().
  for (auto _ : state) {
    KMeans<float> km(k, /*nJobs=*/1);
    if (forced.has_value()) {
      km.forceAlgorithm(*forced);
    }
    km.run(X, /*maxIter=*/300, /*tol=*/1e-4F, /*seed=*/42U);
    benchmark::DoNotOptimize(km.inertia());
    benchmark::ClobberMemory();
  }

  state.counters["n"] = static_cast<double>(n);
  state.counters["d"] = static_cast<double>(d);
  state.counters["k"] = static_cast<double>(k);
}

// Auto-dispatch sweep across the envelope corners so the benchmark surfaces the shape-vs-runtime
// map the Python-facing dispatch will drive.
void BM_KMeansAuto_LowDSmall(benchmark::State &s) { runKMeans(s, 1000, 2, 8, std::nullopt); }
void BM_KMeansAuto_LowDMid(benchmark::State &s) { runKMeans(s, 10000, 4, 16, std::nullopt); }
void BM_KMeansAuto_MidDMid(benchmark::State &s) { runKMeans(s, 10000, 32, 32, std::nullopt); }
void BM_KMeansAuto_HighKMid(benchmark::State &s) { runKMeans(s, 10000, 32, 256, std::nullopt); }
void BM_KMeansAuto_HighKHigh(benchmark::State &s) { runKMeans(s, 50000, 32, 512, std::nullopt); }

} // namespace

BENCHMARK(BM_KMeansAuto_LowDSmall)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK(BM_KMeansAuto_LowDMid)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK(BM_KMeansAuto_MidDMid)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK(BM_KMeansAuto_HighKMid)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK(BM_KMeansAuto_HighKHigh)->Unit(benchmark::kMillisecond)->UseRealTime();

BENCHMARK_MAIN();
