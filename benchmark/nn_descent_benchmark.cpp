#include <benchmark/benchmark.h>

#include <BS_thread_pool.hpp>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>

#include "clustering/index/nn_descent.h"
#include "clustering/math/thread.h"
#include "clustering/ndarray.h"

using clustering::NDArray;
using clustering::index::NnDescentIndex;
using clustering::math::Pool;

namespace {

NDArray<float, 2> vmfLikeUnitSphere(std::size_t n, std::size_t d, std::uint64_t seed = 42U) {
  std::mt19937_64 gen(seed);
  std::normal_distribution<float> nd(0.0F, 1.0F);
  NDArray<float, 2> X({n, d});
  float *data = X.data();
  for (std::size_t i = 0; i < n; ++i) {
    float *row = data + (i * d);
    float norm = 0.0F;
    for (std::size_t j = 0; j < d; ++j) {
      const float v = nd(gen);
      row[j] = v;
      norm += v * v;
    }
    norm = std::sqrt(norm);
    if (norm > 0.0F) {
      for (std::size_t j = 0; j < d; ++j) {
        row[j] /= norm;
      }
    }
  }
  return X;
}

void BM_NnDescentBuild(benchmark::State &state) {
  const auto n = static_cast<std::size_t>(state.range(0));
  const auto d = static_cast<std::size_t>(state.range(1));
  const auto k = static_cast<std::size_t>(state.range(2));
  const auto nJobs = static_cast<std::size_t>(state.range(3));
  const auto X = vmfLikeUnitSphere(n, d);

  BS::light_thread_pool pool(nJobs > 0 ? nJobs : 1);
  const Pool poolHandle{nJobs > 1 ? &pool : nullptr};

  std::size_t lastIters = 0;
  for (auto _ : state) {
    NnDescentIndex<float> index(k, /*maxIter=*/10U, /*delta=*/0.001F, /*seed=*/42U);
    index.build(X, poolHandle);
    lastIters = index.lastIterations();
    benchmark::DoNotOptimize(index.neighbors().data());
  }
  state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations() * n));
  state.SetLabel("n=" + std::to_string(n) + " d=" + std::to_string(d) + " k=" + std::to_string(k) +
                 " jobs=" + std::to_string(nJobs) + " iters=" + std::to_string(lastIters));
}

} // namespace

BENCHMARK(BM_NnDescentBuild)
    ->Args({5000, 128, 14, 1})
    ->Args({5000, 128, 14, 4})
    ->Args({5000, 128, 14, 16})
    ->Args({25000, 128, 14, 1})
    ->Args({25000, 128, 14, 4})
    ->Args({25000, 128, 14, 16})
    ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
