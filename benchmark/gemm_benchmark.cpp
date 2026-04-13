#include <benchmark/benchmark.h>

#include <BS_thread_pool.hpp>
#include <cstddef>
#include <random>

#include "clustering/math/gemm.h"
#include "clustering/math/thread.h"
#include "clustering/ndarray.h"

#ifdef CLUSTERING_BENCHMARK_OPENBLAS
#include <cblas.h>
#endif

using clustering::NDArray;
using clustering::math::gemm;
using clustering::math::Pool;

namespace {

// Fixed worker count: OpenBLAS is pinned to the same value below so comparisons are fair.
constexpr std::size_t kWorkerCount = 8;

// Seeded deterministically -- benchmarks must not tax the RNG for reproducibility across runs.
template <class T> void fillRandom(NDArray<T, 2> &a, std::uint32_t seed) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<T> dist(static_cast<T>(-1.0), static_cast<T>(1.0));
  const std::size_t M = a.dim(0);
  const std::size_t N = a.dim(1);
  for (std::size_t i = 0; i < M; ++i) {
    for (std::size_t j = 0; j < N; ++j) {
      a[i][j] = dist(gen);
    }
  }
}

template <class T> void fillZero(NDArray<T, 2> &a) {
  const std::size_t M = a.dim(0);
  const std::size_t N = a.dim(1);
  for (std::size_t i = 0; i < M; ++i) {
    for (std::size_t j = 0; j < N; ++j) {
      a[i][j] = T{0};
    }
  }
}

void setGflopsCounter(benchmark::State &state, std::size_t M, std::size_t N, std::size_t K) {
  // 2*M*N*K fused multiply-adds per GEMM. kIsIterationInvariantRate multiplies by iterations and
  // divides by wall time, yielding flops/s; pre-scaling by 1e-9 surfaces the number in GFLOPS.
  const double flopsPerIter =
      2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K);
  state.counters["GFLOPS"] =
      benchmark::Counter(flopsPerIter * 1.0e-9, benchmark::Counter::kIsIterationInvariantRate);
}

void runOurGemm(benchmark::State &state, std::size_t M, std::size_t N, std::size_t K) {
  NDArray<float, 2> A({M, K});
  NDArray<float, 2> B({K, N});
  NDArray<float, 2> C({M, N});
  fillRandom(A, 1U);
  fillRandom(B, 2U);
  fillZero(C);

  BS::light_thread_pool pool(kWorkerCount);

  for (auto _ : state) {
    gemm(A, B, C, Pool{&pool}, 1.0F, 0.0F);
    benchmark::DoNotOptimize(C.data());
    benchmark::ClobberMemory();
  }

  setGflopsCounter(state, M, N, K);
}

#ifdef CLUSTERING_BENCHMARK_OPENBLAS
void runOpenBlas(benchmark::State &state, std::size_t M, std::size_t N, std::size_t K) {
  NDArray<float, 2> A({M, K});
  NDArray<float, 2> B({K, N});
  NDArray<float, 2> C({M, N});
  fillRandom(A, 1U);
  fillRandom(B, 2U);
  fillZero(C);

  // Pin OpenBLAS to the same worker count our pool uses. Reread on every benchmark entry because
  // global OpenBLAS state survives across fixtures.
  openblas_set_num_threads(static_cast<int>(kWorkerCount));

  for (auto _ : state) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, static_cast<int>(M), static_cast<int>(N),
                static_cast<int>(K), 1.0F, A.data(), static_cast<int>(K), B.data(),
                static_cast<int>(N), 0.0F, C.data(), static_cast<int>(N));
    benchmark::DoNotOptimize(C.data());
    benchmark::ClobberMemory();
  }

  setGflopsCounter(state, M, N, K);
}
#endif

// Canonical clustering shape: tall-skinny (M=10k, N=10, K=64) f32, the
// pairwise-L2 / kNN-graph workload the math arch is sized around.
void BM_OurGemm_Canonical(benchmark::State &state) { runOurGemm(state, 10000, 10, 64); }
void BM_OurGemm_Square(benchmark::State &state) { runOurGemm(state, 1024, 1024, 1024); }
void BM_OurGemm_TallSkinny(benchmark::State &state) { runOurGemm(state, 10000, 64, 128); }
void BM_OurGemm_Small(benchmark::State &state) { runOurGemm(state, 64, 64, 64); }

#ifdef CLUSTERING_BENCHMARK_OPENBLAS
void BM_OpenBLAS_Canonical(benchmark::State &state) { runOpenBlas(state, 10000, 10, 64); }
void BM_OpenBLAS_Square(benchmark::State &state) { runOpenBlas(state, 1024, 1024, 1024); }
void BM_OpenBLAS_TallSkinny(benchmark::State &state) { runOpenBlas(state, 10000, 64, 128); }
void BM_OpenBLAS_Small(benchmark::State &state) { runOpenBlas(state, 64, 64, 64); }
#else
void reportOpenblasAbsent(benchmark::State &state) {
  for (auto _ : state) {
    // One-shot: Google Benchmark still needs to enter the state loop to produce a row.
    break;
  }
  state.SkipWithMessage("OpenBLAS not linked -- gate skipped");
}
void BM_OpenBLAS_Canonical(benchmark::State &state) { reportOpenblasAbsent(state); }
#endif

} // namespace

// UseRealTime: our path and OpenBLAS both fan out to 8 workers; CPU time on the main thread
// understates the kernel's parallel throughput. Real time is what the caller observes.
BENCHMARK(BM_OurGemm_Canonical)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK(BM_OurGemm_Square)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK(BM_OurGemm_TallSkinny)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK(BM_OurGemm_Small)->Unit(benchmark::kMicrosecond)->UseRealTime();

#ifdef CLUSTERING_BENCHMARK_OPENBLAS
BENCHMARK(BM_OpenBLAS_Canonical)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK(BM_OpenBLAS_Square)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK(BM_OpenBLAS_TallSkinny)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK(BM_OpenBLAS_Small)->Unit(benchmark::kMicrosecond)->UseRealTime();
#else
BENCHMARK(BM_OpenBLAS_Canonical);
#endif

// NOLINTNEXTLINE(misc-const-correctness,modernize-avoid-c-arrays)
BENCHMARK_MAIN();
