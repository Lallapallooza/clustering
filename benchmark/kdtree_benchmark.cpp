#include <benchmark/benchmark.h>
#include <random>
#include <new>
#include "clustering/kdtree.h"
#include "clustering/memory/new_alloc.h"

template<class T, float Min = -1.0f, float Max = 1.0f>
static NDArray<T, 2> generateRandomPoints(size_t num_points, size_t dims) {
  std::random_device                     rd;
  std::mt19937                           gen(rd());
  std::uniform_real_distribution<double> dis(Min, Max);

  NDArray<T, 2> points({num_points, dims});
  for (size_t   i = 0; i < num_points; ++i) {
    for (size_t j = 0; j < dims; ++j) {
      points[i][j] = dis(gen);
    }
  }

  return points;
}

template<class T, float Min = -1.0f, float Max = 1.0f>
static NDArray<T, 1> generateRandomQueryPoint(size_t dimensions) {
  std::random_device                     rd;
  std::mt19937                           gen(rd());
  std::uniform_real_distribution<double> dis(Min, Max);

  NDArray<T, 1> queryPoint({dimensions});
  for (size_t   j = 0; j < dimensions; ++j) {
    queryPoint[j] = dis(gen);
  }

  return queryPoint;
}

template<class KDTreeT, float Min = -1.0f, float Max = 1.0f>
static void BM_KDTreeConstruction(benchmark::State &state) {
  size_t num_points = state.range(0);
  size_t dimensions = state.range(1);

  for (auto _: state) {
    state.PauseTiming();
    auto points = generateRandomPoints<typename KDTreeT::value_type, Min, Max>(num_points, dimensions);
    char* buffer = new char[sizeof(KDTreeT)];

    state.ResumeTiming();
    auto* tree = new(buffer) KDTreeT(points);

    state.PauseTiming();
    tree->~KDTreeT();
    delete[] buffer;
    state.ResumeTiming();
  }
}

template<class KDTreeT, float Min = -1.0f, float Max = 1.0f>
static void BM_KDTreeQuery(benchmark::State &state) {
  size_t num_points = state.range(0);
  size_t dimensions = state.range(1);

  auto    points = generateRandomPoints<typename KDTreeT::value_type, Min, Max>(num_points, dimensions);
  KDTreeT tree(points);

  for (auto _: state) {
    auto queryPoint = generateRandomQueryPoint<typename KDTreeT::value_type, Min, Max>(dimensions);
    state.PauseTiming();
    auto radius = generateRandomQueryPoint<typename KDTreeT::value_type, 0.f, Max>(1)[0];
    state.ResumeTiming();

    benchmark::DoNotOptimize(tree.query(queryPoint, radius));
  }
}

BENCHMARK(BM_KDTreeConstruction<KDTree<float>>)
  ->Ranges({{2, 1 << 8}, {2, 1 << 5}})
    ->RangeMultiplier(2);
BENCHMARK(BM_KDTreeConstruction<KDTree<float, KDTreeDistanceType::kEucledian, NewAllocator<KDTreeNode>>>)
  ->Ranges({{2, 1 << 8}, {2, 1 << 5}})
    ->RangeMultiplier(2);

BENCHMARK(BM_KDTreeQuery<KDTree<float>>)
  ->Ranges({{2, 1 << 8}, {2, 1 << 5}})
    ->RangeMultiplier(2);
BENCHMARK(BM_KDTreeQuery<KDTree<float, KDTreeDistanceType::kEucledian, NewAllocator<KDTreeNode>>>)
  ->Ranges({{2, 1 << 8}, {2, 1 << 5}})
    ->RangeMultiplier(2);


BENCHMARK_MAIN();