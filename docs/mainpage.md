# Clustering {#mainpage}

<div class="hero">
  <div class="hero-tagline">
    Header-only C++20 clustering library &mdash; KD-Tree acceleration, AVX2 hot paths, a BS::thread_pool worker pool, and a nanobind Python binding.
  </div>
  <div class="hero-pills">
    <span class="hero-pill">C++20</span>
    <span class="hero-pill">AVX2 + FMA</span>
    <span class="hero-pill">Header-only</span>
    <span class="hero-pill">CPM / add_subdirectory</span>
    <span class="hero-pill">Python via nanobind</span>
  </div>
</div>

## Algorithms

- \ref clustering::DBSCAN "DBSCAN" &mdash; KD-tree-accelerated region queries; parallel via `BS::thread_pool`.
- \ref clustering::HDBSCAN "HDBSCAN*" &mdash; Campello 2015 hierarchy with Prim / Boruvka / NN-Descent MST backends, EOM + leaf extraction, GLOSH outlier scores.
- \ref clustering::KMeans "k-means" &mdash; fused argmin-GEMM Lloyd with greedy k-means++ and AFK-MC2 seeders; direct small-d kernel; chunked materialized fallback.
- \ref clustering::NDArray "NDArray" &mdash; shape/stride owned-or-borrowed tensor, layout-tagged for contiguity, with aligned-allocator storage.

## Install (CMake via CPM, recommended)

```cmake
include(CPM.cmake)

CPMAddPackage(
    NAME clustering
    GITHUB_REPOSITORY Lallapallooza/clustering
    GIT_TAG v0.8.0
    OPTIONS "CLUSTERING_USE_AVX2 ON"
)

target_link_libraries(MyTarget PRIVATE clustering_header_lib)
```

Consumers get the `clustering_header_lib` INTERFACE target with `-mavx2 -mfma` and the `CLUSTERING_USE_AVX2` compile definition. Tests, benchmarks, and clang-tidy default to **OFF** when pulled in as a dependency.

### Install (add_subdirectory)

```bash
git clone https://github.com/Lallapallooza/clustering.git third_party/clustering
```

```cmake
add_subdirectory(third_party/clustering)
target_link_libraries(MyTarget PRIVATE clustering_header_lib)
```

## Install (Python, via uv from GitHub)

The Python binding is built through scikit-build-core + nanobind. Install straight from a release tag with `uv`:

```bash
uv pip install "clustering @ git+https://github.com/Lallapallooza/clustering.git@v0.8.0"
```

Or from `main`:

```bash
uv pip install "clustering @ git+https://github.com/Lallapallooza/clustering.git"
```

The build needs a C++20 toolchain (`clang++-18` or `g++-13`), CMake >= 3.22, and Ninja; everything else is wheeled.

### Install (Python, editable from a clone)

```bash
git clone https://github.com/Lallapallooza/clustering.git
cd clustering
uv sync --group dev
uv pip install -e .
```

`uv pip install -e .` triggers the scikit-build-core build; subsequent `uv run` invocations reuse the compiled `_clustering` extension.

## C++ usage

### DBSCAN

```cpp
#include "clustering/dbscan.h"

int main() {
  clustering::NDArray<float, 2> points({numPoints, dimensions});
  // ... fill points ...

  clustering::DBSCAN<float> dbscan(points, /*eps=*/0.5f, /*minPts=*/5, /*nJobs=*/4);
  dbscan.run();

  std::cout << "labels:     " << dbscan.labels().dim(0) << '\n';
  std::cout << "n_clusters: " << dbscan.nClusters() << '\n';
}
```

### HDBSCAN

```cpp
#include "clustering/hdbscan.h"

clustering::NDArray<float, 2> X({n, d});
// ... fill X ...

clustering::HDBSCAN<float> hdb(/*minClusterSize=*/5);
hdb.run(X);

const auto& labels   = hdb.labels();          // length n; -1 is noise
const auto& outliers = hdb.outlierScores();   // length n; GLOSH in [0, 1]
std::cout << "n_clusters: " << hdb.nClusters() << '\n';
```

### k-means

```cpp
#include "clustering/kmeans.h"

clustering::NDArray<float, 2> X({n, d});
// ... fill X ...

clustering::KMeans<float> km(/*k=*/16, /*nJobs=*/4);
km.run(X, /*maxIter=*/300, /*tol=*/1e-4f, /*seed=*/42);

const auto& labels    = km.labels();     // length n; entries in [0, k)
const auto& centroids = km.centroids();  // k x d
std::cout << "inertia: " << km.inertia() << '\n';
```

## Python usage

```python
import numpy as np
from _clustering import dbscan, kmeans, hdbscan

X = np.random.rand(10_000, 4).astype(np.float32)

# DBSCAN
dbscan_labels = dbscan(X, eps=0.5, min_pts=5, n_jobs=4)

# HDBSCAN
hdb_labels, outlier_scores, n_clusters = hdbscan(X, min_cluster_size=5)

# k-means
km_labels, centroids, inertia, n_iter, converged = kmeans(X, k=16, n_jobs=4)
```

## Testing

### C++ test suite

```bash
cmake -S . -B build -G Ninja
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Run a single target or a single GTest case:

```bash
ctest --test-dir build --output-on-failure -R kmeans_lloyd_test
./build/tests/kmeans_lloyd_test --gtest_filter='KMeansLloyd.SmallD*'
```

### Python test suite

```bash
uv run pytest                  # exercises pybench internals + binding smoke tests
```

### Benchmarks (head-to-head vs scikit-learn)

```bash
uv run benchmark --list                             # list recipes
uv run benchmark --algo kmeans --sizes 5000 50000   # narrow run
uv run benchmark                                    # full matrix
```

Start narrow (`--algo`, `--sizes`) when iterating; the full matrix takes minutes to tens of minutes. `--ours-only` skips the sklearn baseline for tight CPU-perf loops.

## Build options

All default **ON** when this is the top-level project and **OFF** when consumed as a subdirectory.

| Option | Default | Effect |
|---|---|---|
| `CLUSTERING_BUILD_BENCHMARK` | ON (top-level) | Google Benchmark targets (`kdtree_benchmark`, `gemm_benchmark`, `kmeans_benchmark`, `dbscan_benchmark`, `nn_descent_benchmark`). |
| `CLUSTERING_BUILD_TESTS` | ON (top-level) | GoogleTest target and the `tests/` subtree. |
| `CLUSTERING_ENABLE_CLANG_TIDY` | ON (top-level) | Per-target clang-tidy wiring. Not a pre-commit hook. |
| `CLUSTERING_USE_AVX2` | auto-detected | `-mavx2 -mfma` + `CLUSTERING_USE_AVX2` define on the interface target. |
| `CLUSTERING_BUILD_WITH_SANITIZER` | OFF | Adds `-fsanitize=thread -fno-omit-frame-pointer`. |
| `CLUSTERING_BUILD_DOCS` | OFF | Builds this API documentation site via Doxygen. |

## Pre-commit hooks

```bash
uv sync --group dev
uv run pre-commit install                        # pre-commit hook (once)
uv run pre-commit install --hook-type commit-msg # commit-msg hook (once)
uv run pre-commit run --all-files                # run every hook on the repo
```

`clang-format`, `gersemi`, `ruff`, whitespace/YAML checks, and Conventional-Commits enforcement via commitizen. `clang-tidy` is **not** a pre-commit hook &mdash; it runs inside the build when `CLUSTERING_ENABLE_CLANG_TIDY=ON` and in CI.

## Source

Code lives at [github.com/Lallapallooza/clustering](https://github.com/Lallapallooza/clustering). CI mirrors the local loop (clang-18 + Ninja + ctest).
