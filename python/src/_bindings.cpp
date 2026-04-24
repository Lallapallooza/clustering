#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#include "clustering/dbscan.h"
#include "clustering/hdbscan.h"
#include "clustering/hdbscan/policy/auto_mst_backend.h"
#include "clustering/hdbscan/policy/boruvka_mst_backend.h"
#include "clustering/hdbscan/policy/nn_descent_mst_backend.h"
#include "clustering/hdbscan/policy/prim_mst_backend.h"
#include "clustering/kmeans.h"
#include "clustering/ndarray.h"

#include "ndarray_adapter.h"

namespace nb = nanobind;
using clustering::DBSCAN;
using clustering::HDBSCAN;
using clustering::KMeans;
using clustering::NDArray;
using clustering::hdbscan::ClusterSelectionMethod;
using clustering::python::borrowFromNumpyContig;
using clustering::python::borrowFromNumpyContigReadOnly;
using clustering::python::borrowFromNumpyStrided;
using clustering::python::wrapAsNumpy;

static nb::ndarray<nb::numpy, std::int32_t, nb::ndim<1>>
dbscan_binding(nb::ndarray<float, nb::ndim<2>, nb::c_contig, nb::device::cpu> data, float eps,
               int min_pts, int n_jobs) {
  const std::size_t jobs = (n_jobs <= 0) ? std::size_t{0} : static_cast<std::size_t>(n_jobs);

  const std::size_t N = data.shape(0);
  const std::size_t D = data.shape(1);

  if (min_pts < 1) {
    throw nb::value_error("min_pts must be >= 1");
  }
  if (!(eps >= 0.0F)) {
    throw nb::value_error("eps must be >= 0");
  }

  // Borrow the numpy buffer directly when it is 32-byte aligned: several AVX2 tiers assume
  // 32-byte aligned loads on @p X through the @c NDArray::isAligned<32>() gate. Borrowing a
  // loosely-aligned numpy array would force the dispatcher down fallback paths whose
  // interactions are less thoroughly exercised. Align-gated borrow captures the common case
  // (numpy contiguous arrays from @c make_blobs, @c astype, etc. on x86_64 glibc land on
  // 32-byte boundaries here) without exposing the unaligned fallback.
  constexpr std::size_t kAvx2Align = 32;
  const bool dataAligned = (reinterpret_cast<std::uintptr_t>(data.data()) % kAvx2Align) == 0;
  NDArray<float, 2> xOwned({0, 0});
  auto X = [&] {
    if (dataAligned) {
      return clustering::python::borrowFromNumpyContig<float>(data);
    }
    xOwned = NDArray<float, 2>({N, D});
    std::memcpy(xOwned.data(), data.data(), N * D * sizeof(float));
    return NDArray<float, 2, clustering::Layout::Contig>::borrow(xOwned.data(), {N, D});
  }();

  DBSCAN<float> algo(eps, static_cast<std::size_t>(min_pts), jobs);
  {
    nb::gil_scoped_release release;
    algo.run(X);
  }

  // Detach labels into a fresh owned buffer so the solver remains reusable and the capsule
  // has full ownership of the numpy-visible storage.
  NDArray<std::int32_t, 1> labels_out({N});
  if (N > 0) {
    std::memcpy(labels_out.data(), algo.labels().data(), N * sizeof(std::int32_t));
  }
  return wrapAsNumpy<std::int32_t>(std::move(labels_out));
}

static std::tuple<nb::ndarray<nb::numpy, std::int32_t, nb::ndim<1>>,
                  nb::ndarray<nb::numpy, float, nb::ndim<2>>, double, std::size_t, bool>
kmeans_binding(nb::ndarray<float, nb::ndim<2>, nb::c_contig, nb::device::cpu> data, std::size_t k,
               std::size_t max_iter, float tol, std::uint64_t seed, int n_jobs) {
  // Normalise worker count while GIL is still held; KMeans clamps 0 upward but the binding
  // accepts negative sentinels the same way dbscan_binding does.
  const std::size_t jobs = (n_jobs <= 0)
                               ? static_cast<std::size_t>(std::thread::hardware_concurrency())
                               : static_cast<std::size_t>(n_jobs);

  const std::size_t N = data.shape(0);
  const std::size_t D = data.shape(1);

  if (k == 0) {
    throw nb::value_error("k must be >= 1");
  }
  if (N > 0 && k > N) {
    throw nb::value_error("k must be <= number of rows");
  }
  if (D == 0 && N > 0) {
    throw nb::value_error("data must have d >= 1");
  }
  if (max_iter == 0) {
    throw nb::value_error("max_iter must be >= 1");
  }
  if (!(tol >= 0.0F)) {
    throw nb::value_error("tol must be >= 0");
  }

  // Borrow the numpy buffer directly when it is 32-byte aligned: @c KMeans treats @p X as
  // read-only and several AVX2 tiers assume 32-byte aligned loads on @p X through the
  // @c NDArray::isAligned<32>() gate. Borrowing a loosely-aligned numpy array would force the
  // dispatcher down fallback paths whose interactions are less thoroughly exercised, and in
  // practice we hit a heap-corruption tail on that path. Align-gated borrow captures the common
  // case (numpy contiguous arrays from @c make_blobs, @c astype, etc. on x86_64 glibc land on
  // 32-byte boundaries here) without exposing the unaligned fallback.
  constexpr std::size_t kAvx2Align = 32;
  const bool dataAligned = (reinterpret_cast<std::uintptr_t>(data.data()) % kAvx2Align) == 0;
  NDArray<float, 2> xOwned({0, 0});
  auto X = [&] {
    if (dataAligned) {
      return clustering::python::borrowFromNumpyContig<float>(data);
    }
    xOwned = NDArray<float, 2>({N, D});
    std::memcpy(xOwned.data(), data.data(), N * D * sizeof(float));
    return NDArray<float, 2, clustering::Layout::Contig>::borrow(xOwned.data(), {N, D});
  }();

  KMeans<float> algo(k, jobs);
  {
    nb::gil_scoped_release release;
    algo.run(X, max_iter, tol, seed);
  }
  // GIL re-acquired for the numpy allocation + wrap.

  // Detach labels and centroids into fresh owned buffers so the solver remains reusable and the
  // capsule has full ownership of the numpy-visible storage.
  NDArray<std::int32_t, 1> labels_out({N});
  NDArray<float, 2> centroids_out({k, D});
  std::memcpy(labels_out.data(), algo.labels().data(), N * sizeof(std::int32_t));
  std::memcpy(centroids_out.data(), algo.centroids().data(), k * D * sizeof(float));

  auto labels_np = wrapAsNumpy<std::int32_t>(std::move(labels_out));
  auto centroids_np = wrapAsNumpy<float>(std::move(centroids_out));

  return std::make_tuple(std::move(labels_np), std::move(centroids_np), algo.inertia(),
                         algo.nIter(), algo.converged());
}

static std::tuple<nb::ndarray<nb::numpy, std::int32_t, nb::ndim<1>>,
                  nb::ndarray<nb::numpy, float, nb::ndim<1>>, std::size_t>
hdbscan_binding(nb::ndarray<float, nb::ndim<2>, nb::c_contig, nb::device::cpu> data,
                std::size_t min_cluster_size, std::size_t min_samples, const std::string &method,
                int n_jobs, const std::string &min_samples_convention, const std::string &backend) {
  const std::size_t jobs = (n_jobs <= 0) ? std::size_t{0} : static_cast<std::size_t>(n_jobs);

  const std::size_t N = data.shape(0);
  const std::size_t D = data.shape(1);

  if (min_cluster_size < 2) {
    throw nb::value_error("min_cluster_size must be >= 2");
  }
  if (D == 0 && N > 0) {
    throw nb::value_error("data must have d >= 1");
  }
  if (N > 0 && N < min_cluster_size) {
    throw nb::value_error("data must have at least min_cluster_size rows");
  }

  using clustering::hdbscan::MinSamplesConvention;
  MinSamplesConvention convention{};
  if (min_samples_convention == "sklearn") {
    convention = MinSamplesConvention::kSklearn;
  } else if (min_samples_convention == "campello") {
    convention = MinSamplesConvention::kCampello;
  } else {
    throw nb::value_error("min_samples_convention must be 'sklearn' or 'campello'");
  }

  const std::size_t requested_min_samples = (min_samples == 0) ? min_cluster_size : min_samples;
  // Under the sklearn convention the query point counts as one of the neighbours, so the
  // non-self neighbour count the backend sees is requested - 1; that must be >= 1, hence
  // requested >= 2. Campello passes through untouched.
  const std::size_t effective_min_samples = (convention == MinSamplesConvention::kSklearn)
                                                ? requested_min_samples - 1
                                                : requested_min_samples;
  if (convention == MinSamplesConvention::kSklearn && requested_min_samples < 2) {
    throw nb::value_error(
        "min_samples must be >= 2 under the sklearn convention (the query point counts as one "
        "of the neighbours); pass min_samples_convention='campello' to use min_samples=1");
  }
  if (N > 0 && effective_min_samples >= N) {
    throw nb::value_error("effective min_samples must be < N after applying the convention");
  }
  constexpr std::size_t int32_max =
      static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max());
  if (N > int32_max) {
    throw nb::value_error("N exceeds int32 max");
  }

  ClusterSelectionMethod selectionMethod{};
  if (method == "eom") {
    selectionMethod = ClusterSelectionMethod::kEom;
  } else if (method == "leaf") {
    selectionMethod = ClusterSelectionMethod::kLeaf;
  } else {
    throw nb::value_error("method must be either 'eom' or 'leaf'");
  }

  // Borrow the numpy buffer directly when it is 32-byte aligned: several AVX2 tiers assume
  // 32-byte aligned loads on @p X through the @c NDArray::isAligned<32>() gate. Borrowing a
  // loosely-aligned numpy array would force the dispatcher down fallback paths whose
  // interactions are less thoroughly exercised. Align-gated borrow captures the common case
  // (numpy contiguous arrays from @c make_blobs, @c astype, etc. on x86_64 glibc land on
  // 32-byte boundaries here) without exposing the unaligned fallback.
  constexpr std::size_t kAvx2Align = 32;
  const bool dataAligned = (reinterpret_cast<std::uintptr_t>(data.data()) % kAvx2Align) == 0;
  NDArray<float, 2> xOwned({0, 0});
  auto X = [&] {
    if (dataAligned) {
      return clustering::python::borrowFromNumpyContig<float>(data);
    }
    xOwned = NDArray<float, 2>({N, D});
    std::memcpy(xOwned.data(), data.data(), N * D * sizeof(float));
    return NDArray<float, 2, clustering::Layout::Contig>::borrow(xOwned.data(), {N, D});
  }();

  NDArray<std::int32_t, 1> labels_out({N});
  NDArray<float, 1> scores_out({N});
  std::size_t n_clusters = 0;
  auto run_with = [&](auto &&algo) {
    {
      nb::gil_scoped_release release;
      algo.run(X);
    }
    if (N > 0) {
      std::memcpy(labels_out.data(), algo.labels().data(), N * sizeof(std::int32_t));
      std::memcpy(scores_out.data(), algo.outlierScores().data(), N * sizeof(float));
    }
    n_clusters = algo.nClusters();
  };
  if (backend == "auto") {
    HDBSCAN<float, clustering::hdbscan::AutoMstBackend<float>> algo(
        min_cluster_size, min_samples, selectionMethod, jobs, convention);
    run_with(algo);
  } else if (backend == "prim") {
    HDBSCAN<float, clustering::hdbscan::PrimMstBackend<float>> algo(
        min_cluster_size, min_samples, selectionMethod, jobs, convention);
    run_with(algo);
  } else if (backend == "boruvka") {
    HDBSCAN<float, clustering::hdbscan::BoruvkaMstBackend<float>> algo(
        min_cluster_size, min_samples, selectionMethod, jobs, convention);
    run_with(algo);
  } else if (backend == "nn_descent") {
    HDBSCAN<float, clustering::hdbscan::NnDescentMstBackend<float>> algo(
        min_cluster_size, min_samples, selectionMethod, jobs, convention);
    run_with(algo);
  } else {
    throw nb::value_error("backend must be one of {'auto', 'prim', 'boruvka', 'nn_descent'}");
  }

  auto labels_np = wrapAsNumpy<std::int32_t>(std::move(labels_out));
  auto scores_np = wrapAsNumpy<float>(std::move(scores_out));

  return std::make_tuple(std::move(labels_np), std::move(scores_np), n_clusters);
}

static nb::ndarray<nb::numpy, float, nb::ndim<2>>
roundtrip_zero_copy(nb::ndarray<float, nb::ndim<2>, nb::c_contig, nb::device::cpu> arr) {
  auto view = borrowFromNumpyContig<float>(arr);
  NDArray<float, 2> copy({view.dim(0), view.dim(1)});
  for (std::size_t i = 0; i < view.dim(0); ++i) {
    for (std::size_t j = 0; j < view.dim(1); ++j) {
      copy(i, j) = view(i, j);
    }
  }
  return wrapAsNumpy<float>(std::move(copy));
}

static nb::ndarray<nb::numpy, float, nb::ndim<2>> make_owned_array(std::size_t rows,
                                                                   std::size_t cols) {
  NDArray<float, 2> arr({rows, cols});
  for (std::size_t i = 0; i < rows; ++i) {
    for (std::size_t j = 0; j < cols; ++j) {
      arr(i, j) = static_cast<float>(i * cols + j);
    }
  }
  return wrapAsNumpy<float>(std::move(arr));
}

static std::tuple<std::size_t, std::size_t, bool>
inplace_increment_contig(nb::ndarray<float, nb::ndim<2>, nb::c_contig, nb::device::cpu> arr) {
  auto view = borrowFromNumpyContig<float>(arr);
  for (std::size_t i = 0; i < view.dim(0); ++i) {
    for (std::size_t j = 0; j < view.dim(1); ++j) {
      view(i, j) = view(i, j) + 1.0f;
    }
  }
  return {view.dim(0), view.dim(1), view.isMutable()};
}

static std::tuple<float, std::ptrdiff_t, std::ptrdiff_t>
sum_strided(nb::ndarray<float, nb::ndim<2>, nb::device::cpu> arr) {
  auto view = borrowFromNumpyStrided<float>(arr);
  float sum = 0.0f;
  for (std::size_t i = 0; i < view.dim(0); ++i) {
    for (std::size_t j = 0; j < view.dim(1); ++j) {
      sum += view(i, j);
    }
  }
  return {sum, view.strideAt(0), view.strideAt(1)};
}

static std::tuple<std::int64_t, std::int64_t, std::size_t>
probe_stride_units(nb::ndarray<float, nb::ndim<2>, nb::device::cpu> arr) {
  return {arr.stride(0), arr.stride(1), sizeof(float)};
}

static bool borrow_is_mutable_readonly(
    nb::ndarray<const float, nb::ndim<2>, nb::c_contig, nb::device::cpu> arr) {
  auto view = borrowFromNumpyContigReadOnly<float>(arr);
  return view.isMutable();
}

NB_MODULE(_clustering, m) {
  m.doc() = "DBSCAN clustering C++ extension";

  m.def("dbscan", &dbscan_binding, nb::arg("data"), nb::arg("eps"), nb::arg("min_pts"),
        nb::arg("n_jobs") = -1,
        "Run DBSCAN. Returns int32 label array of shape (N,). "
        "Noise points are labelled -1, clusters start at 0.");

  m.def("kmeans", &kmeans_binding, nb::arg("data"), nb::arg("k"), nb::arg("max_iter") = 300,
        nb::arg("tol") = 1e-4F, nb::arg("seed") = 0, nb::arg("n_jobs") = -1,
        "Run k-means with auto-dispatched algorithm and seeder. Returns a tuple "
        "(labels, centroids, inertia, n_iter, converged) where labels is int32 (N,), "
        "centroids is float32 (k, D), inertia is float64, n_iter is the iteration count, "
        "and converged is True iff the centroid shift fell at or below tol.");

  m.def("hdbscan", &hdbscan_binding, nb::arg("data"), nb::arg("min_cluster_size") = 5,
        nb::arg("min_samples") = 0, nb::arg("method") = "eom", nb::arg("n_jobs") = -1,
        nb::arg("min_samples_convention") = "sklearn", nb::arg("backend") = "auto",
        "Run HDBSCAN. Returns a tuple (labels, outlier_scores, n_clusters) where labels is "
        "int32 (N,) with -1 marking noise, outlier_scores is float32 (N,) in [0, 1], and "
        "n_clusters is the total cluster count. min_samples = 0 resolves to min_cluster_size "
        "at fit time. "
        "method must be 'eom' (excess-of-mass, the default) or 'leaf'. "
        "min_samples_convention selects the neighbour-count semantics: 'sklearn' (default) "
        "treats the query point itself as one of the min_samples neighbours (matches "
        "scikit-learn and the hdbscan package); 'campello' counts only non-self neighbours "
        "(literal Campello 2015 definition). The two differ by one neighbour and produce "
        "different MSTs on high-dimensional near-uniform data. "
        "backend selects the MST algorithm: 'auto' (default) dispatches on input shape "
        "per the rules documented on AutoMstBackend; 'prim' runs streaming dense Prim "
        "(exact, gated by a quadratic compute budget); 'boruvka' runs KDTree-accelerated "
        "Boruvka rounds (exact, scales to large n at low-to-moderate d); 'nn_descent' "
        "builds an approximate kNN graph then runs Kruskal with a connectivity fallback "
        "(approximate, best at large n and high d where the exact backends are too slow).");

  m.def("_roundtrip_zero_copy", &roundtrip_zero_copy, nb::arg("data"),
        "Test helper: borrow contiguous f32 array, copy into Owned NDArray, return as numpy.");
  m.def("_make_owned_array", &make_owned_array, nb::arg("rows"), nb::arg("cols"),
        "Test helper: allocate Owned NDArray<f32,2>, fill with i*cols+j, return as numpy.");
  m.def("_inplace_increment_contig", &inplace_increment_contig, nb::arg("data"),
        "Test helper: borrow contiguous f32 array and add 1.0 to every element in place.");
  m.def("_sum_strided", &sum_strided, nb::arg("data"),
        "Test helper: sum a possibly-strided f32 array via borrowFromNumpyStrided.");
  m.def("_probe_stride_units", &probe_stride_units, nb::arg("data"),
        "Test helper: report nanobind's raw stride values and sizeof(float).");
  m.def("_borrow_is_mutable_readonly", &borrow_is_mutable_readonly, nb::arg("data"),
        "Test helper: borrow read-only contiguous f32 array, return its m_mutable flag.");
}
