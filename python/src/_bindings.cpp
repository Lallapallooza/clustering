#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/tuple.h>

#include <cstdint>
#include <cstring>
#include <thread>
#include <tuple>
#include <vector>

#include "clustering/dbscan.h"
#include "clustering/kmeans.h"
#include "clustering/ndarray.h"

#include "ndarray_adapter.h"

namespace nb = nanobind;
using clustering::DBSCAN;
using clustering::KMeans;
using clustering::NDArray;
using clustering::python::borrowFromNumpyContig;
using clustering::python::borrowFromNumpyContigReadOnly;
using clustering::python::borrowFromNumpyStrided;
using clustering::python::wrapAsNumpy;

static nb::ndarray<nb::numpy, int32_t, nb::ndim<1>>
dbscan_binding(nb::ndarray<float, nb::ndim<2>, nb::c_contig, nb::device::cpu> data, float eps,
               int min_pts, int n_jobs) {
  // Normalise n_jobs while GIL is still held (safe numpy access window).
  size_t jobs = (n_jobs <= 0) ? static_cast<size_t>(std::thread::hardware_concurrency())
                              : static_cast<size_t>(n_jobs);

  const size_t N = data.shape(0);
  const size_t D = data.shape(1);

  // Copy numpy data into NDArray<float,2> while GIL is held.
  NDArray<float, 2> points({N, D});
  std::memcpy(points.data(), data.data(), N * D * sizeof(float));

  // Allocate output buffer (owned by a capsule).
  int32_t *out = new int32_t[N];

  {
    // Release GIL for the C++ computation.
    nb::gil_scoped_release release;

    DBSCAN<float> algo(points, eps, static_cast<size_t>(min_pts), jobs);
    algo.run();

    const auto &labels = algo.labels();
    for (size_t i = 0; i < N; ++i) {
      out[i] = labels[i].load(std::memory_order_relaxed);
    }
  }
  // GIL re-acquired here.

  nb::capsule owner(out, [](void *p) noexcept { delete[] static_cast<int32_t *>(p); });

  size_t shape[1] = {N};
  return nb::ndarray<nb::numpy, int32_t, nb::ndim<1>>(out, 1, shape, owner);
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

  NDArray<float, 2> X({N, D});
  std::memcpy(X.data(), data.data(), N * D * sizeof(float));

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
