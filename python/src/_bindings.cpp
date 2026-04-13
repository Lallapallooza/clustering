#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <cstdint>
#include <cstring>
#include <thread>
#include <vector>

#include "clustering/dbscan.h"
#include "clustering/ndarray.h"

namespace nb = nanobind;

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

NB_MODULE(_clustering, m) {
  m.doc() = "DBSCAN clustering C++ extension";

  m.def("dbscan", &dbscan_binding, nb::arg("data"), nb::arg("eps"), nb::arg("min_pts"),
        nb::arg("n_jobs") = -1,
        "Run DBSCAN. Returns int32 label array of shape (N,). "
        "Noise points are labelled -1, clusters start at 0.");
}
