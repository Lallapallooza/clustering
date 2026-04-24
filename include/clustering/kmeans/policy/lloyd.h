#pragma once

#include <concepts>
#include <cstddef>
#include <cstdint>

#include "clustering/math/thread.h"
#include "clustering/ndarray.h"

namespace clustering::kmeans {

/**
 * @brief Contract for the Lloyd driver that `KMeans<T>` delegates to.
 *
 * A @c LloydStrategy owns its private scratch, takes seeded centroids as input, runs the
 * iteration loop, and writes final labels, inertia, iteration count, and convergence state
 * into caller-provided output slots.
 *
 * @tparam A Candidate Lloyd policy type.
 * @tparam T Element type of the point cloud.
 */
template <class A, class T>
concept LloydStrategy =
    std::default_initializable<A> &&
    requires(A &algo, const NDArray<T, 2> &X, NDArray<T, 2> &centroids, std::size_t k,
             std::size_t maxIter, T tol, math::Pool pool, NDArray<std::int32_t, 1> &labels,
             double &inertia, std::size_t &nIter, bool &converged) {
      { algo.run(X, centroids, k, maxIter, tol, pool, labels, inertia, nIter, converged) };
    };

} // namespace clustering::kmeans
