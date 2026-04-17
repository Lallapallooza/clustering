#pragma once

#include <concepts>
#include <cstddef>
#include <cstdint>

#include "clustering/math/thread.h"
#include "clustering/ndarray.h"

namespace clustering::kmeans {

/**
 * @brief Contract for the seeder that produces initial centroids for the Lloyd driver.
 *
 * The seeder owns its private scratch and writes the @c k initial centroid rows directly into
 * the caller-provided centroid matrix. No auxiliary output buffers cross the seeder boundary;
 * the Lloyd driver owns its own per-point distance scratch.
 *
 * @tparam S Candidate seeder policy type.
 * @tparam T Element type of the point cloud.
 */
template <class S, class T>
concept SeederStrategy =
    std::default_initializable<S> &&
    requires(S &seeder, const NDArray<T, 2> &X, std::size_t k, std::uint64_t seed, math::Pool pool,
             NDArray<T, 2> &outCentroids) {
      { seeder.run(X, k, seed, pool, outCentroids) };
    };

} // namespace clustering::kmeans
