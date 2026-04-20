#pragma once

#include <concepts>
#include <cstddef>

#include "clustering/hdbscan/mst_output.h"
#include "clustering/math/thread.h"
#include "clustering/ndarray.h"

namespace clustering::hdbscan {

/**
 * @brief Contract for an MST backend satisfying the frozen @ref MstOutput shape.
 *
 * A backend is default-constructible and exposes a single @c run entry point that consumes the
 * input dataset, the @c minSamples parameter, and a worker-pool handle, and writes its result
 * into a caller-provided @ref MstOutput. Backends own their private scratch and may amortize
 * shape-indexed buffers across calls; per the HDBSCAN class invariant, data-dependent indices
 * (KDTree, kNN graph) are rebuilt per fit.
 *
 * The concept lives in its own header so that concrete backend implementations can depend on it
 * without pulling in the @ref HDBSCAN class template, and so the dispatcher can take the concept
 * as an input without introducing a header cycle.
 *
 * @tparam B Candidate backend type.
 * @tparam T Element type of the point cloud.
 */
template <class B, class T>
concept MstBackendStrategy = std::default_initializable<B> &&
                             requires(B &backend, const NDArray<T, 2> &X, std::size_t minSamples,
                                      math::Pool pool, MstOutput<T> &out) {
                               { backend.run(X, minSamples, pool, out) };
                             };

} // namespace clustering::hdbscan
