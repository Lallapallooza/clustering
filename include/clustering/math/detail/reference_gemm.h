#pragma once

#include <cstddef>
#include <vector>

#include "clustering/math/detail/gemm_outer.h"
#include "clustering/math/detail/matrix_desc.h"
#include "clustering/math/thread.h"
#include "clustering/ndarray.h"

namespace clustering::math::detail {

/**
 * @brief Backend tag for the hand-rolled Goto-style reference GEMM.
 *
 * Serves as the default @c Backend template argument of the public one-shot @c gemm entry.
 * Concrete backends are expected to expose a static @c run template with the same signature;
 * swapping in (for example) an @c OpenBlasGemm adapter is a one-word change at the call site.
 *
 * @c GemmPlan does not route through @c Backend::run — it owns the pre-packed B buffer and
 * calls @c gemmRunPrepacked directly. The shim is used only by the one-shot entry.
 */
struct ReferenceGemm {
  /**
   * @brief One-shot reference GEMM: allocates per-call scratch arenas and dispatches to
   *        @c gemmRunReference.
   *
   * @tparam T  Element type (@c float or @c double).
   * @tparam LA Layout tag of A (@c Contig or @c MaybeStrided).
   * @tparam LB Layout tag of B.
   * @param A Input matrix A (M x K).
   * @param B Input matrix B (K x N).
   * @param C Output matrix C (M x N); must be mutable.
   * @param pool Parallelism injection; accepted but unused on the serial path.
   * @param alpha Scalar multiplier on @c A*B.
   * @param beta  Scalar multiplier on the prior @c C.
   */
  template <class T, ::clustering::Layout LA, ::clustering::Layout LB>
  static void run(const ::clustering::NDArray<T, 2, LA> &A,
                  const ::clustering::NDArray<T, 2, LB> &B, ::clustering::NDArray<T, 2> &C,
                  ::clustering::math::Pool pool, T alpha, T beta) {
    if (C.dim(0) == 0 || C.dim(1) == 0) {
      return;
    }
    std::vector<T, ::clustering::detail::AlignedAllocator<T, 32>> apArena(kMc<T> * kKc<T>, T{0});
    std::vector<T, ::clustering::detail::AlignedAllocator<T, 32>> bpArena(kKc<T> * kNc<T>, T{0});
    auto Ad = ::clustering::detail::describeMatrix(A);
    auto Bd = ::clustering::detail::describeMatrix(B);
    auto Cd = ::clustering::detail::describeMatrixMut(C);
    gemmRunReference<T>(Ad, Bd, Cd, alpha, beta, apArena.data(), bpArena.data(), pool);
  }
};

} // namespace clustering::math::detail
