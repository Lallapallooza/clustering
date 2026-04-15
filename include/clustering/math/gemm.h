#pragma once

#include <cstddef>
#include <type_traits>

#include "clustering/always_assert.h"
#include "clustering/math/defaults.h"
#include "clustering/math/thread.h"
#include "clustering/ndarray.h"

namespace clustering::math {

/**
 * @brief One-shot dense matrix-matrix multiply: @c C := alpha * A * B + beta * C.
 *
 * @tparam T Element type (@c float or @c double).
 * @tparam LA Layout tag of A; CTAD-resolved at the call site so
 *         @c gemm(A, B.t(), C, pool) binds without explicit template arguments.
 * @tparam LB Layout tag of B.
 * @tparam Backend Backend tag; defaulted to @c defaults::Backend. Swap project-wide via
 *         @c CLUSTERING_MATH_DEFAULT_BACKEND or per call site by naming the argument.
 * @param A Input matrix (M x K).
 * @param B Input matrix (K x N).
 * @param C Output matrix (M x N). @c isMutable() must be true; enforced by an always-assert that
 *        fires in release too.
 * @param pool Parallelism injection; forwarded to the backend.
 * @param alpha Scalar multiplier on @c A*B.
 * @param beta  Scalar multiplier on the prior @c C.
 */
template <class T, Layout LA, Layout LB, class Backend = defaults::Backend>
void gemm(const NDArray<T, 2, LA> &A, const NDArray<T, 2, LB> &B, NDArray<T, 2> &C, Pool pool,
          T alpha = T{1}, T beta = T{0}) {
  // Integer NDArrays are permitted as label / index storage but must never reach numeric math:
  // the microkernel pack layout keys on kKernelMr<T> / kKernelNr<T> which default to 0 for any
  // unspecialized T, so without this gate integer T would divide by zero inside gemm_pack.h.
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                "gemm<T> requires T to be float or double");
  // Mutability check precedes descriptor extraction: describeMatrixMut's debug-only assert would
  // otherwise mask the violation in release builds.
  CLUSTERING_ALWAYS_ASSERT(C.isMutable());

  CLUSTERING_ALWAYS_ASSERT(A.dim(1) == B.dim(0));
  CLUSTERING_ALWAYS_ASSERT(A.dim(0) == C.dim(0));
  CLUSTERING_ALWAYS_ASSERT(B.dim(1) == C.dim(1));

  if (C.dim(0) == 0 || C.dim(1) == 0) {
    return;
  }

  Backend::template run<T, LA, LB>(A, B, C, pool, alpha, beta);
}

} // namespace clustering::math
