#pragma once

#include <cstddef>

#include "clustering/math/detail/gemm_kernel_scalar.h"
#include "clustering/math/detail/matrix_desc.h"

// Scalar element access via MatrixDesc strides handles both Contig and MaybeStrided sources
// uniformly. SIMD packers (loadu-based, possibly aligned-fast-path) are a future optimization;
// for the reference path the packer is bandwidth-bound at clustering shapes anyway.

namespace clustering::math::detail {

/**
 * @brief Pack an @c Mc x @c Kc block of @c A into the microkernel-friendly @c Ap layout.
 *
 * Output is a sequence of @c ceil(mc / Mr) panels. Panel @c p is @c Mr rows by @c kc columns;
 * element @c (r, k) (with @c r the panel-local row and @c k the inner-dim index) lands at
 * @c apOut[p*Mr*kc + k*Mr + r]. The last panel zero-pads any rows beyond @c (mc - p*Mr) for
 * every @c k in @c [0, kc), so the kernel's accumulator math remains valid for tail tiles.
 *
 * @tparam T  Element type.
 * @param Ad  Source matrix descriptor (@c MatrixDescC<T>); arbitrary strides are supported.
 * @param ic  Starting row in @c Ad of the block to pack.
 * @param mc  Row count of the block (last panel zero-pads if @c mc % Mr != 0).
 * @param pc  Starting column in @c Ad of the block.
 * @param kc  Column count of the block.
 * @param apOut Destination buffer of capacity @c ceil(mc/Mr) * Mr * kc, 32-byte aligned.
 */
template <class T>
void packA(const ::clustering::detail::MatrixDescC<T> &Ad, std::size_t ic, std::size_t mc,
           std::size_t pc, std::size_t kc, T *apOut) noexcept {
  constexpr std::size_t kMr = kKernelMr<T>;

  const std::ptrdiff_t aRowStride = Ad.rowStride;
  const std::ptrdiff_t aColStride = Ad.colStride;
  const T *aBase = Ad.ptr;

  const std::size_t panels = (mc + kMr - 1) / kMr;
  for (std::size_t p = 0; p < panels; ++p) {
    T *panelOut = apOut + (p * kMr * kc);
    const std::size_t rBase = p * kMr;
    const std::size_t rValid = (rBase + kMr <= mc) ? kMr : (mc - rBase);

    for (std::size_t k = 0; k < kc; ++k) {
      T *col = panelOut + (k * kMr);
      const std::ptrdiff_t srcCol = (static_cast<std::ptrdiff_t>(pc + k)) * aColStride;
      for (std::size_t r = 0; r < rValid; ++r) {
        const std::ptrdiff_t srcRow = (static_cast<std::ptrdiff_t>(ic + rBase + r)) * aRowStride;
        col[r] = aBase[srcRow + srcCol];
      }
      for (std::size_t r = rValid; r < kMr; ++r) {
        col[r] = T{0};
      }
    }
  }
}

/**
 * @brief Pack a @c Kc x @c Nc block of @c B into the microkernel-friendly @c Bp layout.
 *
 * Output is a sequence of @c ceil(nc / Nr) panels. Panel @c p is @c kc rows by @c Nr columns;
 * element @c (k, c) (with @c c the panel-local column) lands at
 * @c bpOut[p*kc*Nr + k*Nr + c]. The last panel zero-pads any columns beyond @c (nc - p*Nr).
 *
 * @tparam T  Element type.
 * @param Bd  Source matrix descriptor (@c MatrixDescC<T>); arbitrary strides are supported.
 * @param pc  Starting row in @c Bd of the block to pack.
 * @param kc  Row count of the block.
 * @param jc  Starting column in @c Bd of the block.
 * @param nc  Column count of the block (last panel zero-pads if @c nc % Nr != 0).
 * @param bpOut Destination buffer of capacity @c ceil(nc/Nr) * kc * Nr, 32-byte aligned.
 */
template <class T>
void packB(const ::clustering::detail::MatrixDescC<T> &Bd, std::size_t pc, std::size_t kc,
           std::size_t jc, std::size_t nc, T *bpOut) noexcept {
  constexpr std::size_t kNr = kKernelNr<T>;

  const std::ptrdiff_t bRowStride = Bd.rowStride;
  const std::ptrdiff_t bColStride = Bd.colStride;
  const T *bBase = Bd.ptr;

  const std::size_t panels = (nc + kNr - 1) / kNr;
  for (std::size_t p = 0; p < panels; ++p) {
    T *panelOut = bpOut + (p * kc * kNr);
    const std::size_t cBase = p * kNr;
    const std::size_t cValid = (cBase + kNr <= nc) ? kNr : (nc - cBase);

    for (std::size_t k = 0; k < kc; ++k) {
      T *row = panelOut + (k * kNr);
      const std::ptrdiff_t srcRow = (static_cast<std::ptrdiff_t>(pc + k)) * bRowStride;
      for (std::size_t c = 0; c < cValid; ++c) {
        const std::ptrdiff_t srcCol = (static_cast<std::ptrdiff_t>(jc + cBase + c)) * bColStride;
        row[c] = bBase[srcRow + srcCol];
      }
      for (std::size_t c = cValid; c < kNr; ++c) {
        row[c] = T{0};
      }
    }
  }
}

} // namespace clustering::math::detail
