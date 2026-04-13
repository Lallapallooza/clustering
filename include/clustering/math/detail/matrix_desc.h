#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <initializer_list>

#include "clustering/ndarray.h"

namespace clustering::detail {

/**
 * @brief POD descriptor for a rank-2 matrix view consumed by microkernel inner loops.
 *
 * Field order matters: kernels may take @c MatrixDesc by value and expect the ABI to match
 * across translation units. Extending this struct is additive; reordering is a break.
 *
 * @tparam T Element type. Use the const-qualified alias @c MatrixDescC<T> for read-only views.
 */
template <class T> struct MatrixDesc {
  T *ptr;
  std::size_t rows;
  std::size_t cols;
  std::ptrdiff_t rowStride;
  std::ptrdiff_t colStride;
  std::size_t alignment;
  bool isContiguous;
};

/**
 * @brief Const-polarity alias of @c MatrixDesc; the element type is @c const T.
 */
template <class T> using MatrixDescC = MatrixDesc<const T>;

namespace matrix_desc_impl {

/**
 * @brief Largest power of two in {64, 32, 16, 8, 4, 1} dividing @p addr.
 *
 * Returned value is the alignment granularity a microkernel can assume when issuing SIMD
 * loads through @c ptr. A null pointer reports 1 ("no alignment guarantee"): microkernels
 * that branch on @c alignment >= 32 must either short-circuit on an empty @c MatrixDesc
 * (@c rows*cols == 0) or treat the null case as scalar-path, never as aligned.
 */
inline std::size_t largestPow2Alignment(std::uintptr_t addr) noexcept {
  if (addr == 0) {
    return 1;
  }
  for (const std::size_t a :
       {std::size_t{64}, std::size_t{32}, std::size_t{16}, std::size_t{8}, std::size_t{4}}) {
    if ((addr % a) == 0) {
      return a;
    }
  }
  return 1;
}

} // namespace matrix_desc_impl

/**
 * @brief Extracts a read-only microkernel descriptor from a rank-2 NDArray.
 *
 * @tparam T Element type of the source array.
 * @tparam L Layout tag of the source array; @c Layout::Contig or @c Layout::MaybeStrided.
 * @param a Source array. Storage (Owned/Borrowed), mutability, and contiguity are preserved in
 *          the descriptor's fields; the source must outlive the returned @c MatrixDescC.
 * @return @c MatrixDescC<T> with @c ptr aliasing @c a.data().
 */
template <class T, Layout L>
inline MatrixDescC<T> describeMatrix(const NDArray<T, 2, L> &a) noexcept {
  MatrixDescC<T> d{};
  d.ptr = a.data();
  d.rows = a.dim(0);
  d.cols = a.dim(1);
  d.rowStride = a.strideAt(0);
  d.colStride = a.strideAt(1);
  d.alignment = matrix_desc_impl::largestPow2Alignment(reinterpret_cast<std::uintptr_t>(d.ptr));
  d.isContiguous = a.isContiguous();
  return d;
}

/**
 * @brief Extracts a mutable microkernel descriptor from a rank-2 NDArray.
 *
 * Asserts in debug when @p a is a read-only borrow: writes through the returned @c ptr would
 * violate the borrow's const contract.
 *
 * @tparam T Element type of the source array.
 * @tparam L Layout tag of the source array; @c Layout::Contig or @c Layout::MaybeStrided.
 * @param a Source array. Must be mutable (Owned or a mutable Borrowed view).
 * @return @c MatrixDesc<T> with @c ptr aliasing @c a.data().
 */
template <class T, Layout L> inline MatrixDesc<T> describeMatrixMut(NDArray<T, 2, L> &a) noexcept {
  assert(a.isMutable() && "describeMatrixMut requires a mutable array");
  MatrixDesc<T> d{};
  d.ptr = a.data();
  d.rows = a.dim(0);
  d.cols = a.dim(1);
  d.rowStride = a.strideAt(0);
  d.colStride = a.strideAt(1);
  d.alignment = matrix_desc_impl::largestPow2Alignment(reinterpret_cast<std::uintptr_t>(d.ptr));
  d.isContiguous = a.isContiguous();
  return d;
}

} // namespace clustering::detail
