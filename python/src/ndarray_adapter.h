#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>

#include "clustering/ndarray.h"

namespace clustering::python {

namespace nb = nanobind;

/**
 * @brief Borrow a contiguous numpy array as a @c Layout::Contig NDArray view.
 *
 * Zero-copy: the returned NDArray shares the numpy buffer. The caller is responsible for
 * keeping the numpy array alive for the view's lifetime; nanobind already does so for the
 * duration of the function call when @p arr is a function parameter.
 *
 * The numpy array's writability is decided at the type level by nanobind: a parameter of type
 * @c nb::ndarray<T, ...> only binds writable numpy arrays, while @c nb::ndarray<const T, ...>
 * (or with @c nb::ro) only binds read-only arrays. The adapter mirrors that into NDArray's
 * @c m_mutable via the @c borrow(T*) vs. @c borrow(const T*) overload.
 *
 * @tparam T Element type, @c float or @c double.
 * @param arr Contiguous f32/f64 rank-2 numpy array on CPU.
 * @return Borrowed @c Layout::Contig NDArray viewing @p arr.
 */
template <class T>
inline NDArray<T, 2, Layout::Contig>
borrowFromNumpyContig(nb::ndarray<T, nb::ndim<2>, nb::c_contig, nb::device::cpu> arr) noexcept {
  std::array<std::size_t, 2> shape{arr.shape(0), arr.shape(1)};
  return NDArray<T, 2, Layout::Contig>::borrow(arr.data(), shape);
}

/**
 * @brief Borrow a read-only contiguous numpy array as a @c Layout::Contig NDArray view.
 *
 * Selected when the binding declares @c nb::ndarray<const T, ...>; the resulting NDArray has
 * @c m_mutable = false and asserts on writes through @c operator() / @c Accessor in debug.
 */
template <class T>
inline NDArray<T, 2, Layout::Contig> borrowFromNumpyContigReadOnly(
    nb::ndarray<const T, nb::ndim<2>, nb::c_contig, nb::device::cpu> arr) noexcept {
  std::array<std::size_t, 2> shape{arr.shape(0), arr.shape(1)};
  return NDArray<T, 2, Layout::Contig>::borrow(arr.data(), shape);
}

/**
 * @brief Borrow a possibly strided numpy array as a @c Layout::MaybeStrided NDArray view.
 *
 * Nanobind reports strides in elements (the DLPack stride is multiplied by @c sizeof(Scalar)
 * inside @c nb::ndarray::operator()), so the adapter feeds them to the element-stride
 * @c borrow(T*, shape, strides) factory directly. No byte-divisibility check is needed at this
 * boundary; @c borrowBytes is only relevant when the source genuinely measures strides in bytes.
 */
template <class T>
inline NDArray<T, 2, Layout::MaybeStrided>
borrowFromNumpyStrided(nb::ndarray<T, nb::ndim<2>, nb::device::cpu> arr) noexcept {
  std::array<std::size_t, 2> shape{arr.shape(0), arr.shape(1)};
  std::array<std::ptrdiff_t, 2> strides{static_cast<std::ptrdiff_t>(arr.stride(0)),
                                        static_cast<std::ptrdiff_t>(arr.stride(1))};
  return NDArray<T, 2, Layout::MaybeStrided>::borrow(arr.data(), shape, strides);
}

/**
 * @brief Borrow a read-only possibly strided numpy array as a @c Layout::MaybeStrided NDArray.
 */
template <class T>
inline NDArray<T, 2, Layout::MaybeStrided>
borrowFromNumpyStridedReadOnly(nb::ndarray<const T, nb::ndim<2>, nb::device::cpu> arr) noexcept {
  std::array<std::size_t, 2> shape{arr.shape(0), arr.shape(1)};
  std::array<std::ptrdiff_t, 2> strides{static_cast<std::ptrdiff_t>(arr.stride(0)),
                                        static_cast<std::ptrdiff_t>(arr.stride(1))};
  return NDArray<T, 2, Layout::MaybeStrided>::borrow(arr.data(), shape, strides);
}

/**
 * @brief Wrap an Owned NDArray as a numpy array with a capsule keeping the storage alive.
 *
 * The NDArray is moved onto the heap; the returned @c nb::ndarray carries a capsule whose
 * deleter destroys the heap NDArray when Python's GC releases the last reference. No element
 * copy occurs: the numpy view points at the moved-into NDArray's buffer.
 *
 * Only @c Layout::Contig is supported because numpy's default constructor expects row-major
 * contiguous storage; pass a contiguous source or call @c .contiguous() first. The NDArray
 * must be @c Owned: the capsule deleter destroys the heap @c NDArray but cannot free the
 * buffer underlying a @c Borrowed view, so passing a borrow would leave the numpy array
 * pointing at storage whose lifetime the capsule cannot extend.
 *
 * @tparam T Element type, @c float or @c double.
 * @param arr Owned contiguous NDArray to surrender to Python.
 * @return Numpy ndarray viewing the heap-moved NDArray's buffer.
 */
template <class T>
inline nb::ndarray<nb::numpy, T, nb::ndim<2>> wrapAsNumpy(NDArray<T, 2, Layout::Contig> arr) {
  assert(arr.isOwned() &&
         "wrapAsNumpy requires an Owned NDArray; the capsule cannot extend a Borrowed buffer's "
         "lifetime");
  auto heap = std::make_unique<NDArray<T, 2, Layout::Contig>>(std::move(arr));
  nb::capsule owner(
      heap.get(), [](void *p) noexcept { delete static_cast<NDArray<T, 2, Layout::Contig> *>(p); });
  auto *raw = heap.release();
  std::size_t shape[2] = {raw->dim(0), raw->dim(1)};
  return nb::ndarray<nb::numpy, T, nb::ndim<2>>(raw->data(), 2, shape, owner);
}

} // namespace clustering::python
