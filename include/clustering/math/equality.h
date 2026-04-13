#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <utility>

#include "clustering/ndarray.h"

namespace clustering::math {

/**
 * @brief Element-wise exact equality between two NDArrays of matching shape.
 *
 * Shapes are compared dimension-by-dimension before any elements are read; a shape mismatch
 * reports @c false without touching element storage. Empty shapes (any axis zero) compare
 * equal vacuously.
 *
 * @tparam T  Element type of both arrays (must match).
 * @tparam N  Rank of both arrays (must match; enforced by the template signature).
 * @tparam LA Layout tag of @p a.
 * @tparam LB Layout tag of @p b.
 * @param a Left-hand operand.
 * @param b Right-hand operand.
 * @return @c true iff @c a.dim(k) == b.dim(k) for every axis and every pairwise element is
 *         bitwise-equal under @c operator==.
 */
template <class T, std::size_t N, Layout LA, Layout LB>
bool arrayEqual(const NDArray<T, N, LA> &a, const NDArray<T, N, LB> &b) noexcept {
  std::array<std::size_t, N> shape{};
  std::size_t total = 1;
  for (std::size_t k = 0; k < N; ++k) {
    if (a.dim(k) != b.dim(k)) {
      return false;
    }
    shape[k] = a.dim(k);
    total *= shape[k];
  }
  if (total == 0) {
    return true;
  }
  std::array<std::size_t, N> idx{};
  for (std::size_t flat = 0; flat < total; ++flat) {
    const bool equal = [&]<std::size_t... Ks>(std::index_sequence<Ks...>) {
      return a(idx[Ks]...) == b(idx[Ks]...);
    }(std::make_index_sequence<N>{});
    if (!equal) {
      return false;
    }
    for (std::size_t k = N; k-- > 0;) {
      if (++idx[k] < shape[k]) {
        break;
      }
      idx[k] = 0;
    }
  }
  return true;
}

/**
 * @brief Element-wise approximate equality with NumPy-style asymmetric tolerance.
 *
 * True when @c |a_i - b_i| <= atol + rtol * |b_i| for every element. @p b sits on the
 * right-hand side of the tolerance term intentionally, matching @c numpy.allclose; the
 * relation is not symmetric in the two arguments.
 *
 * @tparam T  Element type; must be a floating-point type.
 * @tparam N  Rank of both arrays.
 * @tparam LA Layout tag of @p a.
 * @tparam LB Layout tag of @p b.
 * @param a    Left-hand operand.
 * @param b    Right-hand operand; @c rtol scales against @c |b|.
 * @param rtol Relative tolerance, defaults to @c 1e-5.
 * @param atol Absolute tolerance, defaults to @c 1e-8.
 * @return @c true iff shapes match and every element pair is within tolerance.
 */
template <class T, std::size_t N, Layout LA, Layout LB>
bool allClose(const NDArray<T, N, LA> &a, const NDArray<T, N, LB> &b, T rtol = T(1e-5),
              T atol = T(1e-8)) noexcept {
  std::array<std::size_t, N> shape{};
  std::size_t total = 1;
  for (std::size_t k = 0; k < N; ++k) {
    if (a.dim(k) != b.dim(k)) {
      return false;
    }
    shape[k] = a.dim(k);
    total *= shape[k];
  }
  if (total == 0) {
    return true;
  }
  std::array<std::size_t, N> idx{};
  for (std::size_t flat = 0; flat < total; ++flat) {
    auto [va, vb] = [&]<std::size_t... Ks>(std::index_sequence<Ks...>) {
      return std::pair<T, T>{a(idx[Ks]...), b(idx[Ks]...)};
    }(std::make_index_sequence<N>{});
    const T diff = std::fabs(va - vb);
    const T tol = atol + (rtol * std::fabs(vb));
    if (!(diff <= tol)) {
      return false;
    }
    for (std::size_t k = N; k-- > 0;) {
      if (++idx[k] < shape[k]) {
        break;
      }
      idx[k] = 0;
    }
  }
  return true;
}

} // namespace clustering::math
