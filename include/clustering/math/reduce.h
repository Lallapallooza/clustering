#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <span>
#include <utility>
#include <vector>

#include "clustering/ndarray.h"

namespace clustering::math {

/**
 * @brief Naive single-pass sum of a rank-1 array.
 *
 * Straight accumulation with no compensation. Accurate enough when all magnitudes are close and
 * @c n is modest; use @ref sumKahan when a large offset plus small increments risks absorption.
 *
 * @tparam T Element type; @c float or @c double.
 * @tparam L Layout tag; contiguous and strided inputs are both accepted via @c x(i).
 * @param x Rank-1 array; empty input returns @c T(0).
 * @return Sum of the elements, or @c T(0) for an empty input.
 */
template <class T, Layout L> inline T sum(const NDArray<T, 1, L> &x) noexcept {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                "sum<T> requires T to be float or double");
  const std::size_t n = x.dim(0);
  T s = T{0};
  for (std::size_t i = 0; i < n; ++i) {
    s += x(i);
  }
  return s;
}

/**
 * @brief Kahan-compensated sum of a rank-1 array.
 *
 * Classical Kahan summation: a running compensation scalar @c c absorbs the low-order bits lost
 * when a small addend is added to a large running total. Recovers full precision on cases where
 * a dominant term (e.g. @c 1e9) is combined with many small ones (e.g. @c 1e-3).
 *
 * @tparam T Element type; @c float or @c double.
 * @tparam L Layout tag.
 * @param x Rank-1 array; empty input returns @c T(0).
 * @return Compensated sum of the elements.
 */
template <class T, Layout L> inline T sumKahan(const NDArray<T, 1, L> &x) noexcept {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                "sumKahan<T> requires T to be float or double");
  const std::size_t n = x.dim(0);
  T s = T{0};
  T c = T{0};
  for (std::size_t i = 0; i < n; ++i) {
    const T y = x(i) - c;
    const T t = s + y;
    c = (t - s) - y;
    s = t;
  }
  return s;
}

/**
 * @brief Mean and population variance via Welford's online recurrence.
 *
 * Returns @c (mean, variance) where @c variance = sum((x_i - mean)^2) / n (population form, no
 * Bessel correction). Welford avoids the @c E[x^2] - E[x]^2 cancellation trap that collapses to
 * zero for inputs like @c [1e9+1, 1e9+2, 1e9+3] in f64: the mean update absorbs the offset so
 * the squared-deviation accumulator only ever sees small residuals.
 *
 * Empty input yields @c {T(0), T(0)}: a defined return rather than UB, even though the variance
 * of an empty set is mathematically vacuous.
 *
 * @tparam T Element type; @c float or @c double.
 * @tparam L Layout tag.
 * @param x Rank-1 array.
 * @return @c std::pair<T,T> of (mean, population-variance); @c {0,0} on empty input.
 */
template <class T, Layout L> inline std::pair<T, T> meanVar(const NDArray<T, 1, L> &x) noexcept {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                "meanVar<T> requires T to be float or double");
  const std::size_t n = x.dim(0);
  if (n == 0) {
    return {T{0}, T{0}};
  }
  T mean = T{0};
  T m2 = T{0};
  for (std::size_t i = 0; i < n; ++i) {
    const T xi = x(i);
    const T delta = xi - mean;
    mean += delta / static_cast<T>(i + 1);
    const T delta2 = xi - mean;
    m2 += delta * delta2;
  }
  return {mean, m2 / static_cast<T>(n)};
}

/**
 * @brief Index of the first minimum in a rank-1 array.
 *
 * Strict @c < comparison during the scan: on ties the earliest index wins. Asserts the input is
 * non-empty; an empty argmin has no meaningful answer.
 *
 * @tparam T Element type; @c float or @c double.
 * @tparam L Layout tag.
 * @param x Non-empty rank-1 array.
 * @return Index of the first element attaining the minimum value.
 */
template <class T, Layout L> inline std::size_t argmin(const NDArray<T, 1, L> &x) noexcept {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                "argmin<T> requires T to be float or double");
  const std::size_t n = x.dim(0);
  assert(n > 0 && "argmin requires non-empty input");
  std::size_t bestIdx = 0;
  T bestVal = x(0);
  for (std::size_t i = 1; i < n; ++i) {
    const T v = x(i);
    if (v < bestVal) {
      bestVal = v;
      bestIdx = i;
    }
  }
  return bestIdx;
}

/**
 * @brief Index of the first maximum in a rank-1 array.
 *
 * Mirror of @ref argmin with strict @c > comparison. Asserts non-empty input.
 *
 * @tparam T Element type; @c float or @c double.
 * @tparam L Layout tag.
 * @param x Non-empty rank-1 array.
 * @return Index of the first element attaining the maximum value.
 */
template <class T, Layout L> inline std::size_t argmax(const NDArray<T, 1, L> &x) noexcept {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                "argmax<T> requires T to be float or double");
  const std::size_t n = x.dim(0);
  assert(n > 0 && "argmax requires non-empty input");
  std::size_t bestIdx = 0;
  T bestVal = x(0);
  for (std::size_t i = 1; i < n; ++i) {
    const T v = x(i);
    if (v > bestVal) {
      bestVal = v;
      bestIdx = i;
    }
  }
  return bestIdx;
}

/**
 * @brief Indices of the top-@p k largest values, written in descending value order.
 *
 * Stages @c (value, index) pairs and calls @c std::partial_sort_copy with a comparator that
 * orders by value descending; equal values tie-break by index ascending so the output is stable
 * and deterministic. The output span is an @c std::span<std::size_t> rather than an
 * @c NDArray<std::size_t,1> because @c NDArray's element type must be @c float or @c double.
 *
 * @tparam T Element type; @c float or @c double.
 * @tparam L Layout tag of the input.
 * @param x Rank-1 array of values to rank.
 * @param k Number of indices to emit; must satisfy @c k <= x.dim(0).
 * @param outIdx Output buffer of exactly @c k positions; filled with indices sorted by
 *        descending value.
 */
template <class T, Layout L>
inline void topk(const NDArray<T, 1, L> &x, std::size_t k, std::span<std::size_t> outIdx) noexcept {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                "topk<T> requires T to be float or double");
  assert(outIdx.size() == k && "topk requires outIdx.size() == k");
  if (k == 0) {
    return;
  }
  const std::size_t n = x.dim(0);
  assert(k <= n && "topk requires k <= x.dim(0)");

  std::vector<std::pair<T, std::size_t>> staged;
  staged.reserve(n);
  for (std::size_t i = 0; i < n; ++i) {
    staged.emplace_back(x(i), i);
  }

  std::vector<std::pair<T, std::size_t>> top(k);
  const auto cmp = [](const std::pair<T, std::size_t> &a,
                      const std::pair<T, std::size_t> &b) noexcept {
    // Primary: larger value first. Secondary on ties: smaller index first so the output is
    // deterministic when duplicate values collide.
    if (a.first != b.first) {
      return a.first > b.first;
    }
    return a.second < b.second;
  };
  std::partial_sort_copy(staged.begin(), staged.end(), top.begin(), top.end(), cmp);

  for (std::size_t i = 0; i < k; ++i) {
    outIdx[i] = top[i].second;
  }
}

} // namespace clustering::math
