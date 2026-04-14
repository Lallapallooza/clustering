#pragma once

#include <cstddef>
#include <utility>

#include "clustering/ndarray.h"

namespace clustering::math::distance {

/**
 * @brief Tag selecting the squared Euclidean metric.
 *
 * Passed as the first argument to @c pointwiseSq; overloads of @c tag_invoke keyed on this tag
 * implement @c sum((a_i - b_i)^2).
 */
struct SqEuclideanTag {};

/**
 * @brief Tag selecting the cosine metric.
 *
 * Declared for API completeness. No overload of @c tag_invoke is provided yet; instantiating
 * @c pointwiseSq with this tag is a compile error until a body is added.
 */
struct CosineTag {};

/**
 * @brief Tag selecting the Manhattan (L1) metric.
 *
 * Declared for API completeness. No overload of @c tag_invoke is provided yet; instantiating
 * @c pointwiseSq with this tag is a compile error until a body is added.
 */
struct ManhattanTag {};

namespace detail {

/**
 * @brief Function-object type implementing the @c pointwiseSq customization point.
 *
 * Dispatches to an unqualified @c tag_invoke call so ADL can reach user-defined overloads in the
 * tag's namespace. The function object itself is passed as the first argument so overloads can
 * key on the CPO identity in addition to the tag.
 */
struct PointwiseSqFn {
  template <class Tag, class A, class B>
  constexpr auto operator()(Tag tag, A &&a, B &&b) const
      noexcept(noexcept(tag_invoke(std::declval<const PointwiseSqFn &>(), tag, std::forward<A>(a),
                                   std::forward<B>(b))))
          -> decltype(tag_invoke(std::declval<const PointwiseSqFn &>(), tag, std::forward<A>(a),
                                 std::forward<B>(b))) {
    return tag_invoke(*this, tag, std::forward<A>(a), std::forward<B>(b));
  }
};

} // namespace detail

/**
 * @brief Customization-point object for pairwise squared distances between two rank-1 arrays.
 *
 * @c pointwiseSq(tag, a, b) dispatches through an unqualified @c tag_invoke lookup, so the
 * metric implementation travels with the tag's namespace. Users extend by defining a
 * @c tag_invoke overload taking the CPO and their own tag; library-provided overloads live in
 * @c clustering::math::distance and are found via ADL on the tag parameter.
 *
 * The return type and @c noexcept qualification are inherited from the selected overload; the
 * existing squared-Euclidean fast path in the kd-tree is @c noexcept, and the default overload
 * below preserves that guarantee.
 */
inline constexpr detail::PointwiseSqFn pointwiseSq{};

/**
 * @brief Squared Euclidean distance between two rank-1 NDArrays.
 *
 * Default overload of @c pointwiseSq for @c SqEuclideanTag. Sums @c (a(i) - b(i))^2 from
 * @c i = 0 head-to-tail; the summation order matches the scalar kd-tree path and is
 * deterministic for equal inputs regardless of layout. A zero-length operand returns @c T{0}
 * without indexing storage.
 *
 * @tparam T  Element type; float or double per the NDArray invariant.
 * @tparam LA Layout tag of @p a.
 * @tparam LB Layout tag of @p b.
 * @param a Left-hand operand; must have the same length as @p b.
 * @param b Right-hand operand.
 * @return Sum of squared differences as a @c T.
 */
template <class T, Layout LA, Layout LB>
T tag_invoke(const detail::PointwiseSqFn & /*cpo*/, SqEuclideanTag, const NDArray<T, 1, LA> &a,
             const NDArray<T, 1, LB> &b) noexcept {
  const std::size_t n = a.dim(0);
  T sum = T{0};
  for (std::size_t i = 0; i < n; ++i) {
    const T diff = a(i) - b(i);
    sum += diff * diff;
  }
  return sum;
}

} // namespace clustering::math::distance
