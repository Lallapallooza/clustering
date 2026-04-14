#pragma once

#include <cmath>
#include <cstddef>
#include <type_traits>
#include <utility>

#include "clustering/ndarray.h"

#ifdef CLUSTERING_USE_AVX2
#include "clustering/math/detail/distance_avx2.h"
#endif

namespace clustering::math::distance {

/**
 * @brief Tag selecting the squared Euclidean metric.
 *
 * Passed as the first argument to @c pointwiseSq; overloads of @c tag_invoke keyed on this tag
 * implement @c sum((a_i - b_i)^2).
 */
struct SqEuclideanTag {};

/**
 * @brief Tag selecting the cosine distance metric (1 - cos(angle)).
 */
struct CosineTag {};

/**
 * @brief Tag selecting the Manhattan (L1) metric: sum of absolute differences.
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
 * When @c CLUSTERING_USE_AVX2 is defined, @c T is @c float, @p n is at least 8, and both
 * operands are 32-byte aligned, dispatches to @c detail::sqEuclideanAvx2F32. The 8-lane gate
 * mirrors @c kdtree.h: below 8 dims the horizontal-sum epilogue is pure tax. The alignment
 * check guards against unaligned Borrowed / strided views reaching @c _mm256_load_ps, which is
 * undefined behavior on misaligned inputs.
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
#ifdef CLUSTERING_USE_AVX2
  if constexpr (std::is_same_v<T, float>) {
    // _mm256_load_ps issues an 8-lane contiguous aligned read. Gate dispatch on contiguity
    // (a strided view shares element type but not memory layout), 32-byte alignment (misaligned
    // load is UB), and the 8-dim floor (below 8 the horizontal-sum epilogue is pure tax; matches
    // the kdtree dispatch rationale).
    if (n >= 8 && a.isContiguous() && b.isContiguous() && a.template isAligned<32>() &&
        b.template isAligned<32>()) {
      return detail::sqEuclideanAvx2F32(a.template alignedData<32>(), b.template alignedData<32>(),
                                        n);
    }
  }
#endif
  T sum = T{0};
  for (std::size_t i = 0; i < n; ++i) {
    const T diff = a(i) - b(i);
    sum += diff * diff;
  }
  return sum;
}

/**
 * @brief Manhattan (L1) distance between two rank-1 NDArrays.
 *
 * Default overload of @c pointwiseSq for @c ManhattanTag. Sums @c |a(i) - b(i)| head-to-tail;
 * the summation order is deterministic for equal inputs regardless of layout. A zero-length
 * operand returns @c T{0} without indexing storage. The absolute value is selected via a
 * ternary on the signed difference, which avoids the @c std::abs / @c abs overload-set ambiguity
 * and stays @c noexcept for the @c T in {float, double} substrate.
 *
 * @tparam T  Element type; float or double per the NDArray invariant.
 * @tparam LA Layout tag of @p a.
 * @tparam LB Layout tag of @p b.
 * @param a Left-hand operand; must have the same length as @p b.
 * @param b Right-hand operand.
 * @return Sum of absolute differences as a @c T.
 */
template <class T, Layout LA, Layout LB>
T tag_invoke(const detail::PointwiseSqFn & /*cpo*/, ManhattanTag, const NDArray<T, 1, LA> &a,
             const NDArray<T, 1, LB> &b) noexcept {
  const std::size_t n = a.dim(0);
  T sum = T{0};
  for (std::size_t i = 0; i < n; ++i) {
    const T diff = a(i) - b(i);
    sum += diff < T{0} ? -diff : diff;
  }
  return sum;
}

/**
 * @brief Cosine distance between two rank-1 NDArrays.
 *
 * Default overload of @c pointwiseSq for @c CosineTag. Computes @c 1 - dot(a,b) / (||a||*||b||)
 * with a single head-to-tail pass that accumulates the dot product and both squared norms; the
 * combined @c sqrt on @c na*nb keeps the arithmetic to one square root per call. A zero-norm
 * operand makes the cosine undefined: by convention this returns @c T{1} (maximum dissimilarity),
 * picked so neighbor-search code paths see a large but finite distance instead of NaN.
 *
 * @tparam T  Element type; float or double per the NDArray invariant.
 * @tparam LA Layout tag of @p a.
 * @tparam LB Layout tag of @p b.
 * @param a Left-hand operand; must have the same length as @p b.
 * @param b Right-hand operand.
 * @return Cosine distance as a @c T in @c [0, 2], or @c T{1} when either operand has zero norm.
 */
template <class T, Layout LA, Layout LB>
T tag_invoke(const detail::PointwiseSqFn & /*cpo*/, CosineTag, const NDArray<T, 1, LA> &a,
             const NDArray<T, 1, LB> &b) noexcept {
  const std::size_t n = a.dim(0);
  T dot = T{0};
  T na = T{0};
  T nb = T{0};
  for (std::size_t i = 0; i < n; ++i) {
    const T ai = a(i);
    const T bi = b(i);
    dot += ai * bi;
    na += ai * ai;
    nb += bi * bi;
  }
  if (na == T{0} || nb == T{0}) {
    return T{1};
  }
  return T{1} - (dot / std::sqrt(na * nb));
}

} // namespace clustering::math::distance
