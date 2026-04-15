#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

#include "clustering/always_assert.h"
#include "clustering/ndarray.h"

namespace clustering::kmeans::detail {

/**
 * @brief Reseed a given centroid row to the farthest point from any currently assigned centroid.
 *
 * Scans @p minDistSq for its argmax (the point currently worst-covered by the centroid set),
 * copies that point into @p centroids row @p cluster, and zeroes @c minDistSq at the donor so
 * the same point cannot be picked again for a different empty cluster in the same pass.
 *
 * @return The donor row index in @p X.
 */
template <class T>
[[nodiscard]] inline std::size_t reseedToFarthestPoint(const NDArray<T, 2, Layout::Contig> &X,
                                                       NDArray<T, 2, Layout::Contig> &centroids,
                                                       NDArray<T, 1> &minDistSq,
                                                       std::size_t cluster) noexcept {
  const std::size_t n = X.dim(0);
  const std::size_t d = X.dim(1);

  std::size_t donor = 0;
  T bestVal = minDistSq(0);
  for (std::size_t i = 1; i < n; ++i) {
    const T v = minDistSq(i);
    if (v > bestVal) {
      bestVal = v;
      donor = i;
    }
  }

  std::memcpy(centroids.data() + (cluster * d), X.data() + (donor * d), d * sizeof(T));
  minDistSq(donor) = T{0};
  return donor;
}

/**
 * @brief Reseed every empty cluster after the label fold; return the number of clusters reseeded.
 *
 * Walks @p counts once; for each cluster with zero assigned points, calls
 * @ref reseedToFarthestPoint and resets that cluster's sum row to the donor's coordinates so
 * the mean step yields the reseeded point itself. The single sweep over @p counts bounds the
 * reseed work at @c O(k * n) in the worst case -- the @c k > n/2 pathology cannot loop
 * indefinitely because each donor's @c minDistSq is zeroed before the next argmax scan.
 *
 * @return Number of clusters reseeded (zero on the convergent steady state).
 */
template <class T>
[[nodiscard]] inline std::size_t
reseedEmptyClusters(const NDArray<T, 2, Layout::Contig> &X,
                    NDArray<T, 2, Layout::Contig> &centroids, NDArray<T, 2, Layout::Contig> &sums,
                    NDArray<std::int32_t, 1> &counts, NDArray<T, 1> &minDistSq) noexcept {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                "reseedEmptyClusters<T> requires T to be float or double");
  const std::size_t k = counts.dim(0);
  const std::size_t d = X.dim(1);
  std::size_t reseeds = 0;
  for (std::size_t c = 0; c < k; ++c) {
    if (counts(c) != 0) {
      continue;
    }
    const std::size_t donor = reseedToFarthestPoint<T>(X, centroids, minDistSq, c);
    const T *xRow = X.data() + (donor * d);
    T *sumRow = sums.data() + (c * d);
    for (std::size_t t = 0; t < d; ++t) {
      sumRow[t] = xRow[t];
    }
    counts(c) = 1;
    ++reseeds;
  }
  return reseeds;
}

} // namespace clustering::kmeans::detail
