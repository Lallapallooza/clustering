#pragma once

#include <cstddef>
#include <cstdint>
#include <variant>

#include "clustering/kmeans/policy/afkmc2_seeder.h"
#include "clustering/kmeans/policy/greedy_kmpp_seeder.h"
#include "clustering/math/thread.h"
#include "clustering/ndarray.h"

namespace clustering::kmeans {

/**
 * @brief Seeder that picks between greedy k-means++ and AFK-MC2 against workload shape.
 *
 * AFK-MC2 activates only at @c n >= @ref afkmc2NThreshold AND @c k >= @ref afkmc2KFloor;
 * everything else runs greedy k-means++. The alternative is re-picked lazily when the
 * @c (n, d, k) shape changes between @c run calls so repeated fits at a stable shape preserve
 * the held seeder's scratch.
 *
 * @tparam T Element type of the point cloud.
 */
template <class T> class AutoSeeder {
public:
#ifdef CLUSTERING_KMEANS_AFKMC2_N_THRESHOLD
  /**
   * @brief Auto-dispatch threshold on @c n above which AFK-MC2 is preferred over greedy
   *        k-means++; below the threshold the dispatch falls back to greedy.
   *
   * Override with @c -DCLUSTERING_KMEANS_AFKMC2_N_THRESHOLD=<value>.
   */
  static constexpr std::size_t afkmc2NThreshold = CLUSTERING_KMEANS_AFKMC2_N_THRESHOLD;
#else
  static constexpr std::size_t afkmc2NThreshold = 500000;
#endif

  static constexpr std::size_t afkmc2KFloor = AfkMc2Seeder<T>::kFloor;
  static constexpr std::size_t afkmc2ChainLengthDefault = AfkMc2Seeder<T>::chainLengthDefault;

  void run(const NDArray<T, 2> &X, std::size_t k, std::uint64_t seed, math::Pool pool,
           NDArray<T, 2> &outCentroids) {
    const std::size_t n = X.dim(0);
    const std::size_t d = X.dim(1);
    ensureShape(n, d, k);
    std::visit([&](auto &s) { s.run(X, k, seed, pool, outCentroids); }, m_held);
  }

private:
  void ensureShape(std::size_t n, std::size_t d, std::size_t k) {
    if (n == m_lastN && d == m_lastD && k == m_lastK) {
      return;
    }
    if (n >= afkmc2NThreshold && k >= afkmc2KFloor) {
      if (!std::holds_alternative<AfkMc2Seeder<T>>(m_held)) {
        m_held.template emplace<AfkMc2Seeder<T>>();
      }
    } else {
      if (!std::holds_alternative<GreedyKmppSeeder<T>>(m_held)) {
        m_held.template emplace<GreedyKmppSeeder<T>>();
      }
    }
    m_lastN = n;
    m_lastD = d;
    m_lastK = k;
  }

  std::variant<GreedyKmppSeeder<T>, AfkMc2Seeder<T>> m_held{
      std::in_place_type<GreedyKmppSeeder<T>>};
  std::size_t m_lastN = 0;
  std::size_t m_lastD = 0;
  std::size_t m_lastK = 0;
};

} // namespace clustering::kmeans
