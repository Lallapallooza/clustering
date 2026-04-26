#pragma once

#include <cstddef>
#include <cstdint>
#include <variant>

#include "clustering/kmeans/policy/afkmc2_seeder.h"
#include "clustering/kmeans/policy/greedy_kmpp_seeder.h"
#include "clustering/math/thread.h"
#include "clustering/ndarray.h"

namespace clustering::kmeans {

enum class AutoSeederMode { kSingleRun, kBestOf };

/**
 * @brief Seeder that picks between greedy k-means++ and AFK-MC2 against workload shape.
 *
 * The default @ref AutoSeederMode::kSingleRun mode keeps AFK-MC2 on the large-@c k envelope
 * (`n >= afkmc2NThreshold` and `k >= afkmc2KFloor`). @ref AutoSeederMode::kBestOf additionally
 * enables AFK-MC2 at the tuned restart envelope (`n >= afkmc2BestOfNThreshold`,
 * `d >= afkmc2BestOfDThreshold`, `k >= afkmc2BestOfKFloor`, and
 * `n * d >= afkmc2BestOfWorkThreshold`). The alternative is re-picked lazily when the
 * `(n, d, k)` shape changes between @c run calls so repeated fits at a stable shape preserve
 * the held seeder's scratch.
 *
 * @tparam T Element type of the point cloud.
 * @tparam Mode Dispatch envelope to use for this seeder instance.
 */
template <class T, AutoSeederMode Mode = AutoSeederMode::kSingleRun> class AutoSeeder {
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
  /// @c n threshold above which AFK-MC2 is preferred over greedy k-means++.
  static constexpr std::size_t afkmc2NThreshold = 500000;
#endif

  /// Minimum @c k at which AFK-MC2 is considered; mirrors @ref AfkMc2Seeder::kFloor.
  static constexpr std::size_t afkmc2KFloor = AfkMc2Seeder<T>::kFloor;
#ifdef CLUSTERING_KMEANS_AFKMC2_BEST_OF_N_THRESHOLD
  /**
   * @brief Best-of restart threshold on @c n above which AFK-MC2 is preferred.
   *
   * Override with @c -DCLUSTERING_KMEANS_AFKMC2_BEST_OF_N_THRESHOLD=<value>.
   */
  static constexpr std::size_t afkmc2BestOfNThreshold =
      CLUSTERING_KMEANS_AFKMC2_BEST_OF_N_THRESHOLD;
#else
  /// Best-of restart threshold on @c n above which AFK-MC2 is preferred.
  static constexpr std::size_t afkmc2BestOfNThreshold = 5000;
#endif

#ifdef CLUSTERING_KMEANS_AFKMC2_BEST_OF_D_THRESHOLD
  /**
   * @brief Best-of restart threshold on @c d above which AFK-MC2 is preferred.
   *
   * Override with @c -DCLUSTERING_KMEANS_AFKMC2_BEST_OF_D_THRESHOLD=<value>.
   */
  static constexpr std::size_t afkmc2BestOfDThreshold =
      CLUSTERING_KMEANS_AFKMC2_BEST_OF_D_THRESHOLD;
#else
  /// Best-of restart threshold on @c d above which AFK-MC2 is preferred.
  static constexpr std::size_t afkmc2BestOfDThreshold = 8;
#endif

#ifdef CLUSTERING_KMEANS_AFKMC2_BEST_OF_K_FLOOR
  /**
   * @brief Minimum @c k for best-of restart AFK-MC2 dispatch.
   *
   * Override with @c -DCLUSTERING_KMEANS_AFKMC2_BEST_OF_K_FLOOR=<value>.
   */
  static constexpr std::size_t afkmc2BestOfKFloor = CLUSTERING_KMEANS_AFKMC2_BEST_OF_K_FLOOR;
#else
  /// Minimum @c k for best-of restart AFK-MC2 dispatch.
  static constexpr std::size_t afkmc2BestOfKFloor = 16;
#endif

#ifdef CLUSTERING_KMEANS_AFKMC2_BEST_OF_WORK_THRESHOLD
  /**
   * @brief Minimum @c n * d work envelope for best-of restart AFK-MC2 dispatch.
   *
   * Override with @c -DCLUSTERING_KMEANS_AFKMC2_BEST_OF_WORK_THRESHOLD=<value>.
   */
  static constexpr std::size_t afkmc2BestOfWorkThreshold =
      CLUSTERING_KMEANS_AFKMC2_BEST_OF_WORK_THRESHOLD;
#else
  /// Minimum @c n * d work envelope for best-of restart AFK-MC2 dispatch.
  static constexpr std::size_t afkmc2BestOfWorkThreshold = 80000;
#endif

  /// Default Markov-chain length passed through to AFK-MC2.
  static constexpr std::size_t afkmc2ChainLengthDefault = AfkMc2Seeder<T>::chainLengthDefault;

  /// Seed @p outCentroids with the dispatched seeder; see the class docs for the dispatch rule.
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
    if (shouldUseAfkmc2(n, d, k)) {
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

  [[nodiscard]] static constexpr bool shouldUseAfkmc2(std::size_t n, std::size_t d,
                                                      std::size_t k) noexcept {
    const bool largeKEnvelope = (n >= afkmc2NThreshold) && (k >= afkmc2KFloor);
    if constexpr (Mode == AutoSeederMode::kBestOf) {
      const bool bestOfEnvelope = (n >= afkmc2BestOfNThreshold) && (d >= afkmc2BestOfDThreshold) &&
                                  (k >= afkmc2BestOfKFloor) &&
                                  (n >= ((afkmc2BestOfWorkThreshold + d - 1) / d));
      return largeKEnvelope || bestOfEnvelope;
    } else {
      (void)d;
      return largeKEnvelope;
    }
  }

  std::variant<GreedyKmppSeeder<T>, AfkMc2Seeder<T>> m_held{
      std::in_place_type<GreedyKmppSeeder<T>>};
  std::size_t m_lastN = 0;
  std::size_t m_lastD = 0;
  std::size_t m_lastK = 0;
};

} // namespace clustering::kmeans
