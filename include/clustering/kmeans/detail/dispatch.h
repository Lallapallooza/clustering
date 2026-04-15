#pragma once

#include <cstddef>
#include <cstdint>

namespace clustering::kmeans::detail {

/**
 * @brief Diagnostic tag identifying which inner Lloyd-family algorithm executed.
 *
 * Test-only discriminator returned by @c KMeans::lastAlgorithm; not a stable API surface.
 * Branching on this enum in production couples the caller to the dispatch implementation.
 *
 * Future enumerators reserved (do not reuse the underlying values below):
 *   kHamerly, kShallueHamerly, kBallKMeans.
 */
enum class Algorithm : std::uint8_t {
  kLloydFusedGemm = 0,
  kAuto = 255,
};

/**
 * @brief Diagnostic tag identifying which seeder produced the initial centroids.
 *
 * Same non-contractual status as @ref Algorithm. Future enumerators reserved
 * (do not reuse the underlying values below): kKMeansParallel.
 */
enum class Seeder : std::uint8_t {
  kGreedyKMeansPlusPlus = 0,
  kAfkMc2 = 1,
  kAuto = 255,
};

#ifdef CLUSTERING_KMEANS_AFKMC2_N_THRESHOLD
/**
 * @brief Auto-dispatch threshold on @c n above which AFK-MC2 is preferred over greedy
 *        k-means++; below the threshold the dispatch falls back to greedy.
 *
 * Override with @c -DCLUSTERING_KMEANS_AFKMC2_N_THRESHOLD=<value>.
 */
inline constexpr std::size_t afkmc2NThreshold = CLUSTERING_KMEANS_AFKMC2_N_THRESHOLD;
#else
inline constexpr std::size_t afkmc2NThreshold = 500000;
#endif

#ifdef CLUSTERING_KMEANS_AFKMC2_K_FLOOR
/**
 * @brief Minimum @c k for AFK-MC2 to be selected; below this, dispatch falls back to greedy.
 */
inline constexpr std::size_t afkmc2KFloor = CLUSTERING_KMEANS_AFKMC2_K_FLOOR;
#else
inline constexpr std::size_t afkmc2KFloor = 100;
#endif

#ifdef CLUSTERING_KMEANS_AFKMC2_CHAIN_LENGTH
/**
 * @brief Default Markov chain length for AFK-MC2 per centroid pick.
 *
 * Bachem 2016 reports @c m=200 as the sweet spot for the log-k approximation guarantee.
 * Override with @c -DCLUSTERING_KMEANS_AFKMC2_CHAIN_LENGTH=<value>; values below a few dozen
 * trade the provable bound for faster seeding, values above 200 amortize into larger @c n
 * regimes where the chain's sublinear-in-n behavior is the dominant cost.
 */
inline constexpr std::size_t afkmc2ChainLengthDefault = CLUSTERING_KMEANS_AFKMC2_CHAIN_LENGTH;
#else
inline constexpr std::size_t afkmc2ChainLengthDefault = 200;
#endif

#ifdef CLUSTERING_KMEANS_KAHAN_N_THRESHOLD
/**
 * @brief @c n threshold at which the centroid accumulator switches to the Kahan-compensated
 *        variant. Below this, the plain partial-sum + fold variant is used.
 *
 * Compensation is load-bearing for the 1%-inertia gate at the @c (n=1e6, k=1000) corner where
 * per-cluster running totals are dominated by a large sum plus many small addends. Override
 * with @c -DCLUSTERING_KMEANS_KAHAN_N_THRESHOLD=<value>.
 */
inline constexpr std::size_t kahanNThreshold = CLUSTERING_KMEANS_KAHAN_N_THRESHOLD;
#else
inline constexpr std::size_t kahanNThreshold = 100000;
#endif

/**
 * @brief Auto-dispatch decision: pick the inner algorithm given a workload shape.
 *
 * Currently wires only the fused-argmin-GEMM Lloyd path; @p n, @p d, @p k are consumed to
 * stabilize the signature against future variants (Hamerly, Shallue-Hamerly, Ball) without a
 * call-site churn.
 */
[[nodiscard]] inline Algorithm chooseAlgorithm(std::size_t n, std::size_t d,
                                               std::size_t k) noexcept {
  (void)n;
  (void)d;
  (void)k;
  return Algorithm::kLloydFusedGemm;
}

/**
 * @brief Auto-dispatch decision: pick the seeder given a workload shape.
 *
 * AFK-MC2 activates only at @c n >= @ref afkmc2NThreshold AND @c k >= @ref afkmc2KFloor;
 * everything else falls through to greedy k-means++.
 */
[[nodiscard]] inline Seeder chooseSeeder(std::size_t n, std::size_t k) noexcept {
  if (n >= afkmc2NThreshold && k >= afkmc2KFloor) {
    return Seeder::kAfkMc2;
  }
  return Seeder::kGreedyKMeansPlusPlus;
}

} // namespace clustering::kmeans::detail
