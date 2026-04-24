#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

#include "clustering/always_assert.h"
#include "clustering/index/nn_descent/detail/join_step.h"
#include "clustering/index/nn_descent/detail/neighbor_heap.h"
#include "clustering/index/nn_descent/detail/rp_tree_init.h"
#include "clustering/math/thread.h"
#include "clustering/ndarray.h"

namespace clustering::index {

/**
 * @brief Approximate @c k -nearest-neighbor graph via the NN-Descent algorithm (Dong, Charikar,
 *        Li 2011) with random-projection-tree initialization.
 *
 * Maintains a per-node bounded max-heap of @c k current-best neighbors. Each iteration performs
 * a "local join": for every node @c u the step takes @c u 's current neighbors and the
 * neighbors-of-neighbors as candidates, computing squared Euclidean distances and admitting any
 * candidate that beats the heap's current worst. Convergence is reached when the fraction of
 * updated slots falls below @p delta.
 *
 * Initialization via one random-projection tree seeds each node with coarse neighbors from the
 * same tree leaf. Dong 2011 shows RP-tree initialization typically cuts iteration count by half
 * compared to uniform-random seeding on real workloads; defaults here follow that pattern.
 *
 * @par Warm start
 * Rebuilding on the same `(data pointer, @c n, @c d, @c k)` tuple reuses the existing neighbor
 * graph as warm state. The join loop runs without RP-tree re-initialization, so the second
 * build converges measurably faster (typically in a single iteration). A different @c k, a
 * different input pointer, or a different shape forces a cold start (fresh bank + RP-tree init).
 *
 * @par Thread safety
 * Not thread-safe: a single @c NnDescentIndex instance must not be driven concurrently. The
 * internal join step uses a @c Pool injection to parallelise across nodes per @ref build; no
 * nested pool dispatch.
 *
 * @tparam T Element type; only @c float is currently supported.
 */
template <class T> class NnDescentIndex {
  static_assert(std::is_same_v<T, float>,
                "NnDescentIndex<T> supports only float; add a specialization to extend.");

public:
  /// Per-node kNN entry returned by @ref neighbors. Squared Euclidean distance carried as @c T.
  struct KnnEntry {
    /// Neighbor point index.
    std::int32_t idx;
    /// Squared Euclidean distance from the query to @c idx.
    T sqDist;
  };

  /**
   * @brief Construct an index targeting @p k neighbors per node.
   *
   * @param k       Neighbors per node; must be at least 1. Zero-argument construction is
   *                rejected: @p k has no safe default and the consuming code (HDBSCAN's
   *                NN-Descent MST backend) always knows @p k at call time.
   * @param maxIter Iteration cap on the join loop. A default of 10 gives ample headroom at the
   *                workload shapes this class targets; most inputs converge in three to five.
   * @param delta   Convergence fraction: terminate when the ratio of updated neighbor slots
   *                drops below this. Smaller values spend more iterations.
   * @param seed    PRNG seed for RP-tree partition choices. Two builds with identical @p seed on
   *                identical input produce bit-for-bit identical neighbor graphs at
   *                single-threaded execution.
   */
  explicit NnDescentIndex(std::size_t k, std::size_t maxIter = 10, T delta = T{0.001},
                          std::uint64_t seed = 0)
      : m_k(k), m_maxIter(maxIter), m_delta(delta), m_seed(seed) {
    CLUSTERING_ALWAYS_ASSERT(k >= 1);
  }

  NnDescentIndex(const NnDescentIndex &) = delete;
  NnDescentIndex &operator=(const NnDescentIndex &) = delete;
  /// Defaulted move constructor; transfers the neighbor bank and scratch.
  NnDescentIndex(NnDescentIndex &&) = default;
  /// Defaulted move assignment; transfers the neighbor bank and scratch.
  NnDescentIndex &operator=(NnDescentIndex &&) = default;
  ~NnDescentIndex() = default;

  /**
   * @brief Build (or rebuild) the approximate kNN graph for @p X.
   *
   * Runs RP-tree initialization followed by the NN-Descent join loop. Rebuilding on the same
   * `(data pointer, @c n, @c d, @c k)` tuple reuses the existing neighbors as warm state; any
   * change triggers a cold start.
   *
   * @param X    Row-major @c n x d point matrix; must outlive the call but not the index.
   * @param pool Parallelism injection for the inner join loop.
   */
  void build(const NDArray<T, 2> &X, math::Pool pool) {
    const std::size_t n = X.dim(0);
    const std::size_t d = X.dim(1);
    const bool kFitsN = (n == 0) || (m_k < n);
    CLUSTERING_ALWAYS_ASSERT(kFitsN);

    const bool sameShape =
        (m_lastN == n) && (m_lastD == d) && (m_lastK == m_k) && m_bank.has_value();

    m_lastIterations = 0;

    if (!sameShape) {
      // Cold start: discard any prior bank and rebuild from RP-tree init.
      m_bank.emplace(n, m_k);
      if (n == 0 || m_k == 0) {
        captureShape(X, n, d);
        m_neighborsView.assign(n, {});
        return;
      }
      // Leaf limit sized to roughly @c 2k so the leaf-pair enumeration is `O(k^2)` per leaf,
      // matching Dong 2011's recommendation. A minimum of `max(2k, 8)` keeps very small @c k
      // from degenerating into singleton leaves.
      const std::size_t leafLimit = std::max<std::size_t>(2 * m_k, 8);
      nn_descent::detail::RpTreeInit<T>::build(X, leafLimit, m_seed, *m_bank);
    } else {
      // Warm start: reload the bank from the previously published sorted view so the heap
      // invariant is restored after the destructive @c toSortedLists pass, then re-flag every
      // slot as "new" so the next join iteration revisits them.
      reloadBankFromView();
    }

    if (n == 0 || m_k == 0) {
      captureShape(X, n, d);
      m_neighborsView.assign(n, {});
      return;
    }

    // Join loop. Each iteration counts updated slots; once the fraction drops below @p delta the
    // graph is considered converged.
    const std::size_t total = n * m_k;
    for (std::size_t iter = 0; iter < m_maxIter; ++iter) {
      const std::size_t updates = nn_descent::detail::JoinStep<T>::run(X, *m_bank, pool);
      ++m_lastIterations;
      if (total == 0) {
        break;
      }
      const double ratio = static_cast<double>(updates) / static_cast<double>(total);
      if (ratio < static_cast<double>(m_delta)) {
        break;
      }
    }

    captureShape(X, n, d);

    // Materialize the public neighbor view, sorted ascending by squared distance. This
    // destroys the bank's heap order; a subsequent warm-start call reheapifies via
    // @c reloadBankFromView below.
    m_neighborsView = m_bank->template toSortedLists<KnnEntry>();
    m_bankOrderValid = false;
  }

  /// Per-node @c k -NN neighbor list, sorted ascending by squared distance. Empty until the
  /// first @ref build returns.
  [[nodiscard]] const std::vector<std::vector<KnnEntry>> &neighbors() const noexcept {
    return m_neighborsView;
  }

  /**
   * @brief Whether the undirected k-NN graph covers every node in a single connected component.
   *
   * DFS over the union of forward and reverse edges from the published sorted view. Safe to call
   * any time after @ref build; returns @c true trivially for `n <= 1`.
   */
  [[nodiscard]] bool isConnected() const {
    const std::size_t n = m_neighborsView.size();
    if (n <= 1) {
      return true;
    }
    std::vector<std::vector<std::int32_t>> adj(n);
    for (std::size_t u = 0; u < n; ++u) {
      for (const KnnEntry &e : m_neighborsView[u]) {
        if (e.idx < 0 || std::cmp_greater_equal(e.idx, n)) {
          continue;
        }
        adj[u].push_back(e.idx);
        adj[static_cast<std::size_t>(e.idx)].push_back(static_cast<std::int32_t>(u));
      }
    }
    std::vector<std::uint8_t> visited(n, 0);
    std::vector<std::int32_t> stack;
    stack.reserve(n);
    stack.push_back(0);
    visited[0] = 1;
    std::size_t visitedCount = 1;
    while (!stack.empty()) {
      const std::int32_t u = stack.back();
      stack.pop_back();
      for (const std::int32_t v : adj[static_cast<std::size_t>(u)]) {
        if (visited[static_cast<std::size_t>(v)] == 0U) {
          visited[static_cast<std::size_t>(v)] = 1;
          ++visitedCount;
          stack.push_back(v);
        }
      }
    }
    return visitedCount == n;
  }

  /// @c k specified at construction.
  [[nodiscard]] std::size_t k() const noexcept { return m_k; }

  /// Number of join iterations actually executed during the most recent @ref build. @c 0 until
  /// the first build returns.
  [[nodiscard]] std::size_t lastIterations() const noexcept { return m_lastIterations; }

private:
  /**
   * @brief Reload the bank's heap-order storage from the public sorted view, re-arming every
   *        slot as "new" so the next join iteration visits it.
   *
   * Called on the warm-start path: after @c toSortedLists the bank is sorted ascending (not
   * heap-ordered), so future pushes would violate the max-heap invariant. Reloading via
   * @ref NeighborHeapBank::loadFromSorted rebuilds the heap in place per node. The "new" flag
   * is set during load so the first warm-start join has candidate edges to explore.
   */
  void reloadBankFromView() {
    if (!m_bank.has_value()) {
      return;
    }
    if (m_bankOrderValid) {
      // Bank is already in heap order; we only need to re-arm the "new" flag so the join loop
      // revisits every slot.
      m_bank->rearmAllAsNew();
      return;
    }
    std::vector<std::pair<T, std::int32_t>> buf;
    buf.reserve(m_k);
    for (std::size_t i = 0; i < m_neighborsView.size(); ++i) {
      buf.clear();
      for (const KnnEntry &e : m_neighborsView[i]) {
        buf.emplace_back(e.sqDist, e.idx);
      }
      m_bank->loadFromSorted(static_cast<std::int32_t>(i), buf);
    }
    m_bankOrderValid = true;
  }

  void captureShape(const NDArray<T, 2> & /*X*/, std::size_t n, std::size_t d) noexcept {
    m_lastN = n;
    m_lastD = d;
    m_lastK = m_k;
  }

  std::size_t m_k;
  std::size_t m_maxIter;
  T m_delta;
  std::uint64_t m_seed;

  std::optional<nn_descent::detail::NeighborHeapBank<T>> m_bank;
  std::vector<std::vector<KnnEntry>> m_neighborsView;

  std::size_t m_lastN = 0;
  std::size_t m_lastD = 0;
  std::size_t m_lastK = 0;
  std::size_t m_lastIterations = 0;
  /// @c true when @c m_bank still holds the max-heap invariant; cleared by @c toSortedLists and
  /// restored by @c reloadBankFromView.
  bool m_bankOrderValid = false;
};

} // namespace clustering::index
