#pragma once

#include <cstddef>
#include <type_traits>
#include <variant>

#include "clustering/hdbscan/mst_output.h"
#include "clustering/hdbscan/policy/boruvka_mst_backend.h"
#include "clustering/hdbscan/policy/nn_descent_mst_backend.h"
#include "clustering/hdbscan/policy/prim_mst_backend.h"
#include "clustering/math/thread.h"
#include "clustering/ndarray.h"

namespace clustering::hdbscan {

/**
 * @brief MST backend that dispatches between Prim, Boruvka, and NN-Descent on input shape.
 *
 * The dispatcher holds a @c std::variant over the three concrete backends and re-emplaces the
 * active arm whenever the `(n, d)` shape of the input changes between @c run calls. Each arm
 * runs its own @c run on the held point cloud; the condensed-tree pipeline downstream is
 * monomorphic and reads from @ref MstOutput.
 *
 * Selection rules, total over the full `(N, d)` domain:
 *   - @c d <= @ref AutoMstBackend::boruvkaLowDimCeil -> @ref BoruvkaMstBackend (KDTree pruning
 * dominates at every @c n; dense Prim's quadratic overhead does not amortise).
 *   - @c d > @ref AutoMstBackend::boruvkaLowDimCeil and @c N*N*sizeof(T) <= @ref
 * kPrimMrdMatrixByteBudget ->
 *     @ref PrimMstBackend (streaming dense Prim beats KDTree fan-out once AABB pruning decays
 *     with @c d).
 *   - @c d <= @ref AutoMstBackend::boruvkaDimCeil and Prim is out of budget -> @ref
 * BoruvkaMstBackend (KDTree-accelerated, exact; AABB pruning still fires enough at moderate @c d).
 *   - @c d > @ref AutoMstBackend::boruvkaDimCeil and Prim is out of budget -> @ref
 * NnDescentMstBackend (approximate kNN-graph + Kruskal with a connectivity fallback).
 *
 * The dimensional ceiling is overridable via @c CLUSTERING_HDBSCAN_BORUVKA_DIM_CEIL. The low-d
 * ceiling is overridable via @c CLUSTERING_HDBSCAN_BORUVKA_LOW_DIM_CEIL. The Prim regime is
 * gated directly by the byte budget so no override choice can push Prim above the documented
 * memory ceiling.
 *
 * Staleness uses an `(n, d)` shape tuple only, mirroring @c AutoSeeder. The dispatcher does
 * not cache data-dependent state across calls: every @c run rebuilds the held backend's
 * data-dependent index (KDTree, kNN graph), and the variant arm itself is reconstructed when a
 * shape change crosses a dispatch boundary.
 *
 * @tparam T Element type of the point cloud. Only @c float is supported.
 */
template <class T> class AutoMstBackend {
  static_assert(std::is_same_v<T, float>,
                "AutoMstBackend<T> supports only float; a double specialization is out of scope.");

public:
#ifdef CLUSTERING_HDBSCAN_BORUVKA_LOW_DIM_CEIL
  /**
   * @brief Low-dimensional ceiling at or below which Boruvka is preferred regardless of @c N.
   *
   * KDTree AABB pruning is highly effective at low @c d, so the per-query fan-out of
   * KDTree-Boruvka beats dense-MRD Prim even at small @c N. Above this ceiling the ranking
   * inverts once Prim fits in its byte budget.
   *
   * Override with @c -DCLUSTERING_HDBSCAN_BORUVKA_LOW_DIM_CEIL=<value>.
   */
  static constexpr std::size_t boruvkaLowDimCeil = CLUSTERING_HDBSCAN_BORUVKA_LOW_DIM_CEIL;
#else
  /// Low-dimensional ceiling at or below which Boruvka is preferred regardless of @c N.
  static constexpr std::size_t boruvkaLowDimCeil = 16;
#endif

#ifdef CLUSTERING_HDBSCAN_BORUVKA_DIM_CEIL
  /**
   * @brief Dimensional ceiling above which the KDTree-based Boruvka backend gives way to the
   *        NN-Descent approximate backend (only consulted when Prim is out of byte budget).
   *
   * Override with @c -DCLUSTERING_HDBSCAN_BORUVKA_DIM_CEIL=<value>.
   */
  static constexpr std::size_t boruvkaDimCeil = CLUSTERING_HDBSCAN_BORUVKA_DIM_CEIL;
#else
  /// Dimensional ceiling above which KDTree Boruvka gives way to NN-Descent when Prim is out
  /// of byte budget.
  static constexpr std::size_t boruvkaDimCeil = 60;
#endif

  static_assert(boruvkaLowDimCeil <= boruvkaDimCeil,
                "boruvkaLowDimCeil must not exceed boruvkaDimCeil; the dispatch order assumes a "
                "nested Boruvka regime at low d and a relaxed one at moderate d.");

  /**
   * @brief Whether the Prim regime applies at @p n under the dense-MRD byte budget.
   *
   * Prim materialises an @c n*n matrix of @c T, so the admissible set is @c n*n*sizeof(T) <=
   * @c kPrimMrdMatrixByteBudget. Exposed as a static helper so callers and tests share the
   * exact boundary the dispatcher uses.
   */
  static constexpr bool primFitsBudget(std::size_t n) noexcept {
    if (n == 0) {
      return true;
    }
    constexpr std::size_t kNsqBudget = kPrimMrdMatrixByteBudget / sizeof(T);
    return n <= kNsqBudget / n;
  }

  AutoMstBackend() = default;

  /**
   * @brief Fit a backend arm chosen on the input shape and delegate @c run to it.
   *
   * @param X          Point cloud; caller retains ownership for the call's lifetime.
   * @param minSamples Core-distance neighbour count; forwarded to the dispatched backend.
   * @param pool       Worker pool handle; forwarded to the dispatched backend.
   * @param out        Destination for edges and core distances; written by the dispatched backend.
   */
  void run(const NDArray<T, 2> &X, std::size_t minSamples, math::Pool pool, MstOutput<T> &out) {
    ensureShape(X.dim(0), X.dim(1));
    std::visit([&](auto &b) { b.run(X, minSamples, pool, out); }, m_held);
  }

  /**
   * @brief Index of the currently held variant arm.
   *
   * Returns @c 0 for Boruvka, @c 1 for Prim, and @c 2 for NN-Descent, matching the variant's
   * declared order. Useful for dispatcher-level tests that need to confirm which backend was
   * selected for an input shape without running the full pipeline.
   */
  [[nodiscard]] std::size_t heldIndex() const noexcept { return m_held.index(); }

  /**
   * @brief Resolve the variant arm for @p (n, d) without running the pipeline.
   *
   * Emplaces the correct variant arm per the dispatch rules and returns @ref heldIndex. No MST
   * is built; @c minSamples and @c pool are not consulted. Intended for dispatch-verification
   * tests and benchmark harnesses that want to know which backend a shape would hit without
   * paying the full-fit cost.
   */
  std::size_t peekArm(std::size_t n, std::size_t d) {
    ensureShape(n, d);
    return m_held.index();
  }

private:
  void ensureShape(std::size_t n, std::size_t d) {
    if (n == m_lastN && d == m_lastD) {
      return;
    }
    if (d <= boruvkaLowDimCeil) {
      if (!std::holds_alternative<BoruvkaMstBackend<T>>(m_held)) {
        m_held.template emplace<BoruvkaMstBackend<T>>();
      }
    } else if (d <= boruvkaDimCeil && primFitsBudget(n)) {
      // Streaming Prim beats both Boruvka and NN-Descent in the `d <= 60` band while @c n
      // stays inside its quadratic compute budget. Above @c boruvkaDimCeil the dense pairwise
      // work is dominated by the @c d-wide distance compute and NN-Descent wins; the cap keeps
      // Prim out of that regime.
      if (!std::holds_alternative<PrimMstBackend<T>>(m_held)) {
        m_held.template emplace<PrimMstBackend<T>>();
      }
    } else if (d <= boruvkaDimCeil) {
      if (!std::holds_alternative<BoruvkaMstBackend<T>>(m_held)) {
        m_held.template emplace<BoruvkaMstBackend<T>>();
      }
    } else {
      if (!std::holds_alternative<NnDescentMstBackend<T>>(m_held)) {
        m_held.template emplace<NnDescentMstBackend<T>>();
      }
    }
    m_lastN = n;
    m_lastD = d;
  }

  // Variant order is fixed: Boruvka (0), Prim (1), NN-Descent (2). @ref heldIndex leans on this.
  std::variant<BoruvkaMstBackend<T>, PrimMstBackend<T>, NnDescentMstBackend<T>> m_held{
      std::in_place_type<PrimMstBackend<T>>};
  std::size_t m_lastN = 0;
  std::size_t m_lastD = 0;
};

} // namespace clustering::hdbscan
