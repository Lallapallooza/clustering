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
 * active arm whenever the @c (n, d) shape of the input changes between @c run calls. Each arm
 * runs its own @c run on the held point cloud; the condensed-tree pipeline downstream is
 * monomorphic and reads from @ref MstOutput.
 *
 * Selection rules, total over the full @c (N, d) domain:
 *   - @c N < @ref primNThreshold -> @ref PrimMstBackend (dense MRD matrix, exact).
 *   - @c N >= @ref primNThreshold and @c d <= @ref boruvkaDimCeil -> @ref BoruvkaMstBackend
 *     (KDTree-accelerated, exact).
 *   - @c N >= @ref primNThreshold and @c d > @ref boruvkaDimCeil -> @ref NnDescentMstBackend
 *     (approximate kNN-graph + Kruskal with a connectivity fallback).
 *
 * Thresholds are preprocessor-overridable via @c CLUSTERING_HDBSCAN_PRIM_N_THRESHOLD and
 * @c CLUSTERING_HDBSCAN_BORUVKA_DIM_CEIL. A compile-time invariant ties @ref primNThreshold to
 * the Prim backend's MRD-matrix byte budget (@ref kPrimMrdMatrixByteBudget) so no override
 * choice can push Prim above the documented memory ceiling.
 *
 * Staleness uses an @c (n, d) shape tuple only, mirroring @c AutoSeeder. The dispatcher does
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
#ifdef CLUSTERING_HDBSCAN_PRIM_N_THRESHOLD
  /**
   * @brief Auto-dispatch threshold on @c n below which the exact dense-MRD Prim backend is
   *        preferred; at or above this threshold the dispatch routes to Boruvka or NN-Descent on
   *        dimension.
   *
   * Override with @c -DCLUSTERING_HDBSCAN_PRIM_N_THRESHOLD=<value>.
   */
  static constexpr std::size_t primNThreshold = CLUSTERING_HDBSCAN_PRIM_N_THRESHOLD;
#else
  static constexpr std::size_t primNThreshold = 5000;
#endif

#ifdef CLUSTERING_HDBSCAN_BORUVKA_DIM_CEIL
  /**
   * @brief Dimensional ceiling above which the KDTree-based Boruvka backend gives way to the
   *        NN-Descent approximate backend.
   *
   * Override with @c -DCLUSTERING_HDBSCAN_BORUVKA_DIM_CEIL=<value>.
   */
  static constexpr std::size_t boruvkaDimCeil = CLUSTERING_HDBSCAN_BORUVKA_DIM_CEIL;
#else
  static constexpr std::size_t boruvkaDimCeil = 60;
#endif

  // Budget guard: Prim must never be selected at an @c (N, d) combination whose dense MRD matrix
  // would exceed its documented byte budget. The guard binds the dispatch threshold to the Prim
  // backend's budget so no override choice can violate the memory invariant.
  static_assert(primNThreshold * primNThreshold * sizeof(T) <= kPrimMrdMatrixByteBudget,
                "primNThreshold^2 * sizeof(T) exceeds the Prim MRD-matrix byte budget; lower "
                "CLUSTERING_HDBSCAN_PRIM_N_THRESHOLD or raise the Prim budget.");

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
    if (n < primNThreshold) {
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
