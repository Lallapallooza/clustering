#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

#include "clustering/always_assert.h"
#include "clustering/hdbscan/mst_output.h"
#include "clustering/index/nn_descent.h"
#include "clustering/math/dsu.h"
#include "clustering/math/thread.h"
#include "clustering/ndarray.h"

namespace clustering::hdbscan {

/**
 * @brief Tuning knobs for the NN-Descent MST backend.
 *
 * @c kExtra widens the kNN-graph @c k beyond @c minSamples so the MRD edge set carries enough
 * candidates for Kruskal to recover an MST close to the exact Prim result. @c maxIter, @c delta,
 * and @c seed forward directly to @c NnDescentIndex.
 */
struct NnDescentMstConfig {
  /// Extra neighbours per node on top of @c minSamples; the kNN graph is built with
  /// @c k = minSamples + kExtra. Sized for Dong 2011's @c k = 15 recall target at default
  /// @c min_samples = 5. Larger values trade build time for MRD-edge coverage; raise this
  /// when the input is near-uniform and the MRD-MST topology depends on edges beyond each
  /// point's 15 nearest.
  std::size_t kExtra = 10;
  /// Iteration cap on the NN-Descent join loop; forwarded to @c NnDescentIndex.
  std::size_t maxIter = 10;
  /// Convergence threshold on the update fraction; forwarded to @c NnDescentIndex.
  float delta = 0.001F;
  /// PRNG seed for RP-tree partition choices; forwarded to @c NnDescentIndex.
  std::uint64_t seed = 0;
};

/**
 * @brief Approximate minimum-spanning-tree backend over mutual-reachability distance via an
 *        NN-Descent-built kNN graph plus Kruskal, with a connectivity fallback that closes any
 *        disconnected components.
 *
 * Given a point matrix @p X and a core-distance parameter @p minSamples, the backend:
 *   1. Builds (or reuses) an @ref NnDescentIndex at @c k = @c minSamples + @c kExtra.
 *   2. Extracts per-point core distances as the @c minSamples-th nearest squared distance from
 *      the kNN graph.
 *   3. Promotes the kNN edges to an MRD-weighted undirected edge list:
 *      @c MRD(i, j) = max(coreDist[i], coreDist[j], sqDist(i, j)).
 *   4. Runs Kruskal on the sorted MRD edge list to produce a spanning forest.
 *   5. When the forest spans fewer than @c n components, enumerates minimum-weight bridges
 *      between every pair of disjoint components and adds @c c - 1 bridges Kruskal-style until
 *      the tree is connected.
 *
 * The fallback enumerates @c n_a * n_b * d arithmetic for each pair of components; it is
 * expected-rare on typical inputs where @c k = minSamples + kExtra yields a connected kNN graph.
 *
 * Satisfies @ref MstBackendStrategy. The kNN graph is data-dependent and is rebuilt on every
 * @ref run call; caching it across fits would risk silent failure under in-place mutation
 * of the caller's buffer. Only shape-free scratch (@c out buffers sized to the current
 * input) amortises across calls at fixed shape.
 *
 * @tparam T Element type of the point cloud. Only @c float is supported.
 */
template <class T> class NnDescentMstBackend {
  static_assert(
      std::is_same_v<T, float>,
      "NnDescentMstBackend<T> supports only float; a double specialization is out of scope.");

public:
  /**
   * @brief Default-construct with the default @ref NnDescentMstConfig.
   *
   * Required for the @ref MstBackendStrategy concept. Callers who need custom tuning use
   * @ref NnDescentMstBackend(NnDescentMstConfig).
   */
  NnDescentMstBackend() = default;

  /**
   * @brief Construct with an explicit tuning @p config.
   *
   * Distinct from the zero-arg default ctor to avoid the default-constructor ambiguity in
   * @c [class.default.ctor]/2.
   */
  explicit NnDescentMstBackend(NnDescentMstConfig config) : m_config(config) {}

  /**
   * @brief Build the approximate MRD-weighted minimum spanning tree of @p X.
   *
   * @pre @p minSamples is positive and strictly less than @c X.dim(0).
   *
   * @param X          Contiguous @c (n x d) dataset; caller retains ownership.
   * @param minSamples Neighbour count driving the core-distance definition.
   * @param pool       Worker pool; forwarded to the inner @c NnDescentIndex build.
   * @param out        Destination; @c edges receives the @c n - 1 MST edges in insertion order
   *                   and @c coreDistances is sized to @c n.
   */
  void run(const NDArray<T, 2> &X, std::size_t minSamples, math::Pool pool, MstOutput<T> &out) {
    const std::size_t n = X.dim(0);
    CLUSTERING_ALWAYS_ASSERT(minSamples >= 1);
    CLUSTERING_ALWAYS_ASSERT(minSamples < n);

    out.edges.clear();
    out.edges.reserve(n - 1);
    out.coreDistances = NDArray<T, 1>(std::array<std::size_t, 1>{n});

    // Phase 1: rebuild the NN-Descent kNN graph per fit. The kNN graph encodes the current
    // input's neighbour structure; caching it across fits would risk silent failure under
    // in-place mutation of the caller's buffer. The index is reconstructed on every call so
    // the graph always reflects the data @c run was invoked with.
    const std::size_t k = minSamples + m_config.kExtra;
    CLUSTERING_ALWAYS_ASSERT(k < n);
    m_index.emplace(k, m_config.maxIter, m_config.delta, m_config.seed);
    m_index->build(X, pool);

    const auto &neighbors = m_index->neighbors();

    // Phase 2: core distance per point is the minSamples-th nearest squared distance. The
    // NN-Descent graph returns entries sorted ascending by squared distance; index minSamples - 1
    // picks the correct slot. We assume the graph's k is at least minSamples so the slot exists.
    T *coreDistData = out.coreDistances.data();
    for (std::size_t i = 0; i < n; ++i) {
      CLUSTERING_ALWAYS_ASSERT(neighbors[i].size() >= minSamples);
      coreDistData[i] = neighbors[i][minSamples - 1].sqDist;
    }

    // Phase 3: MRD-weighted edge list. The kNN graph is directed (i -> j when j is a neighbour of
    // i); to form an undirected edge set we emit (min(i,j), max(i,j), weight) and dedupe by
    // sorting plus a forward sweep. The weight takes the max of core[i], core[j], and sqDist(i,j).
    struct Edge {
      T weight;
      std::int32_t u;
      std::int32_t v;
    };
    std::vector<Edge> edges;
    edges.reserve(n * k);
    for (std::size_t i = 0; i < n; ++i) {
      const T coreI = coreDistData[i];
      for (const auto &e : neighbors[i]) {
        const auto j = static_cast<std::size_t>(e.idx);
        if (j == i) {
          continue;
        }
        const T coreJ = coreDistData[j];
        T w = e.sqDist;
        if (coreI > w) {
          w = coreI;
        }
        if (coreJ > w) {
          w = coreJ;
        }
        const auto u32 = static_cast<std::int32_t>(std::min(i, j));
        const auto v32 = static_cast<std::int32_t>(std::max(i, j));
        edges.push_back(Edge{w, u32, v32});
      }
    }

    // Sort ascending by weight then by endpoints so Kruskal consumes a deterministic order and
    // duplicate (u, v) entries from the directed kNN view become adjacent. We do not explicitly
    // dedupe: Kruskal's union-find check rejects a second edge between the same pair implicitly.
    std::sort(edges.begin(), edges.end(), [](const Edge &a, const Edge &b) {
      if (a.weight != b.weight) {
        return a.weight < b.weight;
      }
      if (a.u != b.u) {
        return a.u < b.u;
      }
      return a.v < b.v;
    });

    // Phase 4: Kruskal on the sorted edge list. The UnionFind is keyed on the signed int32 MST
    // index type; cast via uint32 for the union-find that requires an unsigned index.
    UnionFind<std::uint32_t> uf(n);
    for (const Edge &e : edges) {
      if (uf.unite(static_cast<std::uint32_t>(e.u), static_cast<std::uint32_t>(e.v))) {
        out.edges.push_back(MstEdge<T>{e.u, e.v, e.weight});
        if (out.edges.size() + 1 == n) {
          break;
        }
      }
    }

    // Phase 5: connectivity fallback. When the kNN-graph MRD-MST leaves more than one component,
    // enumerate minimum-weight bridges between every pair of disjoint components and Kruskal them
    // in until a single spanning tree remains. Each bridge candidate is the (i, j) MRD edge of
    // minimum weight where i and j live in distinct components; weight uses the true squared
    // Euclidean distance computed row-by-row, then lifted by max with both core distances.
    if (uf.countComponents() > 1) {
      resolveDisconnectedComponents(X, coreDistData, uf, out);
    }
  }

private:
  /**
   * @brief Walk every cross-component (i, j) pair to pick the minimum-weight MRD bridge per
   *        component pair, then Kruskal those bridges into the MST until connectivity is reached.
   *
   * @c n_a * n_b * d scalar work per component pair; expected-rare on typical inputs where the
   * kNN graph at @c k = minSamples + kExtra already spans every point. Correctness is the
   * invariant: the spanning tree grows to @c n - 1 edges regardless of how many components the
   * fallback starts with.
   */
  void resolveDisconnectedComponents(const NDArray<T, 2> &X, const T *coreDistData,
                                     UnionFind<std::uint32_t> &uf, MstOutput<T> &out) {
    const std::size_t n = X.dim(0);
    const std::size_t d = X.dim(1);

    // Materialize the current component membership as a root -> member list so subsequent
    // all-pairs scans iterate only over the members of each component rather than the full
    // point set per pair.
    std::vector<std::vector<std::uint32_t>> members;
    std::vector<std::uint32_t> rootToSlot(n, std::numeric_limits<std::uint32_t>::max());
    for (std::uint32_t i = 0; i < static_cast<std::uint32_t>(n); ++i) {
      const std::uint32_t r = uf.find(i);
      std::uint32_t slot = rootToSlot[r];
      if (slot == std::numeric_limits<std::uint32_t>::max()) {
        slot = static_cast<std::uint32_t>(members.size());
        rootToSlot[r] = slot;
        members.emplace_back();
      }
      members[slot].push_back(i);
    }

    while (uf.countComponents() > 1) {
      // For each pair of surviving components (a, b) find the minimum-MRD bridge. Compare every
      // member of a to every member of b; track the minimum-weight edge. When all pairs are
      // scanned, Kruskal the resulting bridges into the MST. The outer loop runs c - 1 times in
      // the worst case but typically terminates after one round because inserting a single bridge
      // collapses components transitively.
      struct Bridge {
        T weight;
        std::int32_t u;
        std::int32_t v;
      };
      std::vector<Bridge> bridges;
      bridges.reserve(members.size() * (members.size() - 1) / 2);

      for (std::size_t a = 0; a < members.size(); ++a) {
        if (members[a].empty()) {
          continue;
        }
        for (std::size_t b = a + 1; b < members.size(); ++b) {
          if (members[b].empty()) {
            continue;
          }
          // If the two component slots now share a root (after a Kruskal in a prior iteration),
          // skip the pair.
          if (uf.find(members[a][0]) == uf.find(members[b][0])) {
            continue;
          }
          T bestW = std::numeric_limits<T>::infinity();
          std::int32_t bestU = 0;
          std::int32_t bestV = 0;
          for (const std::uint32_t ia : members[a]) {
            const T coreI = coreDistData[ia];
            const T *rowI = X.data() + (static_cast<std::size_t>(ia) * d);
            for (const std::uint32_t jb : members[b]) {
              const T coreJ = coreDistData[jb];
              const T *rowJ = X.data() + (static_cast<std::size_t>(jb) * d);
              T sq = T{0};
              for (std::size_t t = 0; t < d; ++t) {
                const T diff = rowI[t] - rowJ[t];
                sq += diff * diff;
              }
              T w = sq;
              if (coreI > w) {
                w = coreI;
              }
              if (coreJ > w) {
                w = coreJ;
              }
              if (w < bestW) {
                bestW = w;
                bestU = static_cast<std::int32_t>(ia);
                bestV = static_cast<std::int32_t>(jb);
              }
            }
          }
          if (bestW != std::numeric_limits<T>::infinity()) {
            bridges.push_back(Bridge{bestW, bestU, bestV});
          }
        }
      }

      // Sort bridges ascending by weight and Kruskal them into the MST. A bridge whose endpoints
      // are already merged (via a prior bridge in the same round) is skipped by union-find.
      std::sort(bridges.begin(), bridges.end(),
                [](const Bridge &a, const Bridge &b) { return a.weight < b.weight; });
      bool progress = false;
      for (const Bridge &br : bridges) {
        if (uf.unite(static_cast<std::uint32_t>(br.u), static_cast<std::uint32_t>(br.v))) {
          out.edges.push_back(MstEdge<T>{br.u, br.v, br.weight});
          progress = true;
          if (out.edges.size() + 1 == n) {
            break;
          }
        }
      }

      // Guard against an infinite loop: if no bridge was accepted the graph cannot reach
      // connectivity. In practice the precondition (cross-component points exist) guarantees a
      // finite bridge and this branch is never taken.
      CLUSTERING_ALWAYS_ASSERT(progress);

      // Rebuild the component membership for the next iteration: a merge may collapse multiple
      // slots into one root, so the next round's all-pairs scan should only visit surviving
      // components. Slots whose root changed get their members absorbed into the winning slot.
      if (uf.countComponents() > 1) {
        std::vector<std::vector<std::uint32_t>> nextMembers;
        std::fill(rootToSlot.begin(), rootToSlot.end(), std::numeric_limits<std::uint32_t>::max());
        for (const auto &comp : members) {
          for (const std::uint32_t i : comp) {
            const std::uint32_t r = uf.find(i);
            std::uint32_t slot = rootToSlot[r];
            if (slot == std::numeric_limits<std::uint32_t>::max()) {
              slot = static_cast<std::uint32_t>(nextMembers.size());
              rootToSlot[r] = slot;
              nextMembers.emplace_back();
            }
            nextMembers[slot].push_back(i);
          }
        }
        members = std::move(nextMembers);
      }
    }
  }

  NnDescentMstConfig m_config{};
  std::optional<index::NnDescentIndex<T>> m_index;
};

} // namespace clustering::hdbscan
