#pragma once

#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>
#include <vector>

namespace clustering {

/**
 * @brief Disjoint-set-union with iterative path compression and union-by-rank.
 *
 * Stores @c n elements initially as singleton components. @c find compacts the path from @p x
 * to its root in two passes so recursion depth never grows with @c n -- the recursive one-pass
 * form blows the default stack at @c n = 1M. @c unite merges components by attaching the shorter
 * tree under the taller, keeping worst-case tree depth at `O(log n)` even without compression.
 *
 * @tparam Idx Unsigned integer index type; defaults to @c uint32_t. Use @c uint64_t only when
 *         @c n may exceed `2^32`.
 */
template <class Idx = std::uint32_t> class UnionFind {
public:
  static_assert(std::is_unsigned_v<Idx>, "UnionFind Idx must be an unsigned integer type");

  /**
   * @brief Construct @p n singleton components numbered `[0, n)`.
   *
   * @param n Element count; each element starts as its own parent with rank 0 and size 1.
   */
  explicit UnionFind(std::size_t n) : m_parent(n), m_rank(n, 0), m_size(n, 1), m_components(n) {
    for (std::size_t i = 0; i < n; ++i) {
      m_parent[i] = static_cast<Idx>(i);
    }
  }

  /**
   * @brief Root of the component containing @p x, with path compression applied.
   *
   * Two-pass iterative compression: first walk to the root, then rewalk from @p x setting each
   * visited node's parent to the root. The iterative form survives @c n = 1M inputs where the
   * recursive form would overflow the stack.
   *
   * @param x Element index; must satisfy @c x < size().
   * @return Root index of @p x's component.
   */
  Idx find(Idx x) noexcept {
    assert(static_cast<std::size_t>(x) < m_parent.size() && "UnionFind::find index out of range");
    Idx root = x;
    while (m_parent[root] != root) {
      root = m_parent[root];
    }
    Idx node = x;
    while (m_parent[node] != root) {
      const Idx next = m_parent[node];
      m_parent[node] = root;
      node = next;
    }
    return root;
  }

  /**
   * @brief Merge the components containing @p a and @p b.
   *
   * Attaches the shorter tree under the taller; when the two ranks tie, @p a's root becomes the
   * new root and its rank is bumped. No-op if @p a and @p b are already in the same component.
   *
   * @param a First element index.
   * @param b Second element index.
   * @return @c true if the call actually merged two distinct components, @c false if they were
   *         already joined.
   */
  bool unite(Idx a, Idx b) noexcept {
    Idx ra = find(a);
    Idx rb = find(b);
    if (ra == rb) {
      return false;
    }
    if (m_rank[ra] < m_rank[rb]) {
      m_parent[ra] = rb;
      m_size[rb] += m_size[ra];
    } else if (m_rank[ra] > m_rank[rb]) {
      m_parent[rb] = ra;
      m_size[ra] += m_size[rb];
    } else {
      m_parent[rb] = ra;
      m_size[ra] += m_size[rb];
      ++m_rank[ra];
    }
    --m_components;
    return true;
  }

  /**
   * @brief Whether @p a and @p b share a component.
   *
   * Convenience wrapper over `find(a)` == `find(b)`; applies path compression on both paths.
   *
   * @param a First element index.
   * @param b Second element index.
   * @return @c true iff @p a and @p b have the same root.
   */
  bool sameComponent(Idx a, Idx b) noexcept { return find(a) == find(b); }

  /**
   * @brief Current number of distinct components.
   *
   * Starts at @c n and decreases by @c 1 on each successful @c unite.
   *
   * @return Component count.
   */
  [[nodiscard]] std::size_t countComponents() const noexcept { return m_components; }

  /**
   * @brief Total number of elements under management (fixed at construction).
   *
   * @return The @c n passed to the constructor.
   */
  [[nodiscard]] std::size_t size() const noexcept { return m_parent.size(); }

  /**
   * @brief Population of the component whose root is @p root.
   *
   * The caller must pass a root index (typically obtained from @ref find); passing a non-root is
   * undefined by contract. Size is maintained at the root on every @ref unite -- when trees merge,
   * the winning root accumulates the losing root's size so the figure stays accurate without a
   * tree walk at query time.
   *
   * @param root Root index as returned by @ref find.
   * @return Number of elements under @p root's component.
   */
  [[nodiscard]] std::size_t componentSize(Idx root) const noexcept {
    assert(static_cast<std::size_t>(root) < m_parent.size() &&
           "UnionFind::componentSize index out of range");
    return m_size[root];
  }

private:
  std::vector<Idx> m_parent;
  std::vector<std::uint8_t> m_rank;
  std::vector<std::size_t> m_size;
  std::size_t m_components;
};

/**
 * @brief Lock-free disjoint-set-union for concurrent edge folding.
 *
 * Threads call @ref unite concurrently with no external locking: a root is linked under
 * another only through a compare-and-swap on its own parent slot, which can succeed only
 * while it is still a root, so merges linearize. @ref find applies path halving whose
 * racing writes are benign -- parent pointers only ever move toward a root.
 *
 * Links are ordered by index: the larger root always attaches under the smaller, so once
 * every @ref unite has completed (and the callers have synchronized), the root of each
 * component is exactly its minimum member. Flattened roots are therefore reproducible
 * across runs regardless of thread interleaving.
 *
 * @tparam Idx Unsigned integer index type; defaults to @c uint32_t.
 */
template <class Idx = std::uint32_t> class AtomicUnionFind {
public:
  static_assert(std::is_unsigned_v<Idx>, "AtomicUnionFind Idx must be an unsigned integer type");

  /// Construct @p n singleton components numbered `[0, n)`.
  explicit AtomicUnionFind(std::size_t n) : m_parent(n) {
    for (std::size_t i = 0; i < n; ++i) {
      m_parent[i].store(static_cast<Idx>(i), std::memory_order_relaxed);
    }
  }

  /**
   * @brief Root of the component containing @p x at some point during the call.
   *
   * Safe to run concurrently with @ref unite; a concurrent merge may retarget the returned
   * root, so quiescent callers (after a join) read the final component root while racing
   * callers read a then-current one.
   *
   * @param x Element index; must satisfy @c x < size().
   * @return Root index of @p x's component.
   */
  Idx find(Idx x) noexcept {
    assert(static_cast<std::size_t>(x) < m_parent.size() &&
           "AtomicUnionFind::find index out of range");
    while (true) {
      Idx parent = m_parent[x].load(std::memory_order_acquire);
      if (parent == x) {
        return x;
      }
      const Idx grandparent = m_parent[parent].load(std::memory_order_acquire);
      if (grandparent == parent) {
        return parent;
      }
      // Halve the path: retarget x at its grandparent. A lost race means another thread
      // already advanced the slot; the walk continues from the grandparent either way.
      m_parent[x].compare_exchange_weak(parent, grandparent, std::memory_order_release,
                                        std::memory_order_relaxed);
      x = grandparent;
    }
  }

  /**
   * @brief Merge the components containing @p a and @p b; larger root links under smaller.
   *
   * @return @c true if the call merged two distinct components, @c false if they were
   *         already joined when observed.
   */
  bool unite(Idx a, Idx b) noexcept {
    while (true) {
      Idx ra = find(a);
      Idx rb = find(b);
      if (ra == rb) {
        return false;
      }
      if (rb < ra) {
        std::swap(ra, rb);
      }
      Idx expected = rb;
      if (m_parent[rb].compare_exchange_strong(expected, ra, std::memory_order_acq_rel,
                                               std::memory_order_acquire)) {
        return true;
      }
      // rb stopped being a root mid-flight; restart from the advanced positions.
      a = ra;
      b = expected;
    }
  }

  /// Total number of elements under management (fixed at construction).
  [[nodiscard]] std::size_t size() const noexcept { return m_parent.size(); }

private:
  std::vector<std::atomic<Idx>> m_parent;
};

} // namespace clustering
