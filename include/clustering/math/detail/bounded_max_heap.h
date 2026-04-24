#pragma once

#include <cassert>
#include <cstddef>
#include <utility>
#include <vector>

#include "clustering/math/detail/heap_ops.h"

namespace clustering::math::detail {

/**
 * @brief Fixed-capacity binary max-heap of `(key, val)` pairs with deterministic tie-break.
 *
 * Collecting the @c k smallest-key entries from an unsorted stream: insert everything through
 * @ref push, and the heap retains the @c k entries with the smallest keys seen. The heap's root
 * is always the largest-key entry among those retained, so a future candidate is admitted only
 * when its key is strictly smaller than the current root (or equal with a smaller @c Val slot).
 * After all inserts, callers read back in decreasing-key order via @ref top / @ref pop.
 *
 * @par Tie-break rule
 * Equal keys are ordered by @c Val: the entry with the smaller @c Val sorts as "smaller overall"
 * and is preferentially retained. On an admission with equal key and smaller @c Val than the
 * current root, the root is evicted and the new entry takes its place. On equal key and equal-or
 * -larger @c Val, the incoming entry is ignored. This rule is load-bearing for kNN collection:
 * the @c Val slot holds the neighbour index, so identical distances resolve on smaller index and
 * results are reproducible bit-for-bit across runs at the same input.
 *
 * @par Complexity
 * @c push is `O(log capacity)`, @c top is `O(1)`, @c pop is `O(log capacity)`.
 *
 * @tparam Key Orderable key type; larger keys sit at the root.
 * @tparam Val Payload carried alongside the key. Must be equality-comparable to participate in
 *         the tie-break rule.
 */
template <class Key, class Val> class BoundedMaxHeap {
public:
  /**
   * @brief Construct an empty heap that retains at most @p capacity entries.
   *
   * @param capacity Maximum number of entries the heap will keep. Allocations are reserved up
   *                 front so @ref push never reallocates.
   */
  explicit BoundedMaxHeap(std::size_t capacity) : m_capacity(capacity) { m_heap.reserve(capacity); }

  /**
   * @brief Admit @p key, @p val as a candidate.
   *
   * If the heap has spare capacity, the entry is inserted unconditionally. Otherwise the new
   * entry is compared against the root: a strictly smaller key evicts the root; equal key with a
   * strictly smaller @c Val also evicts the root; anything else is ignored. The tie-break rule
   * keeps kNN collection reproducible across runs.
   *
   * @param key Ordering key.
   * @param val Payload; participates in the deterministic tie-break rule.
   */
  void push(Key key, Val val) {
    if (m_heap.size() < m_capacity) {
      m_heap.emplace_back(std::move(key), std::move(val));
      siftUp(m_heap, m_heap.size() - 1, maxWithTieBreak);
      return;
    }
    if (m_capacity == 0) {
      return;
    }
    if (pairLess(key, val, m_heap.front().first, m_heap.front().second)) {
      m_heap.front() = std::pair<Key, Val>(std::move(key), std::move(val));
      siftDown(m_heap, 0, maxWithTieBreak);
    }
  }

  /**
   * @brief Largest-key entry currently retained.
   *
   * Asserts on an empty heap; callers must guard with @ref empty.
   *
   * @return Const reference to the root `(key, val)` pair.
   */
  [[nodiscard]] const std::pair<Key, Val> &top() const noexcept {
    assert(!m_heap.empty() && "BoundedMaxHeap::top on empty heap");
    return m_heap.front();
  }

  /**
   * @brief Remove the largest-key entry.
   *
   * Swaps the root with the tail, pops the tail, then sifts the new root down. `O(log n)`.
   * Asserts on an empty heap.
   */
  void pop() noexcept {
    assert(!m_heap.empty() && "BoundedMaxHeap::pop on empty heap");
    const std::size_t last = m_heap.size() - 1;
    if (last != 0) {
      m_heap[0] = std::move(m_heap[last]);
    }
    m_heap.pop_back();
    if (!m_heap.empty()) {
      siftDown(m_heap, 0, maxWithTieBreak);
    }
  }

  /// Whether the heap holds zero entries.
  [[nodiscard]] bool empty() const noexcept { return m_heap.empty(); }

  /// Current number of entries retained.
  [[nodiscard]] std::size_t size() const noexcept { return m_heap.size(); }

  /// Maximum number of entries the heap will retain (set at construction).
  [[nodiscard]] std::size_t capacity() const noexcept { return m_capacity; }

  /// Remove every entry without changing @ref capacity.
  void clear() noexcept { m_heap.clear(); }

private:
  /**
   * @brief Total-order predicate implementing the (key, val) tie-break.
   *
   * `(ak, av)` is less than `(bk, bv)` when @c ak < bk, or `ak == bk` and @c av < bv. Equality
   * of both coordinates yields @c false (strict ordering).
   */
  static bool pairLess(const Key &ak, const Val &av, const Key &bk, const Val &bv) noexcept {
    if (ak < bk) {
      return true;
    }
    if (bk < ak) {
      return false;
    }
    return av < bv;
  }

  /// Max-heap root-ward ordering with tie-break: @c a comes before @c b iff @c b is less than @c a.
  static bool maxWithTieBreak(const std::pair<Key, Val> &a, const std::pair<Key, Val> &b) noexcept {
    return pairLess(b.first, b.second, a.first, a.second);
  }

  std::vector<std::pair<Key, Val>> m_heap;
  std::size_t m_capacity;
};

} // namespace clustering::math::detail
