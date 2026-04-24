#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>
#include <vector>

namespace clustering::math::detail {

/**
 * @brief Fixed-capacity top-@c k tracker of @c (key, val) pairs with @c O(1) reject fast path.
 *
 * Accumulates the @c k entries with the smallest keys from an unsorted stream. Unlike a heap
 * this keeps the retained pairs in insertion order in two parallel arrays and caches the slot
 * holding the current worst (largest) key. The common "new key is not smaller than the worst
 * retained" path returns in @c O(1) without touching memory; on a successful replace, the
 * worst slot is refreshed with an @c O(capacity) rescan. For small @c k (the kNN use case
 * drives this with @c k typically in @c [4, 32]), the linear rescan beats a heap sift by
 * avoiding pair swaps, function-pointer comparator indirection, and the associated branch
 * misses. @ref drainAscending exports the retained pairs sorted ascending by key and resets
 * the tracker to empty for reuse.
 *
 * @par Tie-break rule
 * Matches @ref BoundedMaxHeap: among equal keys the entry with the smaller @c Val sorts as
 * "smaller overall" and is preferentially retained. A new candidate with equal key and smaller
 * @c Val than the current worst evicts the worst; equal key with equal-or-larger @c Val is
 * ignored.
 *
 * @tparam Key Orderable key type.
 * @tparam Val Orderable payload type; participates in the tie-break rule.
 */
template <class Key, class Val> class TopKNeighbors {
public:
  /**
   * @brief Construct an empty tracker retaining at most @p capacity entries.
   */
  explicit TopKNeighbors(std::size_t capacity) : m_capacity(capacity) {
    m_keys.resize(capacity);
    m_vals.resize(capacity);
  }

  /// Remove every entry without changing @ref capacity. Does not deallocate.
  void clear() noexcept {
    m_size = 0;
    m_worstSlot = 0;
  }

  /// Whether the tracker is at capacity.
  [[nodiscard]] bool full() const noexcept { return m_size == m_capacity; }

  /// Number of entries currently retained.
  [[nodiscard]] std::size_t size() const noexcept { return m_size; }

  /// Maximum number of entries the tracker retains.
  [[nodiscard]] std::size_t capacity() const noexcept { return m_capacity; }

  /**
   * @brief Current pruning bound: the largest retained key, or @c +inf when not yet full.
   *
   * The kNN walker uses this as its subtree prune threshold; returning @c +inf until the
   * tracker fills lets every subtree in early iterations pass the gap test unconditionally.
   */
  [[nodiscard]] Key boundKey() const noexcept {
    if (m_size < m_capacity) {
      return std::numeric_limits<Key>::max();
    }
    return m_keys[m_worstSlot];
  }

  /**
   * @brief Admit @p key, @p val into the retained set.
   *
   * If the tracker has spare capacity, the entry is inserted unconditionally and the worst
   * slot is refreshed only if the new entry is larger than the current worst. Once full,
   * rejects in @c O(1) when the incoming entry does not beat the worst; otherwise overwrites
   * the worst slot and rescans to locate the new worst.
   */
  void push(Key key, Val val) noexcept {
    if (m_size < m_capacity) {
      m_keys[m_size] = key;
      m_vals[m_size] = val;
      if (m_size == 0 || pairLess(m_keys[m_worstSlot], m_vals[m_worstSlot], key, val)) {
        m_worstSlot = m_size;
      }
      ++m_size;
      return;
    }
    if (m_capacity == 0) {
      return;
    }
    const Key worstKey = m_keys[m_worstSlot];
    const Val worstVal = m_vals[m_worstSlot];
    if (!pairLess(key, val, worstKey, worstVal)) {
      return;
    }
    m_keys[m_worstSlot] = key;
    m_vals[m_worstSlot] = val;
    // Rescan capacity slots to locate the new worst. Capacity is small in the driving use
    // case (kNN with minSamples typically in [4, 32]); a linear scan fits in a handful of
    // cache lines and its branches predict well once steady-state.
    std::size_t newWorst = 0;
    Key newWorstKey = m_keys[0];
    Val newWorstVal = m_vals[0];
    for (std::size_t s = 1; s < m_capacity; ++s) {
      if (pairLess(newWorstKey, newWorstVal, m_keys[s], m_vals[s])) {
        newWorstKey = m_keys[s];
        newWorstVal = m_vals[s];
        newWorst = s;
      }
    }
    m_worstSlot = newWorst;
  }

  /**
   * @brief Export retained entries sorted ascending by @c (key, val) and reset to empty.
   *
   * The first @ref size entries of @p keyOut / @p valOut are overwritten. Entries beyond
   * @ref size are untouched, matching the kNN output layout where the tail is padded by the
   * driver. Tie-break identical to the @ref push rule: equal keys sort by smaller @c Val.
   */
  template <class KeyOut, class ValOut>
  void drainAscending(KeyOut *keyOut, ValOut *valOut) noexcept {
    // Index-sort the retained slots ascending; avoids shuffling the parallel arrays in place
    // when capacity is small (the common kNN case is k in [4, 32]).
    std::array<std::size_t, kMaxInlineOrder> order{};
    std::vector<std::size_t> orderHeap;
    std::size_t *ord = order.data();
    if (m_size > kMaxInlineOrder) {
      orderHeap.resize(m_size);
      ord = orderHeap.data();
    }
    for (std::size_t i = 0; i < m_size; ++i) {
      ord[i] = i;
    }
    std::sort(ord, ord + m_size, [this](std::size_t a, std::size_t b) noexcept {
      return pairLess(m_keys[a], m_vals[a], m_keys[b], m_vals[b]);
    });
    for (std::size_t i = 0; i < m_size; ++i) {
      const std::size_t s = ord[i];
      keyOut[i] = static_cast<KeyOut>(m_keys[s]);
      valOut[i] = static_cast<ValOut>(m_vals[s]);
    }
    clear();
  }

private:
  /// Kept small enough that the inline order array fits on the stack for the typical HDBSCAN
  /// minSamples; taller k falls back to a vector. 64 covers minSamples up to 64 with zero
  /// heap traffic per query.
  static constexpr std::size_t kMaxInlineOrder = 64;

  static bool pairLess(const Key &ak, const Val &av, const Key &bk, const Val &bv) noexcept {
    if (ak < bk) {
      return true;
    }
    if (bk < ak) {
      return false;
    }
    return av < bv;
  }

  std::vector<Key> m_keys;
  std::vector<Val> m_vals;
  std::size_t m_capacity;
  std::size_t m_size = 0;
  std::size_t m_worstSlot = 0;
};

} // namespace clustering::math::detail
