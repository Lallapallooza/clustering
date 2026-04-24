#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

#include "clustering/math/detail/heap_ops.h"

namespace clustering {

/**
 * @brief Binary min-heap of `(key, val)` pairs ordered on @c key.
 *
 * Plain d=2 heap backed by a single @c std::vector. @c push, @c top, @c pop are the full surface;
 * no bulk-build, no merge, no handles. Use @ref IndexedHeap when a stable handle and
 * @c decreaseKey are needed.
 *
 * @tparam Key Orderable key type; `operator<` defines the ordering. Smaller keys pop first.
 * @tparam Val Payload carried alongside the key.
 */
template <class Key, class Val> class BinaryHeap {
public:
  BinaryHeap() = default;

  /**
   * @brief Insert a new `(key, val)` entry.
   *
   * Appends at the tail and sifts up to restore the heap invariant; `O(log n)`.
   *
   * @param key Ordering key.
   * @param val Payload.
   */
  void push(Key key, Val val) {
    m_heap.emplace_back(std::move(key), std::move(val));
    math::detail::siftUp(m_heap, m_heap.size() - 1, minOnKey);
  }

  /**
   * @brief Smallest-key entry currently in the heap.
   *
   * Asserts on an empty heap -- the caller must guard with @ref empty.
   *
   * @return Const reference to the top `(key, val)` pair.
   */
  [[nodiscard]] const std::pair<Key, Val> &top() const noexcept {
    assert(!m_heap.empty() && "BinaryHeap::top on empty heap");
    return m_heap.front();
  }

  /**
   * @brief Remove the smallest-key entry.
   *
   * Swaps the root with the tail, pops the tail, then sifts the new root down. `O(log n)`.
   * Asserts on an empty heap.
   */
  void pop() noexcept {
    assert(!m_heap.empty() && "BinaryHeap::pop on empty heap");
    const std::size_t last = m_heap.size() - 1;
    if (last != 0) {
      m_heap[0] = std::move(m_heap[last]);
    }
    m_heap.pop_back();
    if (!m_heap.empty()) {
      math::detail::siftDown(m_heap, 0, minOnKey);
    }
  }

  /**
   * @brief Current number of entries.
   *
   * @return Entry count.
   */
  [[nodiscard]] std::size_t size() const noexcept { return m_heap.size(); }

  /**
   * @brief Whether the heap holds zero entries.
   *
   * @return @c true iff `size()` == 0.
   */
  [[nodiscard]] bool empty() const noexcept { return m_heap.empty(); }

private:
  /// Min-heap ordering on the key component. Comes-before means smaller key sits closer to root.
  static bool minOnKey(const std::pair<Key, Val> &a, const std::pair<Key, Val> &b) noexcept {
    return a.first < b.first;
  }

  std::vector<std::pair<Key, Val>> m_heap;
};

/**
 * @brief Binary min-heap keyed on @c Key with `O(1)` handle-to-position lookup.
 *
 * Each heap entry carries an external @c Idx handle drawn from a fixed range `[0, capacity)`.
 * At most one entry per handle is in the heap at any time; the handle is cleared on @c pop and on
 * construction. @ref decreaseKey looks up the handle's position through @c m_posMap and sifts up;
 * `O(log n)` worst case. Backed by heap-allocated vectors; allocation is constrained to
 * construction and the occasional @c m_heap growth, never inside @c push / @c pop / @c decreaseKey.
 *
 * @tparam Key Orderable key type; smaller keys pop first.
 * @tparam Val Payload carried alongside the key.
 * @tparam Idx Unsigned integer handle type; defaults to @c uint32_t. Handles index into a
 *         @c capacity-sized position map.
 */
template <class Key, class Val, class Idx = std::uint32_t> class IndexedHeap {
public:
  static_assert(std::is_unsigned_v<Idx>, "IndexedHeap Idx must be an unsigned integer type");

  /// Sentinel for "handle not currently in the heap". Equal to @c size_t's max.
  static constexpr std::size_t kNotInHeap = std::numeric_limits<std::size_t>::max();

  /**
   * @brief One heap entry.
   *
   * Public so callers can read the handle returned by @c top / @c pop without unpacking a tuple.
   */
  struct Entry {
    /// External identity supplied by the caller at @c push time.
    Idx handle;
    /// Ordering key; the heap root carries the smallest key.
    Key key;
    /// Payload associated with @c handle.
    Val val;
  };

  /**
   * @brief Construct an empty heap capable of handles in `[0, capacity)`.
   *
   * Allocates the position map to @p capacity entries filled with @c kNotInHeap.
   *
   * @param capacity Upper bound on handle values.
   */
  explicit IndexedHeap(std::size_t capacity) : m_posMap(capacity, kNotInHeap) {}

  /**
   * @brief Insert an entry for @p handle.
   *
   * Asserts @p handle is in range and not already present. Sifts up to restore the heap
   * invariant; `O(log n)`.
   *
   * @param handle External identity of the entry; must be less than @c capacity.
   * @param key Ordering key.
   * @param val Payload.
   */
  void push(Idx handle, Key key, Val val) {
    assert(static_cast<std::size_t>(handle) < m_posMap.size() &&
           "IndexedHeap::push handle out of range");
    assert(m_posMap[handle] == kNotInHeap && "IndexedHeap::push handle already in heap");
    const std::size_t pos = m_heap.size();
    m_heap.push_back(Entry{handle, std::move(key), std::move(val)});
    m_posMap[handle] = pos;
    siftUp(pos);
  }

  /**
   * @brief Whether @p handle is currently present in the heap.
   *
   * @param handle Handle to query; must be less than @c capacity.
   * @return @c true iff an entry for @p handle is present.
   */
  [[nodiscard]] bool contains(Idx handle) const noexcept {
    assert(static_cast<std::size_t>(handle) < m_posMap.size() &&
           "IndexedHeap::contains handle out of range");
    return m_posMap[handle] != kNotInHeap;
  }

  /**
   * @brief Lower the key of @p handle's entry to @p newKey.
   *
   * Asserts @p handle is present and @p newKey is not greater than the current key (the classic
   * @c decreaseKey precondition -- raising a key would require sifting down, a different path).
   * Sifts up from the handle's position; `O(log n)`.
   *
   * @param handle Handle of the entry to update.
   * @param newKey Replacement key; must satisfy `newKey <= current` key.
   */
  void decreaseKey(Idx handle, Key newKey) noexcept {
    assert(static_cast<std::size_t>(handle) < m_posMap.size() &&
           "IndexedHeap::decreaseKey handle out of range");
    const std::size_t pos = m_posMap[handle];
    assert(pos != kNotInHeap && "IndexedHeap::decreaseKey handle not in heap");
    assert(!(m_heap[pos].key < newKey) && "IndexedHeap::decreaseKey newKey > current key");
    m_heap[pos].key = std::move(newKey);
    siftUp(pos);
  }

  /**
   * @brief Smallest-key entry currently in the heap.
   *
   * Asserts on an empty heap.
   *
   * @return Const reference to the top entry.
   */
  [[nodiscard]] const Entry &top() const noexcept {
    assert(!m_heap.empty() && "IndexedHeap::top on empty heap");
    return m_heap.front();
  }

  /**
   * @brief Remove and return the smallest-key entry.
   *
   * Clears the popped handle's slot in @c m_posMap; the handle becomes `contains(h)` == false.
   *
   * @return The popped @ref Entry by value.
   */
  Entry pop() noexcept {
    assert(!m_heap.empty() && "IndexedHeap::pop on empty heap");
    Entry out = std::move(m_heap.front());
    m_posMap[out.handle] = kNotInHeap;
    const std::size_t last = m_heap.size() - 1;
    if (last != 0) {
      m_heap.front() = std::move(m_heap[last]);
      m_posMap[m_heap.front().handle] = 0;
    }
    m_heap.pop_back();
    if (!m_heap.empty()) {
      siftDown(0);
    }
    return out;
  }

  /**
   * @brief Current number of entries.
   *
   * @return Entry count.
   */
  [[nodiscard]] std::size_t size() const noexcept { return m_heap.size(); }

  /**
   * @brief Whether the heap holds zero entries.
   *
   * @return @c true iff `size()` == 0.
   */
  [[nodiscard]] bool empty() const noexcept { return m_heap.empty(); }

  /**
   * @brief Maximum handle + 1 (the capacity passed at construction).
   *
   * @return Size of the internal position map.
   */
  [[nodiscard]] std::size_t capacity() const noexcept { return m_posMap.size(); }

private:
  std::vector<Entry> m_heap;
  std::vector<std::size_t> m_posMap;

  void swapEntries(std::size_t i, std::size_t j) noexcept {
    std::swap(m_heap[i], m_heap[j]);
    m_posMap[m_heap[i].handle] = i;
    m_posMap[m_heap[j].handle] = j;
  }

  void siftUp(std::size_t pos) noexcept {
    while (pos > 0) {
      const std::size_t parent = (pos - 1) / 2;
      if (m_heap[pos].key < m_heap[parent].key) {
        swapEntries(pos, parent);
        pos = parent;
      } else {
        return;
      }
    }
  }

  void siftDown(std::size_t pos) noexcept {
    const std::size_t n = m_heap.size();
    while (true) {
      const std::size_t left = (2 * pos) + 1;
      const std::size_t right = left + 1;
      std::size_t smallest = pos;
      if (left < n && m_heap[left].key < m_heap[smallest].key) {
        smallest = left;
      }
      if (right < n && m_heap[right].key < m_heap[smallest].key) {
        smallest = right;
      }
      if (smallest == pos) {
        return;
      }
      swapEntries(pos, smallest);
      pos = smallest;
    }
  }
};

} // namespace clustering
