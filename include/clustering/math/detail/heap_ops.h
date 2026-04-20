#pragma once

#include <cstddef>
#include <utility>

namespace clustering::math::detail {

/**
 * @brief Sift position @p pos up in @p heap until the heap invariant under @p comesBefore holds.
 *
 * Swaps @p pos with its parent while @c comesBefore(heap[pos], heap[parent]) reports true; returns
 * when the invariant holds or @p pos reaches the root. @c O(log heap.size()).
 *
 * @tparam Container  Random-access container indexable with @c operator[] and @c size(). Elements
 *                    must be swappable.
 * @tparam Compare    Strict weak ordering on heap entries. @c comesBefore(a, b) returns @c true
 *                    when @c a should sit closer to the root than @c b.
 * @param heap        The heap storage.
 * @param pos         Starting position; must be less than @c heap.size().
 * @param comesBefore Comparator deciding root-ward ordering.
 */
template <class Container, class Compare>
void siftUp(Container &heap, std::size_t pos, Compare comesBefore) noexcept {
  while (pos > 0) {
    const std::size_t parent = (pos - 1) / 2;
    if (comesBefore(heap[pos], heap[parent])) {
      using std::swap;
      swap(heap[pos], heap[parent]);
      pos = parent;
    } else {
      return;
    }
  }
}

/**
 * @brief Sift position @p pos down in @p heap until the heap invariant under @p comesBefore holds.
 *
 * Swaps @p pos with the child that most strictly comes before under @p comesBefore, repeating
 * until @p pos is a leaf or no child comes before it. @c O(log heap.size()).
 *
 * @tparam Container  Random-access container indexable with @c operator[] and @c size(). Elements
 *                    must be swappable.
 * @tparam Compare    Strict weak ordering on heap entries.
 * @param heap        The heap storage.
 * @param pos         Starting position; must be less than @c heap.size().
 * @param comesBefore Comparator deciding root-ward ordering.
 */
template <class Container, class Compare>
void siftDown(Container &heap, std::size_t pos, Compare comesBefore) noexcept {
  const std::size_t n = heap.size();
  while (true) {
    const std::size_t left = (2 * pos) + 1;
    const std::size_t right = left + 1;
    std::size_t best = pos;
    if (left < n && comesBefore(heap[left], heap[best])) {
      best = left;
    }
    if (right < n && comesBefore(heap[right], heap[best])) {
      best = right;
    }
    if (best == pos) {
      return;
    }
    using std::swap;
    swap(heap[pos], heap[best]);
    pos = best;
  }
}

} // namespace clustering::math::detail
