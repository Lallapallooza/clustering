#pragma once

#include <cstdint>
#include <type_traits>

namespace clustering {

/**
 * @brief Thin wrapper over @c new T / @c delete that satisfies the library's allocator concept.
 */
template <typename T> class NewAllocator {
public:
  /// Constructs the allocator; the count hint is ignored since allocations are individual.
  NewAllocator(std::size_t /*size*/) {}

  NewAllocator(const NewAllocator &) = delete;
  NewAllocator &operator=(const NewAllocator &) = delete;

  /// Allocates one default-constructed @c T via @c new.
  T *allocate() { return new T; }

  /// Frees a pointer previously returned by @c allocate.
  void deallocate(T *ptr) { delete ptr; }

  /// No-op; the allocator holds no bulk state to rewind.
  void reset() {
    // Do nothing, because we don't have any state
  }

  /// Reports that per-element @c deallocate is supported (true for this allocator).
  bool isDeallocSupported() { return true; }
};

} // namespace clustering
