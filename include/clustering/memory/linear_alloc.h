#pragma once
#include <cstdint>
#include <new>
#include <type_traits>
#include <utility>

namespace clustering {

template <typename T> class LinearAllocator {
  static_assert(std::is_trivially_destructible_v<T>, "T must be trivially destructible");

public:
  /// Reserves room for @p count objects of @c T from a single backing allocation.
  LinearAllocator(std::size_t count)
      : size(count * sizeof(T)), memory(new char[count * sizeof(T)]), next(memory) {}

  ~LinearAllocator() { delete[] memory; }

  LinearAllocator(const LinearAllocator &) = delete;
  LinearAllocator &operator=(const LinearAllocator &) = delete;

  /// Bump-allocates one @c T; throws @c std::bad_alloc when the arena is exhausted.
  T *allocate() {
    if (std::cmp_greater_equal(next - memory, size)) {
      throw std::bad_alloc();
    }

    // Placement new modifies storage at `next` even though tidy can't see
    // it through the pointer; suppress the false-positive const suggestion.
    T *const result = new (next) T; // NOLINT(misc-const-correctness)
    next += sizeof(T);
    return result;
  }

  /// Bump-allocates @p count contiguous @c T objects in one step; throws @c std::bad_alloc
  /// when fewer than @p count slots remain. Requires `count >= 1`.
  T *allocate(std::size_t count) {
    const auto used = static_cast<std::size_t>(next - memory);
    if (count == 0 || (count * sizeof(T)) > (size - used)) {
      throw std::bad_alloc();
    }

    // Element-wise placement new sidesteps the implementation-defined cookie that array
    // placement new may prepend; trivial default construction folds the loop away.
    T *const result = new (next) T; // NOLINT(misc-const-correctness)
    for (std::size_t i = 1; i < count; ++i) {
      new (next + (i * sizeof(T))) T;
    }
    next += count * sizeof(T);
    return result;
  }

  /// No-op: trivial destructibility lets per-element reclamation be skipped.
  void deallocate(T * /*ptr*/) {
    // Do nothing because T is trivially destructible
  }

  /// Rewinds the bump pointer, reclaiming every outstanding allocation in one shot.
  void reset() { next = memory; }

  /// Reports that per-element @c deallocate is not supported (false for this allocator).
  bool isDeallocSupported() { return false; }

private:
  std::size_t size;
  char *memory;
  char *next;
};

} // namespace clustering
