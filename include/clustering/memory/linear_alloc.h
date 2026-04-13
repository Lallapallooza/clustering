#pragma once
#include <cstdint>
#include <new>
#include <type_traits>
#include <utility>

namespace clustering {

template <typename T> class LinearAllocator {
  static_assert(std::is_trivially_destructible_v<T>, "T must be trivially destructible");

public:
  LinearAllocator(std::size_t count)
      : size(count * sizeof(T)), memory(new char[count * sizeof(T)]), next(memory) {}

  ~LinearAllocator() { delete[] memory; }

  LinearAllocator(const LinearAllocator &) = delete;
  LinearAllocator &operator=(const LinearAllocator &) = delete;

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

  void deallocate(T * /*ptr*/) {
    // Do nothing because T is trivially destructible
  }

  void reset() { next = memory; }

  bool isDeallocSupported() { return false; }

private:
  std::size_t size;
  char *memory;
  char *next;
};

} // namespace clustering
