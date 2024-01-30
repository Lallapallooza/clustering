#pragma once
#include <cstdint>
#include <type_traits>

template<typename T>
class LinearAllocator {
  static_assert(std::is_trivially_destructible<T>::value, "T must be trivially destructible");

 public:
  LinearAllocator(size_t size)
    : size(size * sizeof(T)),
      memory(new char[size * sizeof(T)]),
      next(memory) {}

  ~LinearAllocator() {
    delete[] memory;
  }

  LinearAllocator(const LinearAllocator &) = delete;
  LinearAllocator &operator=(const LinearAllocator &) = delete;

  T *allocate() {
    if (static_cast<size_t>(next - memory) >= size) {
      throw std::bad_alloc();
    }

    T *result = reinterpret_cast<T *>(next);
    next += sizeof(T);
    return new(result) T;
  }

  void deallocate(T *ptr) {
    // Do nothing because T is trivially destructible
  }

  void reset() {
    next = memory;
  }

  bool isDeallocSupported() {
    return false;
  }

 private:
  size_t size;
  char *memory;
  char *next;
};