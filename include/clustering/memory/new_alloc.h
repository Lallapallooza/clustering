#pragma once

#include <cstdint>
#include <type_traits>

template<typename T>
class NewAllocator {
 public:
  NewAllocator(size_t size) {}

  NewAllocator(const NewAllocator &) = delete;
  NewAllocator &operator=(const NewAllocator &) = delete;

  T *allocate() {
    return new T;
  }

  void deallocate(T *ptr) {
    delete ptr;
  }

  void reset() {
    // Do nothing, because we don't have any state
  }

  bool isDeallocSupported() {
    return true;
  }
};