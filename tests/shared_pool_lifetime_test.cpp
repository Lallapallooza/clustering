#include <pthread.h>
#include <sys/mman.h>

#include <cstddef>
#include <cstdlib>

#include "clustering/math/thread.h"

namespace {

void *useSharedPool(void *) {
  auto &pool = clustering::math::sharedPool(2);
  pool.parallelFor<citor::HintsDefaults>(0, std::size_t{128},
                                         [](std::size_t /*lo*/, std::size_t /*hi*/) {});
  return nullptr;
}

} // namespace

int main() {
  constexpr std::size_t stackSize = 1024 * 1024;
  void *stack =
      mmap(nullptr, stackSize, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (stack == MAP_FAILED) {
    return EXIT_FAILURE;
  }

  pthread_attr_t attr;
  pthread_t producer;
  if (pthread_attr_init(&attr) != 0 || pthread_attr_setstack(&attr, stack, stackSize) != 0 ||
      pthread_create(&producer, &attr, useSharedPool, nullptr) != 0) {
    return EXIT_FAILURE;
  }
  pthread_attr_destroy(&attr);
  if (pthread_join(producer, nullptr) != 0 || munmap(stack, stackSize) != 0) {
    return EXIT_FAILURE;
  }
  return 0;
}
