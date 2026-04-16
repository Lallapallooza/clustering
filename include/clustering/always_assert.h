#pragma once

#include <cstdio>
#include <cstdlib>

namespace clustering {

/**
 * @brief Emits a diagnostic to @c stderr and terminates via @c std::abort.
 *
 * The format is fixed ("clustering: always-assert failed: <cond> at <file>:<line>\n") so death
 * tests can match the output with a stable regex. @c std::abort is chosen over @c std::terminate
 * so GoogleTest's @c EXPECT_DEATH catches the signal without routing through the terminate
 * handler.
 */
[[noreturn]] inline void alwaysAssertFail(const char *cond, const char *file, int line) noexcept {
  std::fprintf(stderr, "clustering: always-assert failed: %s at %s:%d\n", cond, file, line);
  std::abort();
}

} // namespace clustering

/**
 * @brief Release-active assertion: evaluates @p cond in every build configuration.
 *
 * Unlike @c assert, this check survives @c -DNDEBUG. Use it at public API entry points whose
 * failure would corrupt memory or trigger undefined behavior past the debug boundary -- writes
 * through a read-only borrow, null output buffers, etc.
 */
#define CLUSTERING_ALWAYS_ASSERT(cond)                                                             \
  do {                                                                                             \
    if (!(cond)) {                                                                                 \
      ::clustering::alwaysAssertFail(#cond, __FILE__, __LINE__);                                   \
    }                                                                                              \
  } while (0)
