#pragma once

#include <citor/coherence_cache.h>
#include <citor/thread_pool.h>
#include <citor/version.h>
#include <unistd.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

// Policy layer over citor's coherence-probe mechanism. citor exports a pool's
// one-time inter-core latency probe as a portable blob (`exportCoherenceProbe`)
// and reseeds its process-wide cache from one (`importCoherenceProbe`); this
// header owns where that blob lives on disk, how the cache key is derived, and
// when an upgrade or a topology change invalidates it. A short-lived process
// can then replay a prior run's probe and skip the live calibration.

namespace clustering::math::detail {

// FNV-1a over a byte string. Turns machine identity into a stable,
// filename-safe key so blobs from different CPUs or citor builds never collide.
inline std::uint64_t coherenceKeyHash(std::string_view text) noexcept {
  constexpr std::uint64_t offsetBasis = 1469598103934665603ULL;
  constexpr std::uint64_t prime = 1099511628211ULL;
  std::uint64_t hash = offsetBasis;
  for (const char ch : text) {
    hash ^= static_cast<std::uint8_t>(ch);
    hash *= prime;
  }
  return hash;
}

// Lower-case 16-digit hex of a 64-bit value, fixed width for a stable filename.
inline std::string coherenceKeyHex(std::uint64_t value) {
  static constexpr char digits[] = "0123456789abcdef";
  std::string out(16, '0');
  for (int i = 15; i >= 0; --i) {
    out[static_cast<std::size_t>(i)] = digits[value & 0xFU];
    value >>= 4U;
  }
  return out;
}

// First `model name` line from /proc/cpuinfo, or empty when unreadable. Folds
// into the cache key so a different microarchitecture never reuses a blob whose
// measured latency ratios do not describe this machine.
inline std::string cpuModelName() {
  std::ifstream cpuinfo{"/proc/cpuinfo"};
  if (!cpuinfo) {
    return {};
  }
  std::string line;
  while (std::getline(cpuinfo, line)) {
    if (line.compare(0, 10, "model name") != 0) {
      continue;
    }
    const auto colon = line.find(':');
    if (colon == std::string::npos) {
      return {};
    }
    const auto start = line.find_first_not_of(" \t", colon + 1);
    return start == std::string::npos ? std::string{} : line.substr(start);
  }
  return {};
}

// Base cache directory honouring XDG, with the clustering subdir appended.
// Empty when neither XDG_CACHE_HOME nor HOME is set, which disables caching.
inline std::filesystem::path coherenceCacheDir() {
  if (const char *xdg = std::getenv("XDG_CACHE_HOME"); xdg != nullptr && xdg[0] != '\0') {
    return std::filesystem::path{xdg} / "clustering";
  }
  if (const char *home = std::getenv("HOME"); home != nullptr && home[0] != '\0') {
    return std::filesystem::path{home} / ".cache" / "clustering";
  }
  return {};
}

// Cache file for `workerCount` workers on this machine and citor build. The key
// mixes the CPU model (machine identity), the citor version (an upgrade rotates
// the blob format), and the worker count (different pool sizes pin different
// cpusets). Empty path when the base dir cannot be resolved.
inline std::filesystem::path coherenceCachePath(std::size_t workerCount) {
  const std::filesystem::path base = coherenceCacheDir();
  if (base.empty()) {
    return {};
  }
  std::string material = cpuModelName();
  material += '\n';
  material += CITOR_VERSION_STRING;
  material += '\n';
  material += std::to_string(workerCount);
  return base / ("coherence-" + coherenceKeyHex(coherenceKeyHash(material)) + ".bin");
}

/// Seed citor's process-wide probe cache from the persisted blob for
/// @p workerCount workers. Returns true only when the file existed and citor
/// accepted the blob; a missing file, a read error, or a rejected blob returns
/// false so the next pool runs a cold probe. Never throws.
inline bool importPersistedCoherenceProbe(std::size_t workerCount) noexcept {
  try {
    const std::filesystem::path path = coherenceCachePath(workerCount);
    if (path.empty()) {
      return false;
    }
    std::error_code ec;
    if (!std::filesystem::exists(path, ec) || ec) {
      return false;
    }
    std::ifstream in{path, std::ios::binary};
    if (!in) {
      return false;
    }
    const std::vector<char> raw{std::istreambuf_iterator<char>{in},
                                std::istreambuf_iterator<char>{}};
    if (raw.empty()) {
      return false;
    }
    const std::span<const std::byte> bytes{reinterpret_cast<const std::byte *>(raw.data()),
                                           raw.size()};
    return citor::importCoherenceProbe(bytes);
  } catch (...) {
    return false;
  }
}

/// Persist @p pool's freshly computed coherence probe for @p workerCount
/// workers. Writes to a pid-suffixed temp file and renames it over the cache
/// path so a concurrent first run never observes a half-written blob. Pools
/// with no probe (single-worker, arena) export an empty blob and write nothing.
/// Best-effort: any I/O failure is swallowed, since the probe is a startup
/// optimization and must never fail a fit. Never throws.
inline void exportPersistedCoherenceProbe(const citor::ThreadPool &pool,
                                          std::size_t workerCount) noexcept {
  try {
    const std::vector<std::byte> blob = citor::exportCoherenceProbe(pool);
    if (blob.empty()) {
      return;
    }
    const std::filesystem::path path = coherenceCachePath(workerCount);
    if (path.empty()) {
      return;
    }
    std::error_code ec;
    std::filesystem::create_directories(path.parent_path(), ec);
    std::filesystem::path tmp = path;
    tmp += ".tmp." + std::to_string(static_cast<long>(::getpid()));
    {
      std::ofstream out{tmp, std::ios::binary | std::ios::trunc};
      if (!out) {
        return;
      }
      out.write(reinterpret_cast<const char *>(blob.data()),
                static_cast<std::streamsize>(blob.size()));
      if (!out) {
        out.close();
        std::filesystem::remove(tmp, ec);
        return;
      }
    }
    std::filesystem::rename(tmp, path, ec);
    if (ec) {
      std::filesystem::remove(tmp, ec);
    }
  } catch (...) {
    // best-effort: a cache write failure leaves the next run to re-probe.
  }
}

} // namespace clustering::math::detail
