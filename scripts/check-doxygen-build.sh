#!/usr/bin/env bash
# Rebuild the Doxygen site and fail on any warning.
#
# The generated Doxyfile carries `WARN_AS_ERROR = FAIL_ON_WARNINGS_PRINT`, so
# doxygen itself exits non-zero after printing every warning. The first
# invocation configures a dedicated `build-docs/` tree (fetches the
# doxygen-awesome-css theme via CPM); subsequent runs are incremental and
# typically finish in a few seconds.
#
# Works in three environments:
#   1. CI / non-NixOS local with tools installed (apt / brew / dnf / pacman):
#      uses them directly from PATH.
#   2. NixOS / nix-shell available: auto-wraps with
#      `nix-shell -p doxygen graphviz cmake ninja` so users do not need to
#      pre-activate a shell.
#   3. Tools missing and no nix: prints install hints for common platforms.

set -euo pipefail

missing=()
for bin in doxygen dot cmake ninja; do
  command -v "$bin" >/dev/null 2>&1 || missing+=("$bin")
done

if [[ ${#missing[@]} -gt 0 ]]; then
  if command -v nix-shell >/dev/null 2>&1; then
    exec nix-shell -p doxygen graphviz cmake ninja --run "bash $0 $*"
  fi
  {
    echo "error: missing on PATH: ${missing[*]}"
    echo
    echo "Install the docs toolchain, then re-run:"
    echo "  Debian/Ubuntu: sudo apt-get install -y doxygen graphviz cmake ninja-build"
    echo "  Fedora/RHEL:   sudo dnf install -y doxygen graphviz cmake ninja-build"
    echo "  Arch:          sudo pacman -S --needed doxygen graphviz cmake ninja"
    echo "  macOS:         brew install doxygen graphviz cmake ninja"
    echo "  NixOS:         nix-shell -p doxygen graphviz cmake ninja (auto-wrapped when present)"
  } >&2
  exit 1
fi

build_dir=build-docs

if [[ ! -f "$build_dir/Doxyfile" ]]; then
  cmake -S . -B "$build_dir" -G Ninja \
    -DCLUSTERING_BUILD_DOCS=ON \
    -DCLUSTERING_BUILD_TESTS=OFF \
    -DCLUSTERING_BUILD_BENCHMARK=OFF >/dev/null
fi

cmake --build "$build_dir" --target docs
