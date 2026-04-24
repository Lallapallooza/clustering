#!/usr/bin/env bash
# Reject Doxygen docstrings that will render as broken inline-code chips.
#
# Doxygen's @c command is a one-token inline-code directive. It stops at any
# non-word character, so @c foo[i], @c Foo<T>, @c a.b(), @c +inf, @c (x, y)
# and @c ||x|| render as half-chips with surrounding text bare. Use Markdown
# backticks for anything that isn't a single plain identifier.
#
# This hook also flags docstrings whose backtick count is odd (an unclosed
# span swallows the rest of the comment) and other common rendering traps.
#
# Accepts paths as arguments (pre-commit passes the staged files). Exits 1
# and prints every offending line when it finds issues.

set -euo pipefail

fail=0
files=("$@")

if [[ ${#files[@]} -eq 0 ]]; then
  mapfile -t files < <(find include -name '*.h' -o -name '*.hpp' 2>/dev/null)
fi

say() { printf '%s\n' "$*" >&2; }

# Pattern 1: @c followed by a non-word leading char -- always breaks.
#   @c ||x||, @c (n, d), @c [0, k), @c {a, b}, @c sum_{...}, @c +inf, @c <=
bad_start=$(grep -nE '@c [^a-zA-Z0-9_:+*/.-]' "${files[@]}" 2>/dev/null || true)
if [[ -n "$bad_start" ]]; then
  say "ERROR: @c with non-word leading char (use Markdown backticks):"
  printf '%s\n' "$bad_start" >&2
  fail=1
fi

# Pattern 2: @c identifier followed by [ ( or < -- chip clips mid-expression.
#   @c foo[i], @c foo(x), @c Foo<T>
bad_cont=$(grep -nE '@c [A-Za-z_][A-Za-z0-9_]*[\[(<]' "${files[@]}" 2>/dev/null || true)
if [[ -n "$bad_cont" ]]; then
  say "ERROR: @c identifier followed by [ / ( / < (use backticks around the whole expression):"
  printf '%s\n' "$bad_cont" >&2
  fail=1
fi

# Pattern 3: @c a.b(...) dotted calls -- chip stops before the open paren.
bad_dotcall=$(grep -nE '@c [A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*\(' "${files[@]}" 2>/dev/null || true)
if [[ -n "$bad_dotcall" ]]; then
  say "ERROR: @c dotted call (use backticks):"
  printf '%s\n' "$bad_dotcall" >&2
  fail=1
fi

# Pattern 4: triple or stray-adjacent backticks -- broken inline-code span.
triple_bt=$(grep -nE '```' "${files[@]}" 2>/dev/null || true)
if [[ -n "$triple_bt" ]]; then
  say "ERROR: triple backticks (not a code fence inside a C comment -- use single backticks):"
  printf '%s\n' "$triple_bt" >&2
  fail=1
fi

# Pattern 5: odd backtick count inside a /** ... */ docstring -- unclosed span.
odd_bt=$(awk '
  /\/\*\*/  { inblock = 1; buf = ""; start = NR }
  inblock   { buf = buf $0 "\n" }
  /\*\//    && inblock {
    n = gsub(/`/, "`", buf)
    if (n % 2 != 0) {
      printf "%s:%d: odd backtick count (%d) inside docstring\n", FILENAME, start, n
    }
    inblock = 0
  }
' "${files[@]}" 2>/dev/null || true)
if [[ -n "$odd_bt" ]]; then
  say "ERROR: docstring with unclosed backtick span:"
  printf '%s\n' "$odd_bt" >&2
  fail=1
fi

exit "$fail"
