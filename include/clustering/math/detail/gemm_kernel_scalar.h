#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace clustering::math::detail {

/**
 * @brief Compile-time tag selecting the kernel epilogue: skip-load on @c kZero,
 *        fused read-modify-write on @c kGeneral.
 *
 * Picking the @c BetaKind at the call site (rather than via a runtime branch inside the kernel)
 * keeps the @c kZero path safe against NaN-valued @c C and lets the compiler fully DCE the
 * @c beta-scaled load when it is provably dead.
 */
enum class BetaKind : std::uint8_t { kZero, kGeneral };

/**
 * @brief Per-element-type microkernel tile dimensions (rows of A panel x cols of B panel).
 *
 * The primary template intentionally yields zero so an unspecialized @c T is a hard build error
 * downstream rather than silently producing a degenerate tile.
 */
template <class T> inline constexpr std::size_t kKernelMr = 0;
template <class T> inline constexpr std::size_t kKernelNr = 0;
template <> inline constexpr std::size_t kKernelMr<float> = 8;
template <> inline constexpr std::size_t kKernelNr<float> = 6;
template <> inline constexpr std::size_t kKernelMr<double> = 4;
template <> inline constexpr std::size_t kKernelNr<double> = 6;

/**
 * @brief Scalar reference microkernel: writes a @c kKernelMr<T> x @c kKernelNr<T> output tile.
 *
 * Buffer layouts (shared with the AVX2 kernel variants that consume the same packed panels):
 *   - @p ap holds a @c kc x @c Mr panel of A, with element @c (r, k) at @c ap[k*Mr + r].
 *   - @p bp holds a @c kc x @c Nr panel of B, with element @c (k, c) at @c bp[k*Nr + c].
 *   - @p tile is a row-major @c Mr x @c Nr scratch buffer; element @c (r, c) at @c tile[r*Nr+c].
 *     The outer loop owns @p tile (32-byte aligned, on its stack) and is responsible for
 *     pre-loading it with @c beta-scaled @c C contributions when @c Beta == @c kGeneral.
 *     Cells outside the valid @c (mcTail x ncTail) sub-rectangle must be pre-zeroed by the
 *     outer loop so the kernel's writes to padded cells are harmless.
 *
 * Epilogue:
 *   - @c kZero:    @c tile[r*Nr+c] = alpha * acc[r][c] (kernel never reads @p tile).
 *   - @c kGeneral: @c tile[r*Nr+c] = alpha * acc[r][c] + beta * tile[r*Nr+c].
 *
 * @tparam T    Element type; @c float or @c double.
 * @tparam Beta Compile-time BetaKind selecting the epilogue.
 * @param ap    Packed A panel.
 * @param bp    Packed B panel.
 * @param tile  Mr x Nr scratch tile (row-major); read on @c kGeneral, written on both kinds.
 * @param kc    K-dimension of the panels (number of inner products to accumulate).
 * @param alpha Scalar multiplier for the @c A*B product.
 * @param beta  Scalar multiplier for the prior tile content; ignored on @c kZero.
 */
template <class T, BetaKind Beta>
void gemmKernelMrNrScalar(const T *ap, const T *bp, T *tile, std::size_t kc, T alpha,
                          T beta) noexcept {
  constexpr std::size_t kMr = kKernelMr<T>;
  constexpr std::size_t kNr = kKernelNr<T>;

  // __restrict__ locals captured at the kernel top per the codegen-discipline convention: the
  // AVX2 clones rely on the compiler proving non-aliasing across the K-loop before it can pin
  // accumulators in YMM registers.
  const T *__restrict__ apLocal = ap;
  const T *__restrict__ bpLocal = bp;
  T *__restrict__ tLocal = tile;
  const std::size_t kcLocal = kc;

  std::array<std::array<T, kNr>, kMr> acc{};

  for (std::size_t k = 0; k < kcLocal; ++k) {
    const T *aRow = apLocal + (k * kMr);
    const T *bRow = bpLocal + (k * kNr);
    for (std::size_t r = 0; r < kMr; ++r) {
      const T aVal = aRow[r];
      for (std::size_t c = 0; c < kNr; ++c) {
        acc[r][c] += aVal * bRow[c];
      }
    }
  }

  if constexpr (Beta == BetaKind::kZero) {
    for (std::size_t r = 0; r < kMr; ++r) {
      for (std::size_t c = 0; c < kNr; ++c) {
        tLocal[(r * kNr) + c] = alpha * acc[r][c];
      }
    }
  } else {
    for (std::size_t r = 0; r < kMr; ++r) {
      for (std::size_t c = 0; c < kNr; ++c) {
        const T prior = tLocal[(r * kNr) + c];
        tLocal[(r * kNr) + c] = (alpha * acc[r][c]) + (beta * prior);
      }
    }
  }
}

} // namespace clustering::math::detail
