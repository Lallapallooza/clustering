#pragma once

#include <array>
#include <cstddef>
#include <cstring>
#include <type_traits>

#include "clustering/math/detail/gemm_kernel_scalar.h"
#include "clustering/math/detail/gemm_pack.h"
#include "clustering/math/detail/matrix_desc.h"
#include "clustering/math/thread.h"

#ifdef CLUSTERING_USE_AVX2
#include "clustering/math/detail/gemm_kernel_avx2_f32.h"
#endif

namespace clustering::math::detail {

/**
 * @brief Cache-blocking constants for the Goto-style outer loop, sized for AVX2 targets.
 *
 * @c kMc is a multiple of @c kKernelMr<T>, @c kNc a multiple of @c kKernelNr<T>=6. Per-K-tile
 * @c Bp slice is @c kKc*kNc*sizeof(T): 4.18 MB for f32, fits L3 on Zen 3 and Skylake-X.
 * Per-worker @c Ap slice is @c kMc*kKc*sizeof(T): 96 KB for f32 (L2-resident).
 */
template <class T> inline constexpr std::size_t kMc = 96;
template <class T> inline constexpr std::size_t kNc = 4080;
template <class T> inline constexpr std::size_t kKc = 256;

/**
 * @brief Drive the full Goto-style triple-nested Mc/Nc/Kc loop with the scratch-tile tail
 *        protocol; dispatches to the scalar reference microkernel.
 *
 * Treats the BLAS K==0 contract explicitly: when @c Ad.cols == 0 (equivalently @c Bd.rows == 0)
 * the GEMM identity reduces to @c C := beta * C, which the function applies before returning.
 * Empty M or N is a no-op.
 *
 * @tparam T Element type (@c float or @c double).
 * @param Ad Read-only descriptor of A; arbitrary strides supported via the packer.
 * @param Bd Read-only descriptor of B; arbitrary strides supported via the packer.
 * @param Cd Mutable descriptor of C; the caller guarantees mutability and Layout::Contig in v1.
 * @param alpha Scalar multiplier on @c A*B.
 * @param beta  Scalar multiplier on the prior @c C; @c kZero kernel clone selected when zero.
 * @param apArena Caller-owned scratch of capacity @c kMc<T>*kKc<T> elements (per worker).
 * @param bpArena Caller-owned scratch of capacity @c kKc<T>*kNc<T> elements (per call OR plan).
 * @param pool   Parallelism injection. When @c pool.shouldParallelize is @c true, the Mc-tile
 *               loop fans out via @c submit_blocks; each task indexes its own arena slice via
 *               @c Pool::workerIndex(). Otherwise the function runs serial on slice 0.
 */
template <class T>
void gemmRunReference(::clustering::detail::MatrixDescC<T> Ad,
                      ::clustering::detail::MatrixDescC<T> Bd,
                      ::clustering::detail::MatrixDesc<T> Cd, T alpha, T beta, T *apArena,
                      T *bpArena, ::clustering::math::Pool pool) noexcept {
  constexpr std::size_t kMr = kKernelMr<T>;
  constexpr std::size_t kNr = kKernelNr<T>;
  constexpr std::size_t kMcVal = kMc<T>;
  constexpr std::size_t kNcVal = kNc<T>;
  constexpr std::size_t kKcVal = kKc<T>;

  const std::size_t M = Cd.rows;
  const std::size_t N = Cd.cols;
  const std::size_t K = Ad.cols;

  // Hoist the descriptor fields the inner loops touch into locals; the alias-analysis trap
  // documented for NDArray's stride-honoring accessor applies equally here.
  T *cBase = Cd.ptr;
  const std::ptrdiff_t cRowStride = Cd.rowStride;
  const std::ptrdiff_t cColStride = Cd.colStride;

  if (M == 0 || N == 0) {
    return;
  }

  if (K == 0) {
    // Honour the BLAS K==0 contract: C <- beta*C (equivalent to no accumulation).
    const bool zero = (beta == T{0});
    for (std::size_t i = 0; i < M; ++i) {
      for (std::size_t j = 0; j < N; ++j) {
        T &cell = cBase[(static_cast<std::ptrdiff_t>(i) * cRowStride) +
                        (static_cast<std::ptrdiff_t>(j) * cColStride)];
        cell = zero ? T{0} : (beta * cell);
      }
    }
    return;
  }

  using KernelFn = void (*)(const T *, const T *, T *, std::size_t, T, T) noexcept;
  KernelFn kernelZero = &gemmKernelMrNrScalar<T, BetaKind::kZero>;
  KernelFn kernelGeneral = &gemmKernelMrNrScalar<T, BetaKind::kGeneral>;
#ifdef CLUSTERING_USE_AVX2
  if constexpr (std::is_same_v<T, float>) {
    kernelZero = &gemmKernel8x6Avx2F32<BetaKind::kZero>;
    kernelGeneral = &gemmKernel8x6Avx2F32<BetaKind::kGeneral>;
  }
  // f64 AVX2 kernel plugs in here when the 4x6 variant lands.
#endif

  // Goto order: jc (Nc) outermost, then pc (Kc), then ic (Mc); pack-A inside the pc loop and
  // pack-B once per jc. The first Kc-pass in each jc/ic uses the caller's beta; subsequent
  // Kc-passes accumulate (beta=1, kGeneral path) so partial sums are not overwritten.
  for (std::size_t jc = 0; jc < N; jc += kNcVal) {
    const std::size_t nc = (jc + kNcVal <= N) ? kNcVal : (N - jc);

    for (std::size_t pc = 0; pc < K; pc += kKcVal) {
      const std::size_t kc = (pc + kKcVal <= K) ? kKcVal : (K - pc);
      const bool firstKBlock = (pc == 0);
      const T effBeta = firstKBlock ? beta : T{1};
      const auto kernel = (effBeta == T{0}) ? kernelZero : kernelGeneral;

      packB<T>(Bd, pc, kc, jc, nc, bpArena);

      // Dispatch one Mc-row-slab per task, each task indexing its own slice of apArena. The
      // lambda is identical between the serial-fallback path and the submit_blocks path — the
      // only difference is which worker slice it sees (serial always gets slice 0 because
      // Pool::workerIndex() returns 0 outside a pool task).
      auto runOneMcBlock = [&](std::size_t mcIdx, T *apSlice) noexcept {
        const std::size_t ic = mcIdx * kMcVal;
        const std::size_t mc = (ic + kMcVal <= M) ? kMcVal : (M - ic);
        packA<T>(Ad, ic, mc, pc, kc, apSlice);

        for (std::size_t ir = 0; ir < mc; ir += kMr) {
          const std::size_t mTail = (ir + kMr <= mc) ? kMr : (mc - ir);
          const std::size_t panelA = ir / kMr;
          const T *apPanel = apSlice + (panelA * kMr * kc);

          for (std::size_t jr = 0; jr < nc; jr += kNr) {
            const std::size_t nTail = (jr + kNr <= nc) ? kNr : (nc - jr);
            const std::size_t panelB = jr / kNr;
            const T *bpPanel = bpArena + (panelB * kc * kNr);

            alignas(32) std::array<T, kMr * kNr> tile{};

            // For kGeneral, pre-load the valid (mTail x nTail) sub-rectangle of C (the kernel
            // handles the beta multiply) and zero the padded cells so writes to those cells
            // are arithmetic noise that the writeback discards. For kZero the kernel writes
            // every cell unconditionally; tile contents pre-call are dead.
            if (effBeta != T{0}) {
              for (std::size_t c = 0; c < kNr; ++c) {
                for (std::size_t r = 0; r < kMr; ++r) {
                  if (r < mTail && c < nTail) {
                    const T &cell = cBase[(static_cast<std::ptrdiff_t>(ic + ir + r) * cRowStride) +
                                          (static_cast<std::ptrdiff_t>(jc + jr + c) * cColStride)];
                    tile[(c * kMr) + r] = cell;
                  } else {
                    tile[(c * kMr) + r] = T{0};
                  }
                }
              }
            }

            kernel(apPanel, bpPanel, tile.data(), kc, alpha, effBeta);

            for (std::size_t c = 0; c < nTail; ++c) {
              for (std::size_t r = 0; r < mTail; ++r) {
                T &cell = cBase[(static_cast<std::ptrdiff_t>(ic + ir + r) * cRowStride) +
                                (static_cast<std::ptrdiff_t>(jc + jr + c) * cColStride)];
                cell = tile[(c * kMr) + r];
              }
            }
          }
        }
      };

      const std::size_t mcBlockCount = (M + kMcVal - 1) / kMcVal;
      if (pool.shouldParallelize(mcBlockCount, 1, 2)) {
        // shouldParallelize returning true implies pool.pool != nullptr; the analyzer
        // cannot see through the inlined correlation in Pool::shouldParallelize.
        // NOLINTNEXTLINE(clang-analyzer-core.CallAndMessage)
        pool.pool
            ->submit_blocks(std::size_t{0}, mcBlockCount,
                            [&](std::size_t blockStart, std::size_t blockEnd) {
                              T *apSlice = apArena + (::clustering::math::Pool::workerIndex() *
                                                      kMcVal * kKcVal);
                              for (std::size_t mcIdx = blockStart; mcIdx < blockEnd; ++mcIdx) {
                                runOneMcBlock(mcIdx, apSlice);
                              }
                            })
            .wait();
      } else {
        // Serial path — single worker slice; workerIndex() == 0 outside any pool task.
        for (std::size_t mcIdx = 0; mcIdx < mcBlockCount; ++mcIdx) {
          runOneMcBlock(mcIdx, apArena);
        }
      }
    }
  }
}

} // namespace clustering::math::detail
