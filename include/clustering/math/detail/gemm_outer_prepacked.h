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
 * @brief Drive the Goto-style triple-nested Mc/Nc/Kc loop against a pre-packed B buffer.
 *
 * Mirrors @c gemmRunReference except @c packB is replaced by indexing into @p prepackedBp.
 * The pre-packed layout the caller must supply:
 *   - For each @c jcIdx in `ceil(nDim / kNc<T>)`, each @c pcIdx in `ceil(kDim / kKc<T>)`,
 *     in `(jcIdx, pcIdx)` row-major order, store a Goto-packed `(kc x roundedNc)` region
 *     produced by @c packB for that `(jc, pc, nc, kc)` quadruple.
 *   - @c roundedNc = `ceil(currentNc / kKernelNr<T>)` * kKernelNr<T>, i.e. the packer's
 *     panel-count times @c Nr, matching @c packB's output capacity exactly.
 *   - Total element count: for each @c jcIdx, @c kDim * roundedNc(jcIdx); summed across
 *     all @c jcIdx blocks.
 *
 * Same K==0 BLAS contract as @c gemmRunReference: applies @c C := beta * C and returns.
 * Empty M or N is a no-op.
 *
 * @tparam T Element type (@c float or @c double).
 * @param Ad Read-only descriptor of A.
 * @param prepackedBp Pointer to the pre-packed B buffer laid out as documented above.
 * @param kDim Inner dimension (@c A.cols == B.rows); redundant with @c Ad.cols but used for
 *             offset arithmetic into @p prepackedBp so the function does not need to re-derive it.
 * @param nDim Column count of B (@c C.cols); consulted for @c nc / @c roundedNc arithmetic.
 * @param Cd Mutable descriptor of C.
 * @param alpha Scalar multiplier on @c A*B.
 * @param beta  Scalar multiplier on the prior @c C.
 * @param apArena Caller-owned scratch of capacity `kMc<T>`*kKc<T> elements (per worker slice).
 * @param pool   Parallelism injection. When @c pool.shouldParallelize is @c true, the Mc-tile
 *               loop fans out via @c submit_blocks; each task indexes its own arena slice via
 *               @c Pool::workerIndex(). Otherwise the function runs serial on slice 0.
 */
template <class T>
void gemmRunPrepacked(::clustering::detail::MatrixDescC<T> Ad, const T *prepackedBp,
                      std::size_t kDim, std::size_t nDim, ::clustering::detail::MatrixDesc<T> Cd,
                      T alpha, T beta, T *apArena, ::clustering::math::Pool pool) noexcept {
  constexpr std::size_t kMr = kKernelMr<T>;
  constexpr std::size_t kNr = kKernelNr<T>;
  constexpr std::size_t kMcVal = kMc<T>;
  constexpr std::size_t kNcVal = kNc<T>;
  constexpr std::size_t kKcVal = kKc<T>;

  const std::size_t M = Cd.rows;
  const std::size_t N = nDim;
  const std::size_t K = kDim;

  T *cBase = Cd.ptr;
  const std::ptrdiff_t cRowStride = Cd.rowStride;
  const std::ptrdiff_t cColStride = Cd.colStride;

  if (M == 0 || N == 0) {
    return;
  }

  if (K == 0) {
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
#endif

  // The pre-packed buffer stores jc-blocks back-to-back. Within a jc-block, the pc-sub-blocks
  // are laid out in pc order with element count kc * roundedNc per pc-block; the kc values sum
  // to K, so each jc-block's total size is K * roundedNc(jcIdx). Walk the offset alongside the
  // outer-loop iteration rather than precomputing a jc-offset table -- keeps state local.
  std::size_t jcBase = 0;
  for (std::size_t jc = 0; jc < N; jc += kNcVal) {
    const std::size_t nc = (jc + kNcVal <= N) ? kNcVal : (N - jc);
    const std::size_t roundedNc = ((nc + kNr - 1) / kNr) * kNr;

    std::size_t pcOffInJc = 0;
    for (std::size_t pc = 0; pc < K; pc += kKcVal) {
      const std::size_t kc = (pc + kKcVal <= K) ? kKcVal : (K - pc);
      const bool firstKBlock = (pc == 0);
      const T effBeta = firstKBlock ? beta : T{1};
      const auto kernel = (effBeta == T{0}) ? kernelZero : kernelGeneral;

      const T *bpArena = prepackedBp + jcBase + pcOffInJc;

      // Same Mc-tile parallel dispatch shape as gemmRunReference -- each task gets its own
      // apArena slice. The serial path falls through to slice 0 because Pool::workerIndex()
      // returns 0 outside a pool task body.
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
        for (std::size_t mcIdx = 0; mcIdx < mcBlockCount; ++mcIdx) {
          runOneMcBlock(mcIdx, apArena);
        }
      }

      pcOffInJc += kc * roundedNc;
    }

    jcBase += K * roundedNc;
  }
}

} // namespace clustering::math::detail
