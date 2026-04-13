#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

#include "clustering/math/detail/gemm_outer_prepacked.h"
#include "clustering/math/detail/gemm_pack.h"
#include "clustering/math/detail/matrix_desc.h"
#include "clustering/math/detail/reference_gemm.h"
#include "clustering/math/thread.h"
#include "clustering/ndarray.h"

namespace clustering::math {

/**
 * @brief Reusable GEMM plan: packs @c B once at construction, amortizes the packing cost across
 *        repeated @c execute calls with varying @c A.
 *
 * Ownership model:
 *   - @c m_Bp holds a @c packB-shaped copy of @c B covering every @c (jcIdx, pcIdx) block; the
 *     source @c B may be destroyed after construction.
 *   - @c m_scratch holds per-worker @c A-packing arenas sized @c workerCount * kMc * kKc.
 *   - @c m_pool is captured by value at construction. @c execute uses it unchanged.
 *
 * Concurrency: @c execute is @c const with respect to the plan's observable shape but mutates
 * @c m_scratch via pointer aliasing into the per-worker slice. Two concurrent @c execute calls
 * on the same plan alias the same scratch; callers that pipeline across application threads
 * construct one plan per thread.
 *
 * Lifetime: move-only. @c m_Bp and @c m_scratch are @c std::vector; move is noexcept.
 *
 * @tparam T Element type (@c float or @c double).
 * @tparam Backend Backend tag; defaulted to @c detail::ReferenceGemm. @c GemmPlan does NOT route
 *         through the backend — it calls @c gemmRunPrepacked directly with its owned @c m_Bp.
 *         The template parameter exists so @c GemmPlan is selected at the same
 *         @c <T, Backend> call site as the one-shot @c gemm entry.
 */
template <class T, class Backend = detail::ReferenceGemm> class GemmPlan {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                "GemmPlan: T must be float or double");

public:
  /**
   * @brief Construct the plan and fully pre-pack @p B into @c m_Bp.
   *
   * @tparam LB Layout tag of B (@c Contig or @c MaybeStrided).
   * @param B   Source matrix (K x N); contents are copied into @c m_Bp, the source may outlive
   *            the plan or vice versa without coupling.
   * @param pool Pool captured by value for use by @c execute; @c pool.workerCount() determines
   *            @c m_scratch size and is fixed for the plan's lifetime.
   */
  template <Layout LB>
  GemmPlan(const NDArray<T, 2, LB> &B, Pool pool)
      : m_kDim(B.dim(0)), m_nDim(B.dim(1)), m_workerCount(pool.workerCount()), m_pool(pool) {
    constexpr std::size_t kNr = detail::kKernelNr<T>;
    constexpr std::size_t kNcVal = detail::kNc<T>;
    constexpr std::size_t kKcVal = detail::kKc<T>;
    constexpr std::size_t kMcVal = detail::kMc<T>;

    m_scratch.assign(m_workerCount * kMcVal * kKcVal, T{0});

    if (m_kDim == 0 || m_nDim == 0) {
      // No Bp storage needed: execute() treats K==0 as the BLAS C<-beta*C identity without
      // reading Bp, and N==0 is a no-op.
      return;
    }

    // Two-pass: first compute total Bp size, then pack. Keeping the sizing pass separate lets
    // us call reserve/resize exactly once; the pack loop walks the same offset arithmetic that
    // gemmRunPrepacked uses, so the two stay structurally locked.
    std::size_t total = 0;
    for (std::size_t jc = 0; jc < m_nDim; jc += kNcVal) {
      const std::size_t nc = (jc + kNcVal <= m_nDim) ? kNcVal : (m_nDim - jc);
      const std::size_t roundedNc = ((nc + kNr - 1) / kNr) * kNr;
      total += m_kDim * roundedNc;
    }
    m_Bp.assign(total, T{0});

    auto Bd = ::clustering::detail::describeMatrix(B);

    std::size_t jcBase = 0;
    for (std::size_t jc = 0; jc < m_nDim; jc += kNcVal) {
      const std::size_t nc = (jc + kNcVal <= m_nDim) ? kNcVal : (m_nDim - jc);
      const std::size_t roundedNc = ((nc + kNr - 1) / kNr) * kNr;

      std::size_t pcOffInJc = 0;
      for (std::size_t pc = 0; pc < m_kDim; pc += kKcVal) {
        const std::size_t kc = (pc + kKcVal <= m_kDim) ? kKcVal : (m_kDim - pc);
        detail::packB<T>(Bd, pc, kc, jc, nc, m_Bp.data() + jcBase + pcOffInJc);
        pcOffInJc += kc * roundedNc;
      }
      jcBase += m_kDim * roundedNc;
    }
  }

  /**
   * @brief Execute the plan: compute @c C := alpha * A * B + beta * C against the pre-packed
   *        @c B captured at construction.
   *
   * @tparam LA Layout tag of A.
   * @param A Input matrix (M x @c kDim()).
   * @param C Output matrix (M x @c nDim()); must be mutable.
   * @param alpha Scalar multiplier on @c A*B; defaults to 1.
   * @param beta  Scalar multiplier on the prior @c C; defaults to 0.
   */
  template <Layout LA>
  void execute(const NDArray<T, 2, LA> &A, NDArray<T, 2> &C, T alpha = T{1},
               T beta = T{0}) const noexcept {
    if (A.dim(0) == 0 || m_nDim == 0) {
      return;
    }

    // Pass the full scratch base — gemmRunPrepacked slices per-worker inside its Mc dispatch via
    // Pool::workerIndex(). On the serial path workerIndex() returns 0, so slice 0 is used.
    auto Ad = ::clustering::detail::describeMatrix(A);
    auto Cd = ::clustering::detail::describeMatrixMut(C);
    detail::gemmRunPrepacked<T>(Ad, m_Bp.data(), m_kDim, m_nDim, Cd, alpha, beta, m_scratch.data(),
                                m_pool);
  }

  /// @brief Inner dimension captured at construction (@c B.rows).
  [[nodiscard]] std::size_t kDim() const noexcept { return m_kDim; }

  /// @brief Column count captured at construction (@c B.cols).
  [[nodiscard]] std::size_t nDim() const noexcept { return m_nDim; }

  /// @brief Debug accessor exposing the packed @c B pointer so tests can pin alignment.
  /// Not part of the stable API.
  [[nodiscard]] const T *debugBpData() const noexcept { return m_Bp.data(); }

  /// @brief Debug accessor exposing the scratch capacity so tests can pin the sizing formula.
  /// Not part of the stable API.
  [[nodiscard]] std::size_t debugScratchSize() const noexcept { return m_scratch.size(); }

  GemmPlan(const GemmPlan &) = delete;
  GemmPlan &operator=(const GemmPlan &) = delete;
  GemmPlan(GemmPlan &&) noexcept = default;
  GemmPlan &operator=(GemmPlan &&) noexcept = default;
  ~GemmPlan() = default;

private:
  std::size_t m_kDim = 0;
  std::size_t m_nDim = 0;
  std::size_t m_workerCount = 1;
  Pool m_pool{};
  std::vector<T, ::clustering::detail::AlignedAllocator<T, 32>> m_Bp;
  // mutable: execute() is const on the plan's observable shape but the scratch is a per-call
  // mutation surface sliced by worker index.
  mutable std::vector<T, ::clustering::detail::AlignedAllocator<T, 32>> m_scratch;
};

} // namespace clustering::math
