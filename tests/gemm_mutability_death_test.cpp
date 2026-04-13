#include <gtest/gtest.h>

#include <array>
#include <cstddef>

#include "clustering/math/gemm.h"
#include "clustering/math/thread.h"
#include "clustering/ndarray.h"

using clustering::NDArray;
using clustering::math::gemm;
using clustering::math::Pool;

// Read-only C borrowed via the const-pointer factory: m_mutable is false, yet the returned type
// is NDArray<float, 2, Layout::Contig> so it binds to gemm's mutable-reference parameter. The
// always-assert at gemm entry fires before describeMatrixMut's debug-only assert gets a chance,
// so the death signal matches on the always-assert format regardless of build mode.
TEST(GemmMutabilityDeath, ConstBorrowedCAbortsAtPublicEntry) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");

  alignas(32) std::array<float, 64> aData{};
  alignas(32) std::array<float, 64> bData{};
  alignas(32) std::array<float, 64> cData{};
  aData.fill(1.0F);
  bData.fill(1.0F);
  cData.fill(0.0F);

  auto A = NDArray<float, 2>::borrow(aData.data(), {8, 8});
  auto B = NDArray<float, 2>::borrow(bData.data(), {8, 8});
  auto C = NDArray<float, 2>::borrow(static_cast<const float *>(cData.data()), {8, 8});
  ASSERT_FALSE(C.isMutable());

  EXPECT_DEATH(gemm(A, B, C, Pool{nullptr}, 1.0F, 0.0F),
               "always-assert failed: C\\.isMutable\\(\\)");
}
