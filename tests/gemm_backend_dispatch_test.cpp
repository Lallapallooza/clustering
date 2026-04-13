#include <gtest/gtest.h>

#include <cstddef>

#include "clustering/math/gemm.h"
#include "clustering/math/thread.h"
#include "clustering/ndarray.h"

using clustering::Layout;
using clustering::NDArray;
using clustering::math::gemm;
using clustering::math::Pool;

namespace {

// Sentinel backend tag. Satisfies the same run<T, LA, LB> shape as detail::ReferenceGemm; its
// only job is to fill C with a known sentinel so the test can prove dispatch went through
// Backend::run rather than the default ReferenceGemm.
struct FakeBackend {
  template <class T, Layout LA, Layout LB>
  static void run(const NDArray<T, 2, LA> & /*A*/, const NDArray<T, 2, LB> & /*B*/,
                  NDArray<T, 2> &C, Pool /*pool*/, T /*alpha*/, T /*beta*/) {
    for (std::size_t i = 0; i < C.dim(0); ++i) {
      for (std::size_t j = 0; j < C.dim(1); ++j) {
        C(i, j) = T{42};
      }
    }
  }
};

} // namespace

TEST(GemmBackendDispatch, ExplicitBackendOverrideRoutes) {
  NDArray<float, 2> A({4, 4});
  NDArray<float, 2> B({4, 4});
  NDArray<float, 2> C({4, 4});
  for (std::size_t i = 0; i < 4; ++i) {
    for (std::size_t j = 0; j < 4; ++j) {
      A(i, j) = 1.0F;
      B(i, j) = 1.0F;
      C(i, j) = 0.0F;
    }
  }

  gemm<float, Layout::Contig, Layout::Contig, FakeBackend>(A, B, C, Pool{nullptr}, 1.0F, 0.0F);

  // Reference backend over (1*A*B + 0*C) would yield 4.0F per cell (4 lanes of K=4, all ones);
  // FakeBackend writes the 42 sentinel. Finding 42 -- not 4 -- proves the override took effect.
  for (std::size_t i = 0; i < 4; ++i) {
    for (std::size_t j = 0; j < 4; ++j) {
      EXPECT_FLOAT_EQ(C(i, j), 42.0F);
    }
  }
}

TEST(GemmBackendDispatch, FakeBackendExplicitLayoutsReceivesCall) {
  const NDArray<float, 2> A({3, 5});
  const NDArray<float, 2> B({5, 3});
  NDArray<float, 2> C({3, 3});
  for (std::size_t i = 0; i < 3; ++i) {
    for (std::size_t j = 0; j < 3; ++j) {
      C(i, j) = 0.0F;
    }
  }

  // Pin the four-argument template-slot ordering against accidental reorder.
  gemm<float, Layout::Contig, Layout::Contig, FakeBackend>(A, B, C, Pool{nullptr}, 1.0F, 0.0F);
  for (std::size_t i = 0; i < 3; ++i) {
    for (std::size_t j = 0; j < 3; ++j) {
      EXPECT_FLOAT_EQ(C(i, j), 42.0F);
    }
  }
}
