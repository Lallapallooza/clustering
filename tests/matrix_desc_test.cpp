#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "clustering/math/detail/matrix_desc.h"
#include "clustering/ndarray.h"

namespace {

// A tiny microkernel stub: reads a single element through strides exactly as a real inner loop
// would, so the test exercises the descriptor fields rather than accidentally routing through
// NDArray's own indexing.
template <class T>
T readStrided(const clustering::detail::MatrixDescC<T> &d, std::size_t r, std::size_t c) {
  return d.ptr[static_cast<std::ptrdiff_t>(r) * d.rowStride +
               static_cast<std::ptrdiff_t>(c) * d.colStride];
}

} // namespace

TEST(MatrixDesc, DescribeOwnedContiguous) {
  NDArray<float, 2> a({3, 5});
  for (std::size_t i = 0; i < 3; ++i) {
    for (std::size_t j = 0; j < 5; ++j) {
      a[i][j] = static_cast<float>(i * 5 + j);
    }
  }

  auto d = clustering::detail::describeMatrix(a);
  static_assert(std::is_same_v<decltype(d.ptr), const float *>,
                "describeMatrix must return const-polarity descriptor");
  EXPECT_EQ(d.ptr, a.data());
  EXPECT_EQ(d.rows, 3u);
  EXPECT_EQ(d.cols, 5u);
  EXPECT_EQ(d.rowStride, static_cast<std::ptrdiff_t>(5));
  EXPECT_EQ(d.colStride, static_cast<std::ptrdiff_t>(1));
  EXPECT_TRUE(d.isContiguous);
  EXPECT_GE(d.alignment, 32u);
}

TEST(MatrixDesc, DescribeTransposedStrided) {
  NDArray<float, 2> a({4, 6});
  for (std::size_t i = 0; i < 4; ++i) {
    for (std::size_t j = 0; j < 6; ++j) {
      a[i][j] = static_cast<float>(i * 6 + j);
    }
  }
  auto tv = a.t();

  auto d = clustering::detail::describeMatrix(tv);
  EXPECT_EQ(d.ptr, a.data());
  EXPECT_EQ(d.rows, 6u);
  EXPECT_EQ(d.cols, 4u);
  EXPECT_EQ(d.rowStride, static_cast<std::ptrdiff_t>(1));
  EXPECT_EQ(d.colStride, static_cast<std::ptrdiff_t>(6));
  EXPECT_FALSE(d.isContiguous);
}

TEST(MatrixDesc, ReadThroughDescriptorMatchesOriginalContiguous) {
  NDArray<float, 2> a({3, 4});
  for (std::size_t i = 0; i < 3; ++i) {
    for (std::size_t j = 0; j < 4; ++j) {
      a[i][j] = static_cast<float>(i * 4 + j) + 0.5f;
    }
  }

  auto d = clustering::detail::describeMatrix(a);
  for (std::size_t i = 0; i < d.rows; ++i) {
    for (std::size_t j = 0; j < d.cols; ++j) {
      EXPECT_FLOAT_EQ(readStrided(d, i, j), static_cast<float>(i * 4 + j) + 0.5f);
    }
  }
}

TEST(MatrixDesc, ReadThroughDescriptorMatchesOriginalTransposed) {
  NDArray<float, 2> a({3, 4});
  for (std::size_t i = 0; i < 3; ++i) {
    for (std::size_t j = 0; j < 4; ++j) {
      a[i][j] = static_cast<float>(i * 4 + j) + 0.25f;
    }
  }
  auto tv = a.t();

  auto d = clustering::detail::describeMatrix(tv);
  // Transposed view has shape (4, 3) over the same buffer.
  for (std::size_t i = 0; i < d.rows; ++i) {
    for (std::size_t j = 0; j < d.cols; ++j) {
      // tv(i, j) corresponds to a(j, i) in the original buffer.
      EXPECT_FLOAT_EQ(readStrided(d, i, j), static_cast<float>(j * 4 + i) + 0.25f);
    }
  }
}

TEST(MatrixDesc, DescribeBorrowedReadOnlyIsContiguous) {
  alignas(32) float buf[3 * 4];
  for (std::size_t k = 0; k < 12; ++k) {
    buf[k] = static_cast<float>(k);
  }
  const float *cptr = buf;
  auto view = NDArray<float, 2>::borrow(cptr, std::array<std::size_t, 2>{3, 4});

  auto d = clustering::detail::describeMatrix(view);
  EXPECT_EQ(d.ptr, cptr);
  EXPECT_EQ(d.rows, 3u);
  EXPECT_EQ(d.cols, 4u);
  EXPECT_TRUE(d.isContiguous);
  EXPECT_GE(d.alignment, 32u);
}

TEST(MatrixDesc, DescribeMatrixMutOnOwned) {
  NDArray<float, 2> a({2, 3});
  auto d = clustering::detail::describeMatrixMut(a);
  static_assert(std::is_same_v<decltype(d.ptr), float *>,
                "describeMatrixMut must return mutable-polarity descriptor");

  d.ptr[0 * d.rowStride + 2 * d.colStride] = 11.0f;
  d.ptr[1 * d.rowStride + 0 * d.colStride] = 22.0f;

  EXPECT_FLOAT_EQ(static_cast<float>(a[0][2]), 11.0f);
  EXPECT_FLOAT_EQ(static_cast<float>(a[1][0]), 22.0f);
}

TEST(MatrixDesc, DescribeMatrixMutOnMutableBorrow) {
  alignas(32) float buf[2 * 3]{};
  auto view = NDArray<float, 2>::borrow(buf, std::array<std::size_t, 2>{2, 3});
  EXPECT_TRUE(view.isMutable());

  auto d = clustering::detail::describeMatrixMut(view);
  d.ptr[1 * d.rowStride + 2 * d.colStride] = 42.0f;
  EXPECT_FLOAT_EQ(buf[5], 42.0f);
}

#ifndef NDEBUG
TEST(MatrixDesc, DescribeMatrixMutOnReadOnlyBorrowAsserts) {
  alignas(32) float buf[2 * 3]{};
  const float *cptr = buf;
  auto view = NDArray<float, 2>::borrow(cptr, std::array<std::size_t, 2>{2, 3});
  EXPECT_FALSE(view.isMutable());
  EXPECT_DEATH({ (void)clustering::detail::describeMatrixMut(view); }, "");
}
#endif

TEST(MatrixDesc, AlignmentGranularityReflectsPointer) {
  // A contiguous Owned NDArray is 32-aligned by construction, so describeMatrix must not
  // report a smaller granularity than 32 for the base pointer.
  NDArray<float, 2> a({4, 4});
  auto d = clustering::detail::describeMatrix(a);
  EXPECT_EQ(d.alignment % 32u, 0u);

  // An interior-column slice on a row-major source advances the base pointer by
  // sizeof(float) * begin_col. For begin_col == 1 on a 32-aligned base, that is a 4-byte
  // offset: the descriptor must report 4-byte alignment, not 32.
  auto sliced = a.slice({Range{0, 4, 1}, Range{1, 4, 1}});
  auto dslice = clustering::detail::describeMatrix(sliced);
  const auto addr = reinterpret_cast<std::uintptr_t>(dslice.ptr);
  EXPECT_EQ(addr % dslice.alignment, 0u);
  EXPECT_LE(dslice.alignment, 16u);
  EXPECT_GE(dslice.alignment, 4u);
}
