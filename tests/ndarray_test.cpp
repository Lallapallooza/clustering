#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include "clustering/ndarray.h"

// is_always_equal must propagate through std::allocator_traits so std::vector picks the
// pointer-stealing move path. Compile failure here is the contradiction signal.
static_assert(
    std::allocator_traits<clustering::detail::AlignedAllocator<float, 32>>::is_always_equal::value,
    "AlignedAllocator<float, 32>::is_always_equal must be true_type");

TEST(NDArrayAllocator, AllocateZeroReturnsNullptr) {
  NDArray<float, 2> arr({0, 64});
  EXPECT_EQ(arr.data(), nullptr);
  EXPECT_EQ(arr.dim(0), 0u);
  EXPECT_EQ(arr.dim(1), 64u);
}

TEST(NDArrayAllocator, NonZeroAllocIs32Aligned) {
  NDArray<float, 2> arr({7, 13});
  EXPECT_NE(arr.data(), nullptr);
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(arr.data()) % 32, 0u);
  EXPECT_TRUE(arr.isAligned<32>());
}

TEST(NDArrayAllocator, ReservePreserves32Alignment) {
  std::vector<float, clustering::detail::AlignedAllocator<float, 32>> vec;
  vec.reserve(100);
  ASSERT_NE(vec.data(), nullptr);
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(vec.data()) % 32, 0u);

  vec.reserve(10000);
  ASSERT_NE(vec.data(), nullptr);
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(vec.data()) % 32, 0u);
}

TEST(NDArrayStorage, MoveTransfersPointerAndPreservesAlignment) {
  NDArray<float, 2> src({128, 16});
  src[0][0] = 42.0f;
  src[127][15] = 7.0f;
  float *ptr = src.data();
  ASSERT_NE(ptr, nullptr);

  NDArray<float, 2> dst = std::move(src);
  EXPECT_EQ(dst.data(), ptr);
  EXPECT_TRUE(dst.isAligned<32>());
  EXPECT_EQ(dst.dim(0), 128u);
  EXPECT_EQ(dst.dim(1), 16u);
  EXPECT_FLOAT_EQ(static_cast<float>(dst[0][0]), 42.0f);
  EXPECT_FLOAT_EQ(static_cast<float>(dst[127][15]), 7.0f);
}

TEST(NDArrayStorage, CopyAssignmentProduces32AlignedCopy) {
  NDArray<float, 2> src({64, 8});
  src[3][5] = 2.5f;

  NDArray<float, 2> dst({2, 2});
  dst = src;

  EXPECT_TRUE(dst.isAligned<32>());
  EXPECT_NE(dst.data(), src.data());
  EXPECT_EQ(dst.dim(0), 64u);
  EXPECT_EQ(dst.dim(1), 8u);
  EXPECT_FLOAT_EQ(static_cast<float>(dst[3][5]), 2.5f);
}

TEST(NDArrayStorage, Rank1Construction) {
  NDArray<float, 1> arr({4});
  for (std::size_t i = 0; i < 4; ++i) {
    arr[i] = static_cast<float>(i);
  }
  EXPECT_EQ(arr.dim(0), 4u);
  EXPECT_FLOAT_EQ(static_cast<float>(arr[2]), 2.0f);
}

TEST(NDArrayStorage, DoubleElementType) {
  NDArray<double, 2> arr({8, 4});
  arr[1][2] = 3.25;
  EXPECT_TRUE(arr.isAligned<32>());
  EXPECT_EQ(arr.dim(0), 8u);
  EXPECT_EQ(arr.dim(1), 4u);
  EXPECT_DOUBLE_EQ(static_cast<double>(arr[1][2]), 3.25);
}
