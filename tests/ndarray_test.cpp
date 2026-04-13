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

TEST(OperatorCall, Rank2MatchesSubscript) {
  NDArray<float, 2> arr({5, 7});
  for (std::size_t i = 0; i < 5; ++i) {
    for (std::size_t j = 0; j < 7; ++j) {
      arr[i][j] = static_cast<float>(i * 31 + j * 7 + 1);
    }
  }
  for (std::size_t i = 0; i < 5; ++i) {
    for (std::size_t j = 0; j < 7; ++j) {
      EXPECT_FLOAT_EQ(arr(i, j), static_cast<float>(arr[i][j]));
    }
  }
}

TEST(OperatorCall, Rank1MatchesSubscript) {
  NDArray<float, 1> arr({11});
  for (std::size_t i = 0; i < 11; ++i) {
    arr[i] = static_cast<float>(i * 13 + 2);
  }
  for (std::size_t i = 0; i < 11; ++i) {
    EXPECT_FLOAT_EQ(arr(i), static_cast<float>(arr[i]));
  }
}

TEST(OperatorCall, Rank3MatchesSubscript) {
  NDArray<float, 3> arr({2, 3, 4});
  for (std::size_t i = 0; i < 2; ++i) {
    for (std::size_t j = 0; j < 3; ++j) {
      for (std::size_t k = 0; k < 4; ++k) {
        arr[i][j][k] = static_cast<float>(i * 100 + j * 10 + k);
      }
    }
  }
  for (std::size_t i = 0; i < 2; ++i) {
    for (std::size_t j = 0; j < 3; ++j) {
      for (std::size_t k = 0; k < 4; ++k) {
        EXPECT_FLOAT_EQ(arr(i, j, k), static_cast<float>(arr[i][j][k]));
      }
    }
  }
}

TEST(IsContiguous, ContigInstantiationIsContiguous) {
  NDArray<float, 2> arr({16, 8});
  EXPECT_TRUE(arr.isContiguous());
}

TEST(IsContiguous, EmptyIsTriviallyContiguous) {
  NDArray<float, 2> arr({0, 64});
  EXPECT_TRUE(arr.isContiguous());
}

// Type-level contract: operator[] exists on Contig, is ill-formed on MaybeStrided.
// Misuse is a build error, not a runtime trap.
template <class A>
concept has_subscript_chain = requires(A &arr) { arr[std::size_t{0}]; };

static_assert(has_subscript_chain<NDArray<float, 2, Layout::Contig>>,
              "operator[] must be available on Contig arrays");
static_assert(!has_subscript_chain<NDArray<float, 2, Layout::MaybeStrided>>,
              "operator[] must be ill-formed on MaybeStrided arrays");
static_assert(has_subscript_chain<NDArray<double, 3, Layout::Contig>>,
              "operator[] must be available on Contig arrays of any rank");
static_assert(!has_subscript_chain<NDArray<double, 3, Layout::MaybeStrided>>,
              "operator[] must be ill-formed on MaybeStrided arrays of any rank");
