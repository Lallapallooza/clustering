#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include "clustering/ndarray.h"

using clustering::all;
using clustering::Layout;
using clustering::NDArray;
using clustering::NDArrayStorage;
using clustering::Range;
using clustering::sameStorage;

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

TEST(Transpose, Rank2SwapsShapeAndStrides) {
  NDArray<float, 2> arr({3, 5});
  auto view = arr.t();
  EXPECT_EQ(view.dim(0), 5u);
  EXPECT_EQ(view.dim(1), 3u);
  EXPECT_EQ(view.strideAt(0), 1);
  EXPECT_EQ(view.strideAt(1), 5);
}

TEST(Transpose, ElementAccessMatchesSource) {
  NDArray<float, 2> arr({3, 4});
  for (std::size_t i = 0; i < 3; ++i) {
    for (std::size_t j = 0; j < 4; ++j) {
      arr[i][j] = static_cast<float>(i * 10 + j);
    }
  }
  auto view = arr.t();
  for (std::size_t i = 0; i < 3; ++i) {
    for (std::size_t j = 0; j < 4; ++j) {
      EXPECT_FLOAT_EQ(view(j, i), static_cast<float>(arr[i][j]));
    }
  }
}

TEST(Transpose, SharesStorageWithSource) {
  NDArray<float, 2> arr({3, 5});
  auto view = arr.t();
  EXPECT_TRUE(sameStorage(arr, view));
}

TEST(Row, Rank2RowIsContigRank1) {
  NDArray<float, 2> arr({4, 6});
  for (std::size_t i = 0; i < 4; ++i) {
    for (std::size_t j = 0; j < 6; ++j) {
      arr[i][j] = static_cast<float>(i * 100 + j);
    }
  }
  auto view = arr.row(2);
  EXPECT_EQ(view.dim(0), 6u);
  EXPECT_TRUE(view.isContiguous());
  for (std::size_t j = 0; j < 6; ++j) {
    EXPECT_FLOAT_EQ(view[j], static_cast<float>(arr[2][j]));
  }
}

// Type-level: row(i) preserves the Contig layout tag.
static_assert(std::is_same_v<decltype(std::declval<NDArray<float, 2> &>().row(0)),
                             NDArray<float, 1, Layout::Contig>>,
              "row() on a Contig rank-2 must yield a Contig rank-1 view");

TEST(Col, Rank2ColumnIsStridedRank1) {
  NDArray<float, 2> arr({4, 6});
  for (std::size_t i = 0; i < 4; ++i) {
    for (std::size_t j = 0; j < 6; ++j) {
      arr[i][j] = static_cast<float>(i * 100 + j);
    }
  }
  auto view = arr.col(3);
  EXPECT_EQ(view.dim(0), 4u);
  EXPECT_EQ(view.strideAt(0), 6);
  for (std::size_t i = 0; i < 4; ++i) {
    EXPECT_FLOAT_EQ(view(i), static_cast<float>(arr[i][3]));
  }
}

// col() always returns MaybeStrided: operator[] must be ill-formed on the result.
static_assert(!has_subscript_chain<decltype(std::declval<NDArray<float, 2> &>().col(0))>,
              "col() must yield a MaybeStrided view (no subscript chain)");

TEST(Slice, AxisZeroReducesLeadingDim) {
  NDArray<float, 2> arr({5, 4});
  for (std::size_t i = 0; i < 5; ++i) {
    for (std::size_t j = 0; j < 4; ++j) {
      arr[i][j] = static_cast<float>(i * 10 + j);
    }
  }
  auto view = arr.slice(0, 1, 4);
  EXPECT_EQ(view.dim(0), 3u);
  EXPECT_EQ(view.dim(1), 4u);
  EXPECT_EQ(view.strideAt(0), arr.strideAt(0));
  EXPECT_EQ(view.strideAt(1), arr.strideAt(1));
  for (std::size_t i = 0; i < 3; ++i) {
    for (std::size_t j = 0; j < 4; ++j) {
      EXPECT_FLOAT_EQ(view(i, j), static_cast<float>(arr[i + 1][j]));
    }
  }
}

TEST(Slice, RangeArrayWithAllSentinel) {
  NDArray<float, 2> arr({5, 4});
  for (std::size_t i = 0; i < 5; ++i) {
    for (std::size_t j = 0; j < 4; ++j) {
      arr[i][j] = static_cast<float>(i * 10 + j);
    }
  }
  auto view = arr.slice(std::array<Range, 2>{Range{1, 4}, all()});
  EXPECT_EQ(view.dim(0), 3u);
  EXPECT_EQ(view.dim(1), 4u);
  for (std::size_t i = 0; i < 3; ++i) {
    for (std::size_t j = 0; j < 4; ++j) {
      EXPECT_FLOAT_EQ(view(i, j), static_cast<float>(arr[i + 1][j]));
    }
  }
}

TEST(Slice, RangeArrayWithStep) {
  NDArray<float, 1> arr({10});
  for (std::size_t i = 0; i < 10; ++i) {
    arr[i] = static_cast<float>(i);
  }
  auto view = arr.slice(std::array<Range, 1>{Range{0, 10, 3}});
  EXPECT_EQ(view.dim(0), 4u);
  EXPECT_EQ(view.strideAt(0), 3);
  EXPECT_FLOAT_EQ(view(0), 0.0f);
  EXPECT_FLOAT_EQ(view(1), 3.0f);
  EXPECT_FLOAT_EQ(view(2), 6.0f);
  EXPECT_FLOAT_EQ(view(3), 9.0f);
}

TEST(Permute, Rank2SwapMatchesTranspose) {
  NDArray<float, 2> arr({3, 5});
  for (std::size_t i = 0; i < 3; ++i) {
    for (std::size_t j = 0; j < 5; ++j) {
      arr[i][j] = static_cast<float>(i * 100 + j);
    }
  }
  auto via_permute = arr.permute(std::array<std::size_t, 2>{1, 0});
  auto via_t = arr.t();
  EXPECT_EQ(via_permute.dim(0), via_t.dim(0));
  EXPECT_EQ(via_permute.dim(1), via_t.dim(1));
  EXPECT_EQ(via_permute.strideAt(0), via_t.strideAt(0));
  EXPECT_EQ(via_permute.strideAt(1), via_t.strideAt(1));
  for (std::size_t i = 0; i < 5; ++i) {
    for (std::size_t j = 0; j < 3; ++j) {
      EXPECT_FLOAT_EQ(via_permute(i, j), via_t(i, j));
    }
  }
}

TEST(View, Rank2To1SharesStorage) {
  NDArray<float, 2> arr({100, 10});
  for (std::size_t i = 0; i < 100; ++i) {
    for (std::size_t j = 0; j < 10; ++j) {
      arr[i][j] = static_cast<float>(i * 10 + j);
    }
  }
  auto flat = arr.view(std::array<std::size_t, 1>{1000});
  EXPECT_EQ(flat.dim(0), 1000u);
  EXPECT_EQ(flat.strideAt(0), 1);
  EXPECT_TRUE(sameStorage(arr, flat));
  for (std::size_t i = 0; i < 100; ++i) {
    for (std::size_t j = 0; j < 10; ++j) {
      EXPECT_FLOAT_EQ(flat[i * 10 + j], static_cast<float>(i * 10 + j));
    }
  }
}

// view<M> on a Contig source returns a Contig view, never MaybeStrided.
static_assert(
    std::is_same_v<decltype(std::declval<NDArray<float, 2> &>().view(std::array<std::size_t, 1>{})),
                   NDArray<float, 1, Layout::Contig>>,
    "view<M>() must yield a Contig result");

TEST(Reshape, ContigNoAlloc) {
  NDArray<float, 2> arr({8, 12});
  for (std::size_t i = 0; i < 8; ++i) {
    for (std::size_t j = 0; j < 12; ++j) {
      arr[i][j] = static_cast<float>(i * 12 + j);
    }
  }
  auto view = arr.reshape(std::array<std::size_t, 3>{4, 2, 12});
  EXPECT_EQ(view.dim(0), 4u);
  EXPECT_EQ(view.dim(1), 2u);
  EXPECT_EQ(view.dim(2), 12u);
  EXPECT_TRUE(sameStorage(arr, view));
  for (std::size_t flat = 0; flat < 96; ++flat) {
    EXPECT_FLOAT_EQ(view.flatIndex(flat), static_cast<float>(flat));
  }
}

TEST(Reshape, NonContigAllocates) {
  NDArray<float, 2> arr({4, 5});
  for (std::size_t i = 0; i < 4; ++i) {
    for (std::size_t j = 0; j < 5; ++j) {
      arr[i][j] = static_cast<float>(i * 5 + j);
    }
  }
  auto transposed = arr.t();
  ASSERT_FALSE(transposed.isContiguous());
  auto flat = transposed.reshape(std::array<std::size_t, 1>{20});
  EXPECT_FALSE(sameStorage(arr, flat));
  EXPECT_EQ(flat.dim(0), 20u);
  // Expected dense row-major walk of the transposed (5x4) view: column-major of the source.
  for (std::size_t j = 0; j < 5; ++j) {
    for (std::size_t i = 0; i < 4; ++i) {
      EXPECT_FLOAT_EQ(flat[j * 4 + i], static_cast<float>(i * 5 + j));
    }
  }
}

TEST(Contiguous, AlreadyContigSharesStorage) {
  NDArray<float, 2> arr({6, 7});
  for (std::size_t i = 0; i < 6; ++i) {
    for (std::size_t j = 0; j < 7; ++j) {
      arr[i][j] = static_cast<float>(i * 7 + j);
    }
  }
  auto same = arr.contiguous();
  EXPECT_TRUE(sameStorage(arr, same));
  EXPECT_EQ(same.dim(0), arr.dim(0));
  EXPECT_EQ(same.dim(1), arr.dim(1));
  EXPECT_EQ(same.strideAt(0), 7);
  EXPECT_EQ(same.strideAt(1), 1);
}

TEST(Contiguous, StridedReallocatesAndCopies) {
  NDArray<float, 2> arr({3, 5});
  for (std::size_t i = 0; i < 3; ++i) {
    for (std::size_t j = 0; j < 5; ++j) {
      arr[i][j] = static_cast<float>(i * 100 + j);
    }
  }
  auto transposed = arr.t();
  auto dense = transposed.contiguous();
  EXPECT_FALSE(sameStorage(arr, dense));
  EXPECT_EQ(dense.dim(0), 5u);
  EXPECT_EQ(dense.dim(1), 3u);
  EXPECT_TRUE(dense.isContiguous());
  for (std::size_t i = 0; i < 5; ++i) {
    for (std::size_t j = 0; j < 3; ++j) {
      EXPECT_FLOAT_EQ(dense(i, j), transposed(i, j));
      EXPECT_FLOAT_EQ(dense(i, j), static_cast<float>(j * 100 + i));
    }
  }
}

TEST(Clone, AlwaysAllocates) {
  NDArray<float, 2> arr({4, 3});
  for (std::size_t i = 0; i < 4; ++i) {
    for (std::size_t j = 0; j < 3; ++j) {
      arr[i][j] = static_cast<float>(i * 10 + j);
    }
  }
  auto copy = arr.clone();
  EXPECT_FALSE(sameStorage(arr, copy));
  EXPECT_EQ(copy.dim(0), 4u);
  EXPECT_EQ(copy.dim(1), 3u);
  EXPECT_TRUE(copy.isAligned<32>());
  for (std::size_t i = 0; i < 4; ++i) {
    for (std::size_t j = 0; j < 3; ++j) {
      EXPECT_FLOAT_EQ(copy[i][j], arr[i][j]);
    }
  }
  // Mutating the clone must not touch the source: confirms independent storage.
  copy[0][0] = -1.0f;
  EXPECT_FLOAT_EQ(static_cast<float>(arr[0][0]), 0.0f);
}

TEST(Clone, StridedSourceProducesDenseOwned) {
  NDArray<float, 2> arr({3, 5});
  for (std::size_t i = 0; i < 3; ++i) {
    for (std::size_t j = 0; j < 5; ++j) {
      arr[i][j] = static_cast<float>(i * 5 + j);
    }
  }
  auto transposed = arr.t();
  auto copy = transposed.clone();
  EXPECT_FALSE(sameStorage(arr, copy));
  EXPECT_TRUE(copy.isContiguous());
  EXPECT_EQ(copy.dim(0), 5u);
  EXPECT_EQ(copy.dim(1), 3u);
  for (std::size_t i = 0; i < 5; ++i) {
    for (std::size_t j = 0; j < 3; ++j) {
      EXPECT_FLOAT_EQ(copy(i, j), transposed(i, j));
    }
  }
}

// view<M> on a non-contiguous MaybeStrided source must assert in debug. Transpose yields a
// MaybeStrided view whose strides no longer match the contiguous layout, so .view<1>({20})
// violates the precondition and must trap.
TEST(ViewDeathTest, NonContigSourceAborts) {
  NDArray<float, 2> arr({4, 5});
  auto transposed = arr.t();
  ASSERT_FALSE(transposed.isContiguous());
  EXPECT_DEBUG_DEATH(
      { (void)transposed.view(std::array<std::size_t, 1>{20}); },
      "view<M> requires a contiguous source");
}

TEST(Borrow, ContigMutablePointerBuildsWritableView) {
  alignas(32) float raw[12] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  auto view = NDArray<float, 2>::borrow(raw, std::array<std::size_t, 2>{3, 4});
  EXPECT_EQ(view.dim(0), 3u);
  EXPECT_EQ(view.dim(1), 4u);
  EXPECT_EQ(view.strideAt(0), 4);
  EXPECT_EQ(view.strideAt(1), 1);
  EXPECT_EQ(view.data(), raw);
  EXPECT_TRUE(view.isContiguous());
  EXPECT_FLOAT_EQ(view(2, 3), 11.0f);
  view(0, 0) = -7.0f;
  EXPECT_FLOAT_EQ(raw[0], -7.0f);
}

TEST(Borrow, ContigConstPointerProducesReadOnlyView) {
  alignas(32) float raw[6] = {1, 2, 3, 4, 5, 6};
  const float *cptr = raw;
  const auto view = NDArray<float, 2>::borrow(cptr, std::array<std::size_t, 2>{2, 3});
  EXPECT_EQ(view.dim(0), 2u);
  EXPECT_EQ(view.dim(1), 3u);
  // Bind through std::as_const to force the const operator() overload; non-const operator()
  // asserts on read-only borrows (that behavior is covered by the DeathTests below).
  EXPECT_FLOAT_EQ(std::as_const(view)(1, 2), 6.0f);
  EXPECT_EQ(view.data(), raw);
}

TEST(BorrowDeathTest, WriteViaOperatorCallOnConstBorrowAborts) {
  alignas(32) float raw[4] = {0, 0, 0, 0};
  const float *cptr = raw;
  auto view = NDArray<float, 1>::borrow(cptr, std::array<std::size_t, 1>{4});
  EXPECT_DEBUG_DEATH({ view(0) = 1.0f; }, "write to read-only borrow");
}

TEST(BorrowDeathTest, WriteViaAccessorOnConstBorrowAborts) {
  alignas(32) float raw[4] = {0, 0, 0, 0};
  const float *cptr = raw;
  auto view = NDArray<float, 1>::borrow(cptr, std::array<std::size_t, 1>{4});
  EXPECT_DEBUG_DEATH({ view[0] = 1.0f; }, "write to read-only borrow");
}

TEST(Borrow, StridedMutableBuildsStridedView) {
  alignas(32) float raw[8] = {0, 1, 2, 3, 4, 5, 6, 7};
  // Interpret raw as a 2x2 matrix whose row stride is 4 and column stride is 2 (every other).
  auto view = NDArray<float, 2, Layout::MaybeStrided>::borrow(raw, std::array<std::size_t, 2>{2, 2},
                                                              std::array<std::ptrdiff_t, 2>{4, 2});
  EXPECT_EQ(view.dim(0), 2u);
  EXPECT_EQ(view.dim(1), 2u);
  EXPECT_EQ(view.strideAt(0), 4);
  EXPECT_EQ(view.strideAt(1), 2);
  EXPECT_FLOAT_EQ(view(0, 0), 0.0f);
  EXPECT_FLOAT_EQ(view(0, 1), 2.0f);
  EXPECT_FLOAT_EQ(view(1, 0), 4.0f);
  EXPECT_FLOAT_EQ(view(1, 1), 6.0f);
  view(1, 1) = -9.0f;
  EXPECT_FLOAT_EQ(raw[6], -9.0f);
}

TEST(BorrowDeathTest, StridedConstBorrowTrapsOnWrite) {
  alignas(32) float raw[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  const float *cptr = raw;
  auto view = NDArray<float, 2, Layout::MaybeStrided>::borrow(
      cptr, std::array<std::size_t, 2>{2, 2}, std::array<std::ptrdiff_t, 2>{4, 2});
  EXPECT_DEBUG_DEATH({ view(0, 0) = 1.0f; }, "write to read-only borrow");
}

TEST(Borrow, OneDFromMutablePointer) {
  alignas(32) float raw[5] = {10, 20, 30, 40, 50};
  auto view = NDArray<float, 1>::borrow1D(raw, 5);
  EXPECT_EQ(view.dim(0), 5u);
  EXPECT_FLOAT_EQ(view[3], 40.0f);
  view[0] = -1.0f;
  EXPECT_FLOAT_EQ(raw[0], -1.0f);
}

TEST(Borrow, OneDFromConstPointerReadOnly) {
  alignas(32) float raw[3] = {1.5f, 2.5f, 3.5f};
  const float *cptr = raw;
  const auto view = NDArray<float, 1>::borrow1D(cptr, 3);
  EXPECT_EQ(view.dim(0), 3u);
  EXPECT_FLOAT_EQ(std::as_const(view)(1), 2.5f);
}

TEST(Borrow, BytesDividesByElementSize) {
  alignas(32) float raw[12] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  // Treat raw as 3x4 with byte-stride (16 bytes row, 4 bytes column) for float -> element 4, 1.
  auto view = NDArray<float, 2, Layout::MaybeStrided>::borrowBytes(
      raw, std::array<std::size_t, 2>{3, 4}, std::array<std::ptrdiff_t, 2>{16, 4},
      /*isMutable=*/true);
  EXPECT_EQ(view.strideAt(0), 4);
  EXPECT_EQ(view.strideAt(1), 1);
  EXPECT_FLOAT_EQ(view(2, 3), 11.0f);
  view(0, 0) = 99.0f;
  EXPECT_FLOAT_EQ(raw[0], 99.0f);
}

TEST(Borrow, BytesHonorsImmutableFlag) {
  alignas(32) float raw[4] = {0, 0, 0, 0};
  auto view = NDArray<float, 1, Layout::MaybeStrided>::borrowBytes(
      raw, std::array<std::size_t, 1>{4}, std::array<std::ptrdiff_t, 1>{4},
      /*isMutable=*/false);
  EXPECT_DEBUG_DEATH({ view(0) = 1.0f; }, "write to read-only borrow");
}

TEST(BorrowBytesDeathTest, NonDivisibleByteStrideAborts) {
  alignas(32) float raw[4] = {0, 0, 0, 0};
  // 6 is not divisible by sizeof(float) == 4 -> must trap.
  auto make_bad = [&]() {
    using Arr = NDArray<float, 1, Layout::MaybeStrided>;
    (void)Arr::borrowBytes(raw, std::array<std::size_t, 1>{2}, std::array<std::ptrdiff_t, 1>{6},
                           /*isMutable=*/true);
  };
  EXPECT_DEBUG_DEATH(make_bad(), "byte strides divisible by sizeof");
}

TEST(Borrow, FromSpanMutable) {
  alignas(32) float raw[4] = {1, 2, 3, 4};
  std::span<float> s(raw, 4);
  auto view = NDArray<float, 1>::fromSpan(s);
  EXPECT_EQ(view.dim(0), 4u);
  EXPECT_FLOAT_EQ(view[2], 3.0f);
  view[2] = -8.0f;
  EXPECT_FLOAT_EQ(raw[2], -8.0f);
}

TEST(Borrow, FromSpanConstReadOnly) {
  alignas(32) float raw[4] = {1, 2, 3, 4};
  std::span<const float> s(raw, 4);
  const auto view = NDArray<float, 1>::fromSpan(s);
  EXPECT_EQ(view.dim(0), 4u);
  EXPECT_FLOAT_EQ(std::as_const(view)(3), 4.0f);
}

TEST(AlignedData, OwnedArrayIs32Aligned) {
  NDArray<float, 2> arr({8, 16});
  auto *ptr = arr.alignedData<32>();
  EXPECT_EQ(ptr, arr.data());
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(ptr) % 32, 0u);
}

TEST(AlignedData, ConstOverload) {
  NDArray<float, 2> arr({4, 8});
  arr[0][0] = 42.0f;
  const auto &cref = arr;
  const auto *ptr = cref.alignedData<32>();
  EXPECT_EQ(ptr, arr.data());
  EXPECT_FLOAT_EQ(ptr[0], 42.0f);
}

TEST(AlignedDataDeathTest, UnalignedBorrowAborts) {
  alignas(32) float raw[32] = {};
  // Offset by one float (4 bytes) to force a non-32-byte-aligned pointer.
  float *misaligned = raw + 1;
  auto view = NDArray<float, 1>::borrow(misaligned, std::array<std::size_t, 1>{16});
  EXPECT_DEBUG_DEATH(
      { (void)view.alignedData<32>(); }, "alignedData<A>\\(\\) requires A-byte aligned data");
}

// The signatures on the borrow factories are constrained to their natural layout.
// Type-level checks: contiguous borrow is only on Contig, strided borrow is only on
// MaybeStrided, and the 1D convenience factories gate on N==1.
template <class A>
concept has_contig_borrow_shape = requires {
  { A::borrow(static_cast<typename A::value_type *>(nullptr), std::array<std::size_t, 1>{}) };
};

static_assert(std::is_same_v<decltype(NDArray<float, 1>::borrow(static_cast<float *>(nullptr),
                                                                std::array<std::size_t, 1>{1})),
                             NDArray<float, 1, Layout::Contig>>,
              "borrow(T*, shape) on Contig must yield Contig");

static_assert(std::is_same_v<decltype(NDArray<float, 2, Layout::MaybeStrided>::borrow(
                                 static_cast<float *>(nullptr), std::array<std::size_t, 2>{1, 1},
                                 std::array<std::ptrdiff_t, 2>{1, 1})),
                             NDArray<float, 2, Layout::MaybeStrided>>,
              "strided borrow must yield MaybeStrided");

static_assert(std::is_same_v<decltype(NDArray<float, 1>::borrow1D(static_cast<float *>(nullptr),
                                                                  std::size_t{4})),
                             NDArray<float, 1, Layout::Contig>>,
              "borrow1D must yield Contig rank-1");

TEST(Borrow, MovePreservesBorrowedPointer) {
  alignas(32) float raw[4] = {1, 2, 3, 4};
  auto view = NDArray<float, 1>::borrow(raw, std::array<std::size_t, 1>{4});
  ASSERT_EQ(view.data(), raw);
  auto moved = std::move(view);
  EXPECT_EQ(moved.data(), raw);
  EXPECT_EQ(moved.dim(0), 4u);
  EXPECT_FLOAT_EQ(moved(2), 3.0f);
}

TEST(Borrow, CopyPreservesBorrowedPointer) {
  alignas(32) float raw[4] = {1, 2, 3, 4};
  auto view = NDArray<float, 1>::borrow(raw, std::array<std::size_t, 1>{4});
  ASSERT_EQ(view.data(), raw);
  auto copied = view;
  EXPECT_EQ(copied.data(), raw);
  EXPECT_EQ(view.data(), raw);
  EXPECT_EQ(copied.dim(0), 4u);
  EXPECT_FLOAT_EQ(copied(1), 2.0f);
}

TEST(Borrow, MoveAssignPreservesBorrowedPointer) {
  alignas(32) float raw[4] = {5, 6, 7, 8};
  auto view = NDArray<float, 1>::borrow(raw, std::array<std::size_t, 1>{4});
  ASSERT_EQ(view.data(), raw);
  NDArray<float, 1> sink({1});
  sink = std::move(view);
  EXPECT_EQ(sink.data(), raw);
  EXPECT_EQ(sink.dim(0), 4u);
  EXPECT_FLOAT_EQ(sink(3), 8.0f);
}

TEST(Borrow, CopyAssignPreservesBorrowedPointer) {
  alignas(32) float raw[4] = {5, 6, 7, 8};
  auto view = NDArray<float, 1>::borrow(raw, std::array<std::size_t, 1>{4});
  ASSERT_EQ(view.data(), raw);
  NDArray<float, 1> sink({1});
  sink = view;
  EXPECT_EQ(sink.data(), raw);
  EXPECT_EQ(view.data(), raw);
  EXPECT_EQ(sink.dim(0), 4u);
  EXPECT_FLOAT_EQ(sink(0), 5.0f);
}

TEST(BorrowDeathTest, WriteViaFlatIndexOnConstBorrowAborts) {
  alignas(32) float raw[4] = {0, 0, 0, 0};
  const float *cptr = raw;
  auto view = NDArray<float, 1>::borrow(cptr, std::array<std::size_t, 1>{4});
  EXPECT_DEBUG_DEATH({ view.flatIndex(0) = 1.0f; }, "write to read-only borrow");
}

TEST(BorrowDeathTest, WriteViaDataPointerOnConstBorrowAborts) {
  alignas(32) float raw[4] = {0, 0, 0, 0};
  const float *cptr = raw;
  auto view = NDArray<float, 1>::borrow(cptr, std::array<std::size_t, 1>{4});
  EXPECT_DEBUG_DEATH({ (void)view.data(); }, "write to read-only borrow");
}

TEST(DebugDump, BorrowedViewReportsViewableContents) {
  alignas(32) float raw[6] = {10, 20, 30, 40, 50, 60};
  auto view = NDArray<float, 2>::borrow(raw, std::array<std::size_t, 2>{2, 3});
  std::string dump = view.debugDump();
  // Every source element must appear in the dump; the old m_vec-only walk produced "data: []".
  EXPECT_NE(dump.find("10"), std::string::npos);
  EXPECT_NE(dump.find("60"), std::string::npos);
  EXPECT_NE(dump.find("size: 6"), std::string::npos);
}

TEST(DebugDump, TransposeReadsThroughStrides) {
  NDArray<float, 2> arr({2, 3});
  for (std::size_t i = 0; i < 2; ++i) {
    for (std::size_t j = 0; j < 3; ++j) {
      arr[i][j] = static_cast<float>(i * 3 + j + 1);
    }
  }
  auto tv = arr.t();
  std::string dump = tv.debugDump();
  EXPECT_NE(dump.find("size: 6"), std::string::npos);
  // Row-major walk of the 3x2 transposed view: a(0,0), a(1,0), a(0,1), a(1,1), a(0,2), a(1,2).
  EXPECT_NE(dump.find("1, 4, 2, 5, 3, 6"), std::string::npos);
}

// operator==/operator!= are deleted: silent element-wise, storage-identity, or deep-value
// semantics are all plausible and surprising. The compile-time check enforces that a future
// drive-by addition cannot restore one of those semantics unnoticed.
template <class A>
concept has_equality_compare = requires(const A &a, const A &b) {
  { a == b } -> std::same_as<bool>;
};
template <class A>
concept has_inequality_compare = requires(const A &a, const A &b) {
  { a != b } -> std::same_as<bool>;
};

static_assert(!has_equality_compare<NDArray<float, 2>>, "NDArray::operator== must remain deleted");
static_assert(!has_inequality_compare<NDArray<float, 2>>,
              "NDArray::operator!= must remain deleted");
static_assert(!has_equality_compare<NDArray<double, 3, Layout::MaybeStrided>>,
              "NDArray::operator== must remain deleted for MaybeStrided too");

TEST(ViewOffset, TransposeThenRowStillReadsSource) {
  // Composing view verbs must preserve correctness end-to-end: row() on a transposed source
  // must read the same elements the caller would have reached via tv(i, j) directly.
  NDArray<float, 2> arr({3, 4});
  for (std::size_t i = 0; i < 3; ++i) {
    for (std::size_t j = 0; j < 4; ++j) {
      arr[i][j] = static_cast<float>(i * 10 + j);
    }
  }
  auto tv = arr.t();
  auto r = tv.row(2);
  EXPECT_EQ(r.dim(0), 3u);
  for (std::size_t i = 0; i < 3; ++i) {
    EXPECT_FLOAT_EQ(r(i), static_cast<float>(i * 10 + 2));
  }
}

TEST(ViewOffset, TransposeThenSliceStillReadsSource) {
  NDArray<float, 2> arr({3, 4});
  for (std::size_t i = 0; i < 3; ++i) {
    for (std::size_t j = 0; j < 4; ++j) {
      arr[i][j] = static_cast<float>(i * 10 + j);
    }
  }
  auto tv = arr.t();
  auto s = tv.slice(0, 1, 3);
  EXPECT_EQ(s.dim(0), 2u);
  EXPECT_EQ(s.dim(1), 3u);
  for (std::size_t i = 0; i < 2; ++i) {
    for (std::size_t j = 0; j < 3; ++j) {
      EXPECT_FLOAT_EQ(s(i, j), tv(i + 1, j));
    }
  }
}

TEST(ViewOffset, TransposeThenColStillReadsSource) {
  NDArray<float, 2> arr({3, 4});
  for (std::size_t i = 0; i < 3; ++i) {
    for (std::size_t j = 0; j < 4; ++j) {
      arr[i][j] = static_cast<float>(i * 10 + j);
    }
  }
  auto tv = arr.t();
  auto c = tv.col(1);
  EXPECT_EQ(c.dim(0), 4u);
  for (std::size_t j = 0; j < 4; ++j) {
    EXPECT_FLOAT_EQ(c(j), tv(j, 1));
  }
}

TEST(SameStorage, RowAndColShareBase) {
  NDArray<float, 2> arr({4, 5});
  auto r0 = arr.row(0);
  auto r1 = arr.row(1);
  auto c2 = arr.col(2);
  EXPECT_TRUE(sameStorage(arr, r0));
  EXPECT_TRUE(sameStorage(arr, r1));
  EXPECT_TRUE(sameStorage(arr, c2));
  EXPECT_TRUE(sameStorage(r0, r1));
  EXPECT_TRUE(sameStorage(r0, c2));
}

TEST(SameStorage, SliceShareBase) {
  NDArray<float, 2> arr({6, 4});
  auto s0 = arr.slice(0, 1, 5);
  auto s1 = arr.slice(0, 3, 6);
  EXPECT_TRUE(sameStorage(arr, s0));
  EXPECT_TRUE(sameStorage(arr, s1));
  EXPECT_TRUE(sameStorage(s0, s1));
}

TEST(SameStorage, CloneBreaksBase) {
  NDArray<float, 2> arr({3, 3});
  auto copy = arr.clone();
  auto r = arr.row(0);
  EXPECT_FALSE(sameStorage(arr, copy));
  EXPECT_FALSE(sameStorage(copy, r));
}
