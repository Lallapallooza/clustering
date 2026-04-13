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
  auto view =
      arr.slice(std::array<clustering::Range, 2>{clustering::Range{1, 4}, clustering::all()});
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
  auto view = arr.slice(std::array<clustering::Range, 1>{clustering::Range{0, 10, 3}});
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
