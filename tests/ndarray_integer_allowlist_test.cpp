#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <type_traits>
#include <utility>
#include <vector>

#include "clustering/ndarray.h"

using clustering::NDArray;

// Positive pins: integer widths (signed, unsigned, multiple sizes) instantiate. Attempting to
// instantiate any of these types with a narrower allowlist would produce a compile error here
// whose message points directly at the regression.
namespace {

[[maybe_unused]] void integerAllowlistInstantiationProbe() {
  const NDArray<std::int32_t, 1> a({std::size_t{1}});
  const NDArray<std::uint8_t, 2> b({std::size_t{1}, std::size_t{1}});
  const NDArray<std::int64_t, 1> c({std::size_t{1}});
  const NDArray<std::uint32_t, 1> d({std::size_t{1}});
  const NDArray<float, 1> e({std::size_t{1}});
  const NDArray<double, 1> f({std::size_t{1}});
  (void)a;
  (void)b;
  (void)c;
  (void)d;
  (void)e;
  (void)f;
}

// Negative pin for bool: SFINAE cannot observe a hard error raised inside a class template's
// body (the static_assert fails at the point of class-template instantiation, not as a
// substitution failure). The intent is captured here; re-enabling the line below must refuse
// to compile, and the compile failure is the oracle. A CI job could un-comment this one line
// and assert the build fails, but within the test binary a compile-time probe is all we can
// land without introducing a new harness.
//
// static NDArray<bool, 1> mustNotCompile({std::size_t{1}});

} // namespace

TEST(NDArrayIntegerAllowlist, Int32Rank1ConstructAndIndex) {
  NDArray<std::int32_t, 1> labels({std::size_t{16}});
  EXPECT_EQ(labels.dim(0), 16U);
  for (std::size_t i = 0; i < 16; ++i) {
    labels(i) = static_cast<std::int32_t>(i * 3);
  }
  for (std::size_t i = 0; i < 16; ++i) {
    EXPECT_EQ(labels(i), static_cast<std::int32_t>(i * 3));
  }
}

TEST(NDArrayIntegerAllowlist, Int32Rank1IsAlignedAndAlignedDataReachable) {
  NDArray<std::int32_t, 1> labels({std::size_t{32}});
  ASSERT_NE(labels.data(), nullptr);
  EXPECT_TRUE(labels.isAligned<32>());
  // alignedData<32>() must be callable for integer element types; its return is a raw pointer
  // to the same storage, so a write through it must round-trip through operator().
  std::int32_t *aligned = labels.alignedData<32>();
  ASSERT_EQ(aligned, labels.data());
  aligned[0] = 42;
  EXPECT_EQ(labels(0), 42);
}

TEST(NDArrayIntegerAllowlist, Int32Rank1EmptyShapeHasNullData) {
  NDArray<std::int32_t, 1> labels({std::size_t{0}});
  EXPECT_EQ(labels.data(), nullptr);
  EXPECT_EQ(labels.dim(0), 0U);
}

TEST(NDArrayIntegerAllowlist, Uint8Rank2AccessorChainAndMutation) {
  NDArray<std::uint8_t, 2> mask({std::size_t{4}, std::size_t{3}});
  EXPECT_EQ(mask.dim(0), 4U);
  EXPECT_EQ(mask.dim(1), 3U);
  for (std::size_t i = 0; i < 4; ++i) {
    for (std::size_t j = 0; j < 3; ++j) {
      mask[i][j] = static_cast<std::uint8_t>((i * 3) + j);
    }
  }
  for (std::size_t i = 0; i < 4; ++i) {
    for (std::size_t j = 0; j < 3; ++j) {
      const std::uint8_t got = mask(i, j);
      EXPECT_EQ(got, static_cast<std::uint8_t>((i * 3) + j));
    }
  }
}

TEST(NDArrayIntegerAllowlist, Uint8Rank2IsContiguousAndAligned) {
  const NDArray<std::uint8_t, 2> mask({std::size_t{8}, std::size_t{8}});
  EXPECT_TRUE(mask.isContiguous());
  EXPECT_TRUE(mask.isAligned<32>());
  // numel deduced from shape: 8 * 8 = 64 elements.
  std::size_t count = 0;
  for (std::size_t i = 0; i < mask.dim(0); ++i) {
    for (std::size_t j = 0; j < mask.dim(1); ++j) {
      ++count;
    }
  }
  EXPECT_EQ(count, 64U);
}

TEST(NDArrayIntegerAllowlist, Int32Rank1BorrowSharesStorage) {
  std::vector<std::int32_t> raw(8, 0);
  for (std::size_t i = 0; i < raw.size(); ++i) {
    raw[i] = static_cast<std::int32_t>(i);
  }
  auto view = NDArray<std::int32_t, 1>::borrow(raw.data(), {raw.size()});
  EXPECT_TRUE(view.isMutable());
  EXPECT_FALSE(view.isOwned());
  for (std::size_t i = 0; i < raw.size(); ++i) {
    EXPECT_EQ(view(i), static_cast<std::int32_t>(i));
  }
}

TEST(NDArrayIntegerAllowlist, Int32Rank1MoveTransfersOwnership) {
  NDArray<std::int32_t, 1> src({std::size_t{4}});
  src(0) = 10;
  src(1) = 20;
  src(2) = 30;
  src(3) = 40;
  const std::int32_t *srcPtr = src.data();
  NDArray<std::int32_t, 1> dst = std::move(src);
  EXPECT_EQ(dst.data(), srcPtr);
  EXPECT_EQ(dst(0), 10);
  EXPECT_EQ(dst(3), 40);
}
