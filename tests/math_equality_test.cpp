#include <gtest/gtest.h>

#include <cstddef>

#include "clustering/math/equality.h"
#include "clustering/ndarray.h"

using clustering::Layout;
using clustering::NDArray;

TEST(MathArrayEqual, ShapeMismatchReturnsFalse) {
  NDArray<float, 2> a({2, 3});
  NDArray<float, 2> b({3, 2});
  for (std::size_t i = 0; i < 2; ++i) {
    for (std::size_t j = 0; j < 3; ++j) {
      a[i][j] = 1.0F;
    }
  }
  for (std::size_t i = 0; i < 3; ++i) {
    for (std::size_t j = 0; j < 2; ++j) {
      b[i][j] = 1.0F;
    }
  }
  EXPECT_FALSE(clustering::math::arrayEqual(a, b));
}

TEST(MathArrayEqual, ElementMismatchReturnsFalse) {
  NDArray<float, 2> a({2, 3});
  NDArray<float, 2> b({2, 3});
  for (std::size_t i = 0; i < 2; ++i) {
    for (std::size_t j = 0; j < 3; ++j) {
      a[i][j] = static_cast<float>(i * 3 + j);
      b[i][j] = static_cast<float>(i * 3 + j);
    }
  }
  b[1][2] = 99.0F;
  EXPECT_FALSE(clustering::math::arrayEqual(a, b));
}

TEST(MathArrayEqual, ExactMatchReturnsTrue) {
  NDArray<double, 2> a({4, 5});
  NDArray<double, 2> b({4, 5});
  for (std::size_t i = 0; i < 4; ++i) {
    for (std::size_t j = 0; j < 5; ++j) {
      const double v = static_cast<double>(i) * 0.25 - static_cast<double>(j);
      a[i][j] = v;
      b[i][j] = v;
    }
  }
  EXPECT_TRUE(clustering::math::arrayEqual(a, b));
}

TEST(MathArrayEqual, EmptyArraysReturnTrue) {
  NDArray<float, 2> a({0, 5});
  NDArray<float, 2> b({0, 5});
  EXPECT_TRUE(clustering::math::arrayEqual(a, b));
}

TEST(MathArrayEqual, StridedVsContiguousSameValues) {
  NDArray<float, 2> a({3, 2});
  for (std::size_t i = 0; i < 3; ++i) {
    for (std::size_t j = 0; j < 2; ++j) {
      a[i][j] = static_cast<float>(i * 2 + j);
    }
  }
  auto roundTrip = a.t().t();
  EXPECT_EQ(roundTrip.dim(0), 3u);
  EXPECT_EQ(roundTrip.dim(1), 2u);
  EXPECT_TRUE(clustering::math::arrayEqual(a, roundTrip));
}

TEST(MathAllClose, ExactEqualReturnsTrue) {
  NDArray<float, 2> a({2, 3});
  NDArray<float, 2> b({2, 3});
  for (std::size_t i = 0; i < 2; ++i) {
    for (std::size_t j = 0; j < 3; ++j) {
      a[i][j] = 0.5F + static_cast<float>(j);
      b[i][j] = 0.5F + static_cast<float>(j);
    }
  }
  EXPECT_TRUE(clustering::math::allClose(a, b));
}

TEST(MathAllClose, DifferenceWithinTolReturnsTrue) {
  NDArray<float, 1> a({4});
  NDArray<float, 1> b({4});
  a(0) = 1.0F;
  a(1) = 2.0F;
  a(2) = 3.0F;
  a(3) = 4.0F;
  b(0) = 1.0F;
  b(1) = 2.0F + 5.0e-7F;
  b(2) = 3.0F;
  b(3) = 4.0F;
  EXPECT_TRUE(clustering::math::allClose(a, b));
}

TEST(MathAllClose, DifferenceOutsideTolReturnsFalse) {
  NDArray<float, 1> a({4});
  NDArray<float, 1> b({4});
  a(0) = 1.0F;
  a(1) = 2.0F;
  a(2) = 3.0F;
  a(3) = 4.0F;
  b(0) = 1.0F;
  b(1) = 2.0F;
  b(2) = 3.0F + 1.0F;
  b(3) = 4.0F;
  EXPECT_FALSE(clustering::math::allClose(a, b));
}

TEST(MathAllClose, ShapeMismatchReturnsFalse) {
  NDArray<double, 2> a({2, 4});
  NDArray<double, 2> b({4, 2});
  for (std::size_t i = 0; i < 2; ++i) {
    for (std::size_t j = 0; j < 4; ++j) {
      a[i][j] = 0.0;
    }
  }
  for (std::size_t i = 0; i < 4; ++i) {
    for (std::size_t j = 0; j < 2; ++j) {
      b[i][j] = 0.0;
    }
  }
  EXPECT_FALSE(clustering::math::allClose(a, b));
}
