#include <gtest/gtest.h>

#include <BS_thread_pool.hpp>
#include <cmath>
#include <cstddef>
#include <random>

#include "clustering/math/centroid_shift.h"
#include "clustering/math/thread.h"
#include "clustering/ndarray.h"

using clustering::NDArray;
using clustering::math::centroidShift;
using clustering::math::Pool;

namespace {

template <class T> void fillRandom(NDArray<T, 2> &a, std::uint32_t seed) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<T> dist(static_cast<T>(-1.0), static_cast<T>(1.0));
  const std::size_t n = a.dim(0);
  const std::size_t d = a.dim(1);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < d; ++j) {
      a[i][j] = dist(gen);
    }
  }
}

} // namespace

TEST(CentroidShift, MatchesScalarReference) {
  constexpr std::size_t k = 32;
  constexpr std::size_t d = 16;
  NDArray<float, 2> cOld({k, d});
  NDArray<float, 2> cNew({k, d});
  fillRandom(cOld, 101U);
  fillRandom(cNew, 102U);

  NDArray<float, 1> got({k});
  centroidShift(cOld, cNew, got, Pool{nullptr});

  for (std::size_t c = 0; c < k; ++c) {
    float expected = 0.0F;
    for (std::size_t t = 0; t < d; ++t) {
      const float diff = cNew(c, t) - cOld(c, t);
      expected += diff * diff;
    }
    const float denom = std::max(std::abs(expected), 1e-6F);
    EXPECT_LE(std::abs(got(c) - expected) / denom, 1e-5F) << "mismatch at c=" << c;
  }
}

TEST(CentroidShift, ZeroDeltaProducesZeroShift) {
  constexpr std::size_t k = 8;
  constexpr std::size_t d = 12;
  NDArray<float, 2> cOld({k, d});
  NDArray<float, 2> cNew({k, d});
  fillRandom(cOld, 201U);
  // Copy cOld into cNew so delta is identically zero everywhere.
  for (std::size_t c = 0; c < k; ++c) {
    for (std::size_t t = 0; t < d; ++t) {
      cNew(c, t) = cOld(c, t);
    }
  }

  NDArray<float, 1> got({k});
  centroidShift(cOld, cNew, got, Pool{nullptr});

  for (std::size_t c = 0; c < k; ++c) {
    EXPECT_FLOAT_EQ(got(c), 0.0F) << "c=" << c;
  }
}

TEST(CentroidShift, SerialAndThreadedAgree) {
  constexpr std::size_t k = 128;
  constexpr std::size_t d = 32;
  NDArray<float, 2> cOld({k, d});
  NDArray<float, 2> cNew({k, d});
  fillRandom(cOld, 301U);
  fillRandom(cNew, 302U);

  NDArray<float, 1> serial({k});
  centroidShift(cOld, cNew, serial, Pool{nullptr});

  BS::light_thread_pool tp(4);
  NDArray<float, 1> threaded({k});
  centroidShift(cOld, cNew, threaded, Pool{&tp});

  for (std::size_t c = 0; c < k; ++c) {
    // Per-row arithmetic is untouched by the outer fanout; bit-identical agreement expected.
    EXPECT_FLOAT_EQ(serial(c), threaded(c)) << "c=" << c;
  }
}

TEST(CentroidShift, EmptyKIsNoOp) {
  const NDArray<float, 2> cOld({0, 8});
  const NDArray<float, 2> cNew({0, 8});
  NDArray<float, 1> out({0});
  centroidShift(cOld, cNew, out, Pool{nullptr});
  EXPECT_EQ(out.dim(0), 0U);
}

TEST(CentroidShift, DoubleVariantMatchesScalar) {
  constexpr std::size_t k = 10;
  constexpr std::size_t d = 7;
  NDArray<double, 2> cOldMut({k, d});
  NDArray<double, 2> cNewMut({k, d});
  fillRandom(cOldMut, 401U);
  fillRandom(cNewMut, 402U);
  const NDArray<double, 2> &cOld = cOldMut;
  const NDArray<double, 2> &cNew = cNewMut;

  NDArray<double, 1> got({k});
  centroidShift(cOld, cNew, got, Pool{nullptr});

  for (std::size_t c = 0; c < k; ++c) {
    double expected = 0.0;
    for (std::size_t t = 0; t < d; ++t) {
      const double diff = cNew(c, t) - cOld(c, t);
      expected += diff * diff;
    }
    const double denom = std::max(std::abs(expected), 1e-10);
    EXPECT_LE(std::abs(got(c) - expected) / denom, 1e-12) << "c=" << c;
  }
}
