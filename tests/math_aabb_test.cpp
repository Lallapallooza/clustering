#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <random>
#include <span>
#include <vector>

#include "clustering/math/aabb.h"

namespace {

// Scalar reference implementation, kept inline in the test so the assertion oracle is
// independent of any code path the public dispatcher might select. Mirrors the prior in-tree
// scalar `pointAabbGapSq` exactly.
float scalarPointAabbGapSq(std::span<const float> point, std::span<const float> boxMin,
                           std::span<const float> boxMax) {
  const std::size_t d = boxMin.size();
  float sum = 0.0F;
  for (std::size_t j = 0; j < d; ++j) {
    float gap = 0.0F;
    if (point[j] < boxMin[j]) {
      gap = boxMin[j] - point[j];
    } else if (point[j] > boxMax[j]) {
      gap = point[j] - boxMax[j];
    }
    sum += gap * gap;
  }
  return sum;
}

// ULP-bound absolute tolerance scaled by the magnitude of the larger of the two compared values.
// The AVX2 path differs from the scalar left-fold in two ways that compound rounding error:
//   (a) FMA (`fmadd(gap, gap, acc)`) does one rounding where the scalar does two (`gap*gap`
//       then add). FMA is generally more accurate per step; the result drifts a fraction of an
//       ulp per dim against the scalar reference, in either direction.
//   (b) The 8-lane horizontal reduction sums partials in a different order than the scalar
//       head-to-tail accumulator, so the rounding-tree depth differs.
// At single-tile d (8 / 16) the drift sits at <= ULP-2; at higher d (32 / 64) the accumulated
// error can reach a small constant times the per-tile ULP-2 budget. The tolerance scales by the
// number of accumulated 8-wide tiles plus one, which is a physically motivated upper bound on
// the per-element rounding budget; it stays at ULP-2 for d in [1, 8] and grows linearly with
// the tile count.
float ulpTol(float a, float b, std::size_t d) {
  constexpr float kEps = std::numeric_limits<float>::epsilon();
  const std::size_t tiles = ((d + 7) / 8) + 1;
  return 2.0F * static_cast<float>(tiles) * kEps * std::max({std::abs(a), std::abs(b), 1.0F});
}

// Single-tile (and below) overload: ULP-2 absolute tolerance for the case where the horizontal
// reduction has at most one ymm-wide partial. Used by deterministic test cases whose expected
// value is exact (zero or a small integer).
float ulpTol(float a, float b) {
  constexpr float kEps = std::numeric_limits<float>::epsilon();
  return 2.0F * kEps * std::max({std::abs(a), std::abs(b), 1.0F});
}

// Build a width-1 box centred at zero in d dimensions.
struct UnitBox {
  std::vector<float> mn;
  std::vector<float> mx;
};

UnitBox makeUnitBox(std::size_t d) {
  UnitBox box;
  box.mn.assign(d, -1.0F);
  box.mx.assign(d, 1.0F);
  return box;
}

} // namespace

// ---------------------------------------------------------------------------
// Acceptance criterion: point inside the box -> gap is zero, at d in {2, 4, 8, 16, 32, 64}.
// ---------------------------------------------------------------------------

TEST(MathAabbGapSq, PointInsideBoxGapIsZero) {
  for (const std::size_t d : {std::size_t{2}, std::size_t{4}, std::size_t{8}, std::size_t{16},
                              std::size_t{32}, std::size_t{64}}) {
    const auto box = makeUnitBox(d);
    std::vector<float> point(d, 0.25F);
    const auto got = clustering::math::pointAabbGapSq<float>(point.data(), box.mn, box.mx);
    EXPECT_EQ(got, 0.0F) << "d=" << d;
  }
}

// ---------------------------------------------------------------------------
// Acceptance criterion: point outside one face -> gap matches the squared face distance, at d
// in {2, 4, 8, 16, 32, 64}.
// ---------------------------------------------------------------------------

TEST(MathAabbGapSq, PointOutsideOneFaceGapMatches) {
  for (const std::size_t d : {std::size_t{2}, std::size_t{4}, std::size_t{8}, std::size_t{16},
                              std::size_t{32}, std::size_t{64}}) {
    const auto box = makeUnitBox(d);
    std::vector<float> point(d, 0.0F);
    // Push only dimension 0 outside the box by +0.5; expected gap^2 == 0.25.
    point[0] = 1.5F;
    const float expected = 0.25F;
    const auto got = clustering::math::pointAabbGapSq<float>(point.data(), box.mn, box.mx);
    EXPECT_NEAR(got, expected, ulpTol(got, expected)) << "d=" << d;
  }
}

// ---------------------------------------------------------------------------
// Acceptance criterion: point outside two faces -> gap sums the two squared face distances.
// ---------------------------------------------------------------------------

TEST(MathAabbGapSq, PointOutsideTwoFacesGapMatches) {
  for (const std::size_t d : {std::size_t{2}, std::size_t{4}, std::size_t{8}, std::size_t{16},
                              std::size_t{32}, std::size_t{64}}) {
    const auto box = makeUnitBox(d);
    std::vector<float> point(d, 0.0F);
    // Dim 0 above, dim 1 below; both gaps are 0.5; sum-of-squares = 0.5.
    point[0] = 1.5F;
    point[1] = -1.5F;
    const float expected = 0.5F;
    const auto got = clustering::math::pointAabbGapSq<float>(point.data(), box.mn, box.mx);
    EXPECT_NEAR(got, expected, ulpTol(got, expected)) << "d=" << d;
  }
}

// ---------------------------------------------------------------------------
// Acceptance criterion: point exactly on a face -> gap is zero. The reference's strict-less-than
// / strict-greater-than predicate treats on-face as inside.
// ---------------------------------------------------------------------------

TEST(MathAabbGapSq, PointOnFaceGapIsZero) {
  for (const std::size_t d : {std::size_t{2}, std::size_t{4}, std::size_t{8}, std::size_t{16},
                              std::size_t{32}, std::size_t{64}}) {
    const auto box = makeUnitBox(d);
    std::vector<float> point(d, 0.0F);
    point[0] = 1.0F; // exactly on the +x face
    const auto got = clustering::math::pointAabbGapSq<float>(point.data(), box.mn, box.mx);
    EXPECT_EQ(got, 0.0F) << "d=" << d;
  }
}

// ---------------------------------------------------------------------------
// Acceptance criterion: randomized inputs at d in {2, 4, 8, 16, 32, 64} agree with the scalar
// reference within ULP-2. Generates 50 random (point, boxMin, boxMax) triples per d. Boxes are
// constructed by sampling two independent corners and ordering them per-dimension so the AABB
// invariant (mn[j] <= mx[j]) holds. Points are sampled from a distribution that straddles the
// box centre so a healthy mix of inside / outside lanes appears at every d.
// ---------------------------------------------------------------------------

TEST(MathAabbGapSq, RandomizedAvx2VsScalarAgreesWithinTolerance) {
  std::mt19937_64 gen(0xA1B2C3D4ULL);
  std::uniform_real_distribution<float> coord(-3.0F, 3.0F);
  std::uniform_real_distribution<float> halfWidth(0.1F, 1.5F);

  constexpr std::size_t kTrialsPerD = 50;
  // Sweep covers: smallest tail-only (d=1), entirely-tail (d=2, 4), single-tile no-tail (d=8),
  // tile-plus-tail (d=11, d=20), multi-tile no-tail (d=16, 32, 64). The tile-plus-tail cases pin
  // the tail loop alongside the SIMD accumulator so a regression in either path surfaces.
  for (const std::size_t d :
       {std::size_t{1}, std::size_t{2}, std::size_t{4}, std::size_t{8}, std::size_t{11},
        std::size_t{16}, std::size_t{20}, std::size_t{32}, std::size_t{64}}) {
    std::vector<float> point(d);
    std::vector<float> mn(d);
    std::vector<float> mx(d);
    for (std::size_t trial = 0; trial < kTrialsPerD; ++trial) {
      for (std::size_t j = 0; j < d; ++j) {
        const float c = coord(gen);
        const float h = halfWidth(gen);
        mn[j] = c - h;
        mx[j] = c + h;
        point[j] = coord(gen);
      }
      const auto gotKernel = clustering::math::pointAabbGapSq<float>(point.data(), mn, mx);
      const float gotScalar = scalarPointAabbGapSq(point, mn, mx);
      EXPECT_NEAR(gotKernel, gotScalar, ulpTol(gotKernel, gotScalar, d))
          << "d=" << d << " trial=" << trial;
    }
  }
}
