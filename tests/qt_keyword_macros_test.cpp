// Every public header must compile after Qt's keyword macros are defined. Qt
// defines `slots`, `signals`, `emit` and `foreach` as macros, so an identifier
// with one of those names breaks or silently changes meaning in any
// translation unit that includes a Qt header first. The definitions below
// match Qt's `qobjectdefs.h` and `qforeach.h`.

#define Q_SLOTS
#define Q_SIGNALS public
#define Q_EMIT
// Qt's definition: `variable` is a declaration such as `auto x`, which cannot
// be parenthesized.
// NOLINTNEXTLINE(bugprone-macro-parentheses)
#define Q_FOREACH(variable, container) for (variable : container)
#define slots Q_SLOTS
#define signals Q_SIGNALS
#define emit Q_EMIT
#define foreach Q_FOREACH

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>

#include "clustering/always_assert.h"
#include "clustering/dbscan.h"
#include "clustering/hdbscan.h"
#include "clustering/index/brute_force_pairwise.h"
#include "clustering/index/kdtree.h"
#include "clustering/kmeans.h"
#include "clustering/ndarray.h"
#include "clustering/ndarray_range.h"

using clustering::BruteForcePairwise;
using clustering::DBSCAN;
using clustering::KDTree;
using clustering::KDTreeDistanceType;
using clustering::NDArray;

namespace {

constexpr std::size_t kPointsPerBlob = 200;
constexpr std::size_t kDims = 16;

// Two blobs far apart along every axis; points within a blob sit on a small
// deterministic lattice so each one has many neighbours inside `eps`.
NDArray<float, 2> twoBlobs() {
  NDArray<float, 2> points({2 * kPointsPerBlob, kDims});
  for (std::size_t i = 0; i < 2 * kPointsPerBlob; ++i) {
    const float centre = i < kPointsPerBlob ? 0.0F : 100.0F;
    for (std::size_t d = 0; d < kDims; ++d) {
      const auto step = static_cast<float>((i * 7 + d * 3) % 5);
      points[i][d] = centre + (0.05F * step);
    }
  }
  return points;
}

template <class Index> void expectTwoBlobs() {
  const NDArray<float, 2> points = twoBlobs();
  DBSCAN<float, Index> dbscan(/*eps=*/1.0F, /*minPts=*/4, /*nJobs=*/2);
  dbscan.run(points);
  ASSERT_EQ(dbscan.nClusters(), 2U);
  const std::int32_t first = dbscan.labels().flatIndex(0);
  const std::int32_t second = dbscan.labels().flatIndex(kPointsPerBlob);
  EXPECT_NE(first, second);
  for (std::size_t i = 0; i < kPointsPerBlob; ++i) {
    EXPECT_EQ(dbscan.labels().flatIndex(i), first);
    EXPECT_EQ(dbscan.labels().flatIndex(kPointsPerBlob + i), second);
  }
}

} // namespace

TEST(QtKeywordMacros, DbscanWithKdTreeFindsBothBlobs) {
  expectTwoBlobs<KDTree<float, KDTreeDistanceType::kEucledian>>();
}

TEST(QtKeywordMacros, DbscanWithBruteForceFindsBothBlobs) {
  expectTwoBlobs<BruteForcePairwise<float>>();
}
