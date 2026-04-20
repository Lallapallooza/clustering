#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "clustering/hdbscan/detail/condensed_tree.h"
#include "clustering/hdbscan/detail/glosh.h"

using clustering::hdbscan::detail::computeGlosh;
using clustering::hdbscan::detail::CondensedTree;

// Hand-crafted condensed tree matching the Campello 2015 GLOSH definition.
//
// Convention: N points, cluster ids in [N, N + numClusters). For N = 6, cluster ids live in
// [6, 6 + numClusters). Root is always cluster id N.
//
// Tree layout (N = 6 points; one root cluster id = 6, two child clusters id = 7, 8):
//
//   root(6) -- child 7 at lambda = 1.0 (internal cluster, size 3)
//   root(6) -- child 8 at lambda = 1.0 (internal cluster, size 3)
//   7 -- point 0 at lambda = 3.0
//   7 -- point 1 at lambda = 3.0
//   7 -- point 2 at lambda = 3.0
//   8 -- point 3 at lambda = 2.0
//   8 -- point 4 at lambda = 2.0
//   8 -- point 5 at lambda = 2.0
//
// lambdaMax(root 6) = max over subtree = max(1.0, 1.0, 3.0, 3.0, 3.0, 2.0, 2.0, 2.0) = 3.0.
// lambdaMax(cluster 7) = 3.0; lambdaMax(cluster 8) = 2.0.
//
// By the Campello formula:
//   points 0,1,2 in cluster 7: score = (3.0 - 3.0) / 3.0 = 0.0
//   points 3,4,5 in cluster 8: score = (2.0 - 2.0) / 2.0 = 0.0
TEST(Glosh, HandCraftedAllPointsAtSubtreeMax) {
  CondensedTree<float> tree;
  tree.parent = {6, 6, 7, 7, 7, 8, 8, 8};
  tree.child = {7, 8, 0, 1, 2, 3, 4, 5};
  tree.lambdaVal = {1.0F, 1.0F, 3.0F, 3.0F, 3.0F, 2.0F, 2.0F, 2.0F};
  tree.childSize = {3, 3, 1, 1, 1, 1, 1, 1};
  tree.numClusters = 3; // clusters 6, 7, 8 (count including root).

  const std::vector<std::int32_t> labels(6, 0);
  std::vector<float> scores;
  computeGlosh(tree, 6, labels, scores);
  ASSERT_EQ(scores.size(), 6u);
  for (std::size_t i = 0; i < 6; ++i) {
    EXPECT_NEAR(scores[i], 0.0F, 1e-5F);
  }
}

// Same tree, but with a "dying sub-cluster" shape: cluster 8 contains points that fall out BEFORE
// cluster 8's subtree reaches its own max.
//
//   root(6) -- child 7 at lambda = 1.0 (internal, size 3)
//   root(6) -- child 8 at lambda = 1.0 (internal, size 3)
//   7 -- point 0 at lambda = 5.0
//   7 -- point 1 at lambda = 5.0
//   7 -- point 2 at lambda = 5.0
//   8 -- point 3 at lambda = 2.0      # drops out early
//   8 -- point 4 at lambda = 2.0      # drops out early
//   8 -- point 5 at lambda = 2.0      # drops out early
//
// Campello formula uses `lambdaMax(8)` = 2.0 (the cluster's subtree max) for points in cluster 8.
// So points 3, 4, 5 score (2.0 - 2.0) / 2.0 = 0.0.
//
// The point of the dying-subcluster construction comes when a sub-cluster 8 drops out, with its
// siblings in 7 continuing to merge tightly. A scikit-learn-contrib/hdbscan implementation of
// GLOSH that used the CHILD's own lambda rather than the PARENT's subtree max would diverge here.
// Since both tree layouts here correctly use subtree max, scores are 0.
//
// The crucial fixture for the bug lives below.
TEST(Glosh, HandCraftedDyingSubcluster) {
  CondensedTree<float> tree;
  tree.parent = {6, 6, 7, 7, 7, 8, 8, 8};
  tree.child = {7, 8, 0, 1, 2, 3, 4, 5};
  tree.lambdaVal = {1.0F, 1.0F, 5.0F, 5.0F, 5.0F, 2.0F, 2.0F, 2.0F};
  tree.childSize = {3, 3, 1, 1, 1, 1, 1, 1};
  tree.numClusters = 3;

  const std::vector<std::int32_t> labels(6, 0);
  std::vector<float> scores;
  computeGlosh(tree, 6, labels, scores);

  // Every score must be in [0, 1].
  for (std::size_t i = 0; i < 6; ++i) {
    EXPECT_GE(scores[i], 0.0F);
    EXPECT_LE(scores[i], 1.0F);
  }
  // Points 0, 1, 2 are in cluster 7 with lambdaMax(7) = 5.0 and lambda(x) = 5.0: score = 0.
  EXPECT_NEAR(scores[0], 0.0F, 1e-5F);
  EXPECT_NEAR(scores[1], 0.0F, 1e-5F);
  EXPECT_NEAR(scores[2], 0.0F, 1e-5F);
  // Points 3, 4, 5 are in cluster 8 with lambdaMax(8) = 2.0 and lambda(x) = 2.0: score = 0.
  EXPECT_NEAR(scores[3], 0.0F, 1e-5F);
  EXPECT_NEAR(scores[4], 0.0F, 1e-5F);
  EXPECT_NEAR(scores[5], 0.0F, 1e-5F);
}

// Non-trivial closed-form: points that fall out before lambdaMax get non-zero scores.
//
// Convention: for N = 3, cluster ids live in [3, 3 + numClusters).
//
//   root(3) -- child 4 at lambda = 1.0 (internal, size 3)
//   4 -- point 0 at lambda = 10.0       # persists to depth
//   4 -- point 1 at lambda = 5.0        # drops early
//   4 -- point 2 at lambda = 2.0        # drops earlier
//
// lambdaMax(4) = 10.0, lambdaMax(3) = 10.0.
// Points 0, 1, 2 are in cluster 4:
//   score(0) = (10 - 10) / 10 = 0
//   score(1) = (10 - 5) / 10 = 0.5
//   score(2) = (10 - 2) / 10 = 0.8
TEST(Glosh, HandCraftedClosedForm) {
  CondensedTree<float> tree;
  tree.parent = {3, 4, 4, 4};
  tree.child = {4, 0, 1, 2};
  tree.lambdaVal = {1.0F, 10.0F, 5.0F, 2.0F};
  tree.childSize = {3, 1, 1, 1};
  tree.numClusters = 2;

  const std::vector<std::int32_t> labels{0, 0, 0};
  std::vector<float> scores;
  computeGlosh(tree, 3, labels, scores);
  ASSERT_EQ(scores.size(), 3u);
  EXPECT_NEAR(scores[0], 0.0F, 1e-5F);
  EXPECT_NEAR(scores[1], 0.5F, 1e-5F);
  EXPECT_NEAR(scores[2], 0.8F, 1e-5F);
}

// Bug-distinguishing fixture: a construction where the buggy "use the dying sub-cluster's own
// death lambda" formula would yield scores > 1, while the correct Campello formula yields [0, 1].
//
// The classic layout: a parent cluster whose lambdaMax is large (contributed by a high-lambda
// sibling sub-tree), and a point contained in the parent that falls out at a *low* lambda. The
// Campello normaliser is the parent's lambdaMax (large); the bugged normaliser would be the
// child's own death lambda (low), producing scores > 1 for any point whose fall-out lambda exceeds
// the child's own death.
//
//   root(3) -- cluster 4 at lambda = 0.5 (internal, size 2)
//   root(3) -- point 0 at lambda = 0.5  # falls out of root early
//   4 -- point 1 at lambda = 10.0
//   4 -- point 2 at lambda = 10.0
//
// lambdaMax(root=3) = 10.0 (propagated from cluster 4). lambdaMax(4) = 10.0.
// Point 0 is in cluster 3; Campello: score(0) = (10 - 0.5) / 10 = 0.95. Bounded in [0, 1].
// Points 1, 2 in cluster 4: score = (10 - 10) / 10 = 0.
TEST(Glosh, BugDistinguishingFixtureBoundedScore) {
  CondensedTree<float> tree;
  tree.parent = {3, 3, 4, 4};
  tree.child = {4, 0, 1, 2};
  tree.lambdaVal = {0.5F, 0.5F, 10.0F, 10.0F};
  tree.childSize = {2, 1, 1, 1};
  tree.numClusters = 2;

  const std::vector<std::int32_t> labels{-1, 0, 0};
  std::vector<float> scores;
  computeGlosh(tree, 3, labels, scores);
  ASSERT_EQ(scores.size(), 3u);
  EXPECT_NEAR(scores[0], 0.95F, 1e-5F);
  EXPECT_NEAR(scores[1], 0.0F, 1e-5F);
  EXPECT_NEAR(scores[2], 0.0F, 1e-5F);
  // Strict bound: every score in [0, 1].
  for (const float s : scores) {
    EXPECT_GE(s, 0.0F);
    EXPECT_LE(s, 1.0F);
  }
}

// Degenerate: empty tree produces zero scores for all points (no condensed structure).
TEST(Glosh, EmptyTreeZeroScores) {
  CondensedTree<float> tree;
  tree.numClusters = 0;
  const std::vector<std::int32_t> labels(4, -1);
  std::vector<float> scores;
  computeGlosh(tree, 4, labels, scores);
  ASSERT_EQ(scores.size(), 4u);
  for (const float s : scores) {
    EXPECT_EQ(s, 0.0F);
  }
}

// Degenerate: infinite lambdaMax (zero-distance merge produced lambda = +inf) yields score 1 for
// finite-lambda points and score 0 for infinite-lambda points.
//
// N = 3 so the only valid cluster id is 3 (the root).
TEST(Glosh, InfiniteLambdaMaxFallbacks) {
  CondensedTree<float> tree;
  const float kInf = std::numeric_limits<float>::infinity();
  tree.parent = {3, 3, 3};
  tree.child = {0, 1, 2};
  tree.lambdaVal = {kInf, 2.0F, kInf};
  tree.childSize = {1, 1, 1};
  tree.numClusters = 1;

  const std::vector<std::int32_t> labels(3, 0);
  std::vector<float> scores;
  computeGlosh(tree, 3, labels, scores);
  ASSERT_EQ(scores.size(), 3u);
  EXPECT_EQ(scores[0], 0.0F);
  EXPECT_EQ(scores[1], 1.0F);
  EXPECT_EQ(scores[2], 0.0F);
}
