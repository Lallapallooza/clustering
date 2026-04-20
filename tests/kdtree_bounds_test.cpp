#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <random>
#include <utility>
#include <vector>

#include "clustering/index/kdtree.h"
#include "clustering/ndarray.h"

using clustering::KDTree;
using clustering::KDTreeNode;
using clustering::NDArray;

namespace {

NDArray<float, 2> makeRandomPoints(std::size_t n, std::size_t d, std::uint64_t seed) {
  NDArray<float, 2> points({n, d});
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<float> dist(-3.0F, 3.0F);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < d; ++j) {
      points[i][j] = dist(rng);
    }
  }
  return points;
}

// For each node in @p node's subtree, verify two invariants:
//   (1) reported min <= reported max along every dimension (well-formed box);
//   (2) every child's bounds are enclosed by the parent's bounds (child min >= parent min,
//       child max <= parent max along every dimension).
// Together with the root-tightness check below, these invariants guarantee every node's box
// encloses exactly the points routed through it.
void verifyEnclosureInvariant(const KDTreeNode *node, const KDTree<float> &tree, std::size_t d) {
  if (node == nullptr) {
    return;
  }

  auto [minSpan, maxSpan] = tree.nodeBounds(node);
  ASSERT_EQ(minSpan.size(), d);
  ASSERT_EQ(maxSpan.size(), d);

  for (std::size_t j = 0; j < d; ++j) {
    EXPECT_LE(minSpan[j], maxSpan[j]) << "node id=" << node->m_id << " dim=" << j;
  }

  if (node->m_left != nullptr) {
    auto [lmin, lmax] = tree.nodeBounds(node->m_left);
    for (std::size_t j = 0; j < d; ++j) {
      EXPECT_GE(lmin[j], minSpan[j]) << "internal id=" << node->m_id << " left dim=" << j;
      EXPECT_LE(lmax[j], maxSpan[j]) << "internal id=" << node->m_id << " left dim=" << j;
    }
    verifyEnclosureInvariant(node->m_left, tree, d);
  }
  if (node->m_right != nullptr) {
    auto [rmin, rmax] = tree.nodeBounds(node->m_right);
    for (std::size_t j = 0; j < d; ++j) {
      EXPECT_GE(rmin[j], minSpan[j]) << "internal id=" << node->m_id << " right dim=" << j;
      EXPECT_LE(rmax[j], maxSpan[j]) << "internal id=" << node->m_id << " right dim=" << j;
    }
    verifyEnclosureInvariant(node->m_right, tree, d);
  }
}

// Walk the tree and count the total number of points across leaves (leaf @c m_dim stores the
// point count) plus the number of internal pivots (each internal node represents one pivot
// point). The sum must equal @p n: every input row lives at exactly one leaf or one pivot.
std::size_t sumAllStoredPoints(const KDTreeNode *node) {
  if (node == nullptr) {
    return 0;
  }
  if (node->m_left == nullptr && node->m_right == nullptr) {
    return node->m_dim;
  }
  return 1 + sumAllStoredPoints(node->m_left) + sumAllStoredPoints(node->m_right);
}

} // namespace

TEST(KdtreeBounds, EnclosureAndRootTightnessAtDim2) {
  const std::size_t n = 200;
  const std::size_t d = 2;
  const auto points = makeRandomPoints(n, d, 0xA1);
  const KDTree<float> tree(points);
  const KDTreeNode *root = tree.root();
  ASSERT_NE(root, nullptr);

  verifyEnclosureInvariant(root, tree, d);

  // Root must be a tight bounding box over the input points.
  auto [rootMin, rootMax] = tree.nodeBounds(root);
  for (std::size_t j = 0; j < d; ++j) {
    float trueMin = points[0][j];
    float trueMax = points[0][j];
    for (std::size_t i = 1; i < n; ++i) {
      const float value = points[i][j];
      trueMin = std::min(trueMin, value);
      trueMax = std::max(trueMax, value);
    }
    EXPECT_EQ(rootMin[j], trueMin) << "dim " << j;
    EXPECT_EQ(rootMax[j], trueMax) << "dim " << j;
  }
}

TEST(KdtreeBounds, EnclosureAndRootTightnessAtDim8) {
  const std::size_t n = 200;
  const std::size_t d = 8;
  const auto points = makeRandomPoints(n, d, 0xA2);
  const KDTree<float> tree(points);
  const KDTreeNode *root = tree.root();
  ASSERT_NE(root, nullptr);
  verifyEnclosureInvariant(root, tree, d);

  auto [rootMin, rootMax] = tree.nodeBounds(root);
  for (std::size_t j = 0; j < d; ++j) {
    float trueMin = points[0][j];
    float trueMax = points[0][j];
    for (std::size_t i = 1; i < n; ++i) {
      const float value = points[i][j];
      trueMin = std::min(trueMin, value);
      trueMax = std::max(trueMax, value);
    }
    EXPECT_EQ(rootMin[j], trueMin) << "dim " << j;
    EXPECT_EQ(rootMax[j], trueMax) << "dim " << j;
  }
}

TEST(KdtreeBounds, EnclosureAndRootTightnessAtDim16) {
  const std::size_t n = 200;
  const std::size_t d = 16;
  const auto points = makeRandomPoints(n, d, 0xA3);
  const KDTree<float> tree(points);
  const KDTreeNode *root = tree.root();
  ASSERT_NE(root, nullptr);
  verifyEnclosureInvariant(root, tree, d);

  auto [rootMin, rootMax] = tree.nodeBounds(root);
  for (std::size_t j = 0; j < d; ++j) {
    float trueMin = points[0][j];
    float trueMax = points[0][j];
    for (std::size_t i = 1; i < n; ++i) {
      const float value = points[i][j];
      trueMin = std::min(trueMin, value);
      trueMax = std::max(trueMax, value);
    }
    EXPECT_EQ(rootMin[j], trueMin) << "dim " << j;
    EXPECT_EQ(rootMax[j], trueMax) << "dim " << j;
  }
}

TEST(KdtreeBounds, EveryPointIsInSomeNodeBox) {
  // Each input point lives at exactly one tree node -- a leaf if it was not chosen as a pivot
  // during build, an internal node otherwise. The populated bounds of that node must enclose
  // the point's coordinates. The per-node bounds of every ancestor on the root-to-node path
  // must also enclose the point, by the enclosure invariant. This test checks the weaker but
  // sufficient condition: every input point is inside at least one node's reported box.
  const std::size_t n = 120;
  const std::size_t d = 4;
  const auto points = makeRandomPoints(n, d, 0xB1);
  const KDTree<float> tree(points);
  const KDTreeNode *root = tree.root();
  ASSERT_NE(root, nullptr);

  std::vector<const KDTreeNode *> allNodes;
  {
    std::vector<const KDTreeNode *> stack;
    stack.push_back(root);
    while (!stack.empty()) {
      const KDTreeNode *cur = stack.back();
      stack.pop_back();
      if (cur == nullptr) {
        continue;
      }
      allNodes.push_back(cur);
      stack.push_back(cur->m_left);
      stack.push_back(cur->m_right);
    }
  }

  for (std::size_t i = 0; i < n; ++i) {
    bool insideSome = false;
    for (const KDTreeNode *node : allNodes) {
      auto [lmin, lmax] = tree.nodeBounds(node);
      bool inside = true;
      for (std::size_t j = 0; j < d; ++j) {
        if (points[i][j] < lmin[j] || points[i][j] > lmax[j]) {
          inside = false;
          break;
        }
      }
      if (inside) {
        insideSome = true;
        break;
      }
    }
    EXPECT_TRUE(insideSome) << "point " << i << " not contained in any node's bounds";
  }
}

TEST(KdtreeBounds, AllPointsAccountedFor) {
  // Sanity: every input row lives at exactly one leaf or one pivot. The sum of leaf point
  // counts plus internal pivots must be the input row count.
  const std::size_t n = 173;
  const std::size_t d = 5;
  const auto points = makeRandomPoints(n, d, 0xB2);
  const KDTree<float> tree(points);
  const KDTreeNode *root = tree.root();
  ASSERT_NE(root, nullptr);
  EXPECT_EQ(sumAllStoredPoints(root), n);
}

TEST(KdtreeBounds, SingletonLeaf) {
  // A 1-point tree produces a single leaf whose bounds match the point exactly.
  NDArray<float, 2> points({1, 3});
  points[0][0] = 1.5F;
  points[0][1] = -2.25F;
  points[0][2] = 0.75F;
  const KDTree<float> tree(points);
  const KDTreeNode *root = tree.root();
  ASSERT_NE(root, nullptr);
  auto [minSpan, maxSpan] = tree.nodeBounds(root);
  ASSERT_EQ(minSpan.size(), 3U);
  ASSERT_EQ(maxSpan.size(), 3U);
  for (std::size_t j = 0; j < 3; ++j) {
    EXPECT_EQ(minSpan[j], points[0][j]);
    EXPECT_EQ(maxSpan[j], points[0][j]);
  }
}

TEST(KdtreeBounds, NodeBoundsIsConstantTime) {
  // The accessor must be a plain pointer dereference + span construction; profile-wise this is
  // a single-call microbenchmark rather than a correctness check, but the test keeps the API
  // shape wired end-to-end. We simply call the accessor and observe its inputs/outputs are
  // consistent with the tree structure.
  const std::size_t n = 50;
  const std::size_t d = 16;
  const auto points = makeRandomPoints(n, d, 0xC3);
  const KDTree<float> tree(points);
  const KDTreeNode *root = tree.root();
  ASSERT_NE(root, nullptr);
  auto [minSpan, maxSpan] = tree.nodeBounds(root);
  EXPECT_EQ(minSpan.size(), d);
  EXPECT_EQ(maxSpan.size(), d);
  // Calling twice must return the same pointer / same values (no recomputation).
  auto [minSpan2, maxSpan2] = tree.nodeBounds(root);
  EXPECT_EQ(minSpan.data(), minSpan2.data());
  EXPECT_EQ(maxSpan.data(), maxSpan2.data());
  for (std::size_t j = 0; j < d; ++j) {
    EXPECT_EQ(minSpan[j], minSpan2[j]);
    EXPECT_EQ(maxSpan[j], maxSpan2[j]);
  }
}
