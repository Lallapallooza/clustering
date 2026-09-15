// Packaging smoke test. One DBSCAN run checks that the consumer finds the
// clustering headers, the citor dependency, and the target's compile flags.

#include <cstddef>
#include <cstdlib>
#include <iostream>

#include "clustering/dbscan.h"

int main() {
  clustering::NDArray<float, 2> points({10, 2});
  for (std::size_t i = 0; i < 10; ++i) {
    const float base = i < 5 ? 0.0f : 10.0f;
    points[i][0] = base + (static_cast<float>(i % 5) * 0.1f);
    points[i][1] = 0.0f;
  }

  clustering::DBSCAN<float> dbscan(0.3f, 2, 2);
  dbscan.run(points);

  if (dbscan.nClusters() != 2U) {
    std::cerr << "expected 2 clusters, got " << dbscan.nClusters() << "\n";
    return EXIT_FAILURE;
  }
  std::cout << "clustering packaging smoke: OK\n";
  return EXIT_SUCCESS;
}
