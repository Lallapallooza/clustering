#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#include "clustering/dbscan.h"

using clustering::DBSCAN;
using clustering::NDArray;

namespace {

NDArray<float, 2> readCSV(const std::string &filename, size_t num_points, size_t dimensions) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    throw std::runtime_error("Unable to open file: " + filename);
  }

  NDArray<float, 2> data({num_points, dimensions});
  std::string line;
  size_t pointIndex = 0;

  while (std::getline(file, line) && pointIndex < num_points) {
    std::stringstream lineStream(line);
    std::string cell;
    size_t dimIndex = 0;

    while (std::getline(lineStream, cell, ',') && dimIndex < dimensions) {
      data[pointIndex][dimIndex] = std::stof(cell);
      ++dimIndex;
    }
    ++pointIndex;
  }

  return data;
}

} // namespace

int main(int argc, char *argv[]) try {
  if (argc < 8) {
    std::cerr << "Usage: " << argv[0]
              << " <CSV file> "
                 "<numPoints> "
                 "<dimensions> "
                 "<eps> "
                 "<minPts> "
                 "<njobs> "
                 "<output file>\n";
    return 1;
  }

  const std::string csvFile = argv[1];
  const size_t numPoints = std::stoul(argv[2]);
  const size_t dimensions = std::stoul(argv[3]);
  const float eps = std::stof(argv[4]);
  const size_t minPts = std::stoul(argv[5]);
  const size_t n_jobs = std::stoul(argv[6]);
  const std::string outputFile = argv[7];

  std::cout << "CSV File                : " << csvFile << "\n";
  std::cout << "Number of Points        : " << numPoints << "\n";
  std::cout << "Dimensions              : " << dimensions << "\n";
  std::cout << "Epsilon (eps)           : " << eps << "\n";
  std::cout << "Minimum Points (minPts) : " << minPts << "\n";
  std::cout << "Number of Jobs (njobs)  : " << n_jobs << "\n";
  std::cout << "Output File             : " << outputFile << "\n\n";

  const NDArray<float, 2> points = readCSV(csvFile, numPoints, dimensions);

  std::cout << "Start!\n";

  const std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
  DBSCAN<float> dbscan(points, eps, minPts, n_jobs);
  dbscan.run();
  const std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

  std::cout << "Elapsed = "
            << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()
            << "us or "
            << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "ms"
            << "\n";

  const auto &labels = dbscan.labels();
  std::cout << "labels size: " << labels.size() << "\n";
  std::cout << "nClusters: " << dbscan.nClusters() << "\n";

  std::ofstream out(outputFile);
  if (!out.is_open()) {
    std::cerr << "Unable to open output file: " << outputFile << "\n";
    return 1;
  }

  for (const auto &i : labels) {
    out << i << "\n";
  }

  return 0;
} catch (const std::exception &e) {
  std::cerr << "Error: " << e.what() << "\n";
  return 1;
}
