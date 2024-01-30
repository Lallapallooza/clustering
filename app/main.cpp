#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <vector>
#include "clustering/dbscan.h"

NDArray<float, 2> readCSV(const std::string &filename, size_t num_points, size_t dimensions) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    throw std::runtime_error("Unable to open file: " + filename);
  }

  NDArray<float, 2> data({num_points, dimensions});
  std::string       line;
  size_t            pointIndex = 0;

  while (std::getline(file, line) && pointIndex < num_points) {
    std::stringstream lineStream(line);
    std::string       cell;
    size_t            dimIndex = 0;

    while (std::getline(lineStream, cell, ',') && dimIndex < dimensions) {
      data[pointIndex][dimIndex] = std::stof(cell);
      ++dimIndex;
    }
    ++pointIndex;
  }

  return data;
}

int main(int argc, char *argv[]) {
  if (argc < 8) {
    std::cerr << "Usage: " << argv[0] << " <CSV file> "
                                         "<numPoints> "
                                         "<dimensions> "
                                         "<eps> "
                                         "<minPts> "
                                         "<njobs> "
                                         "<output file>\n";
    return 1;
  }

  std::string csvFile    = argv[1];
  size_t      numPoints  = std::stoul(argv[2]);
  size_t      dimensions = std::stoul(argv[3]);
  float       eps        = std::stof(argv[4]);
  size_t      minPts     = std::stoul(argv[5]);
  size_t      n_jobs     = std::stoul(argv[6]);
  std::string outputFile = argv[7];

  std::cout << "CSV File                : " << csvFile << std::endl;
  std::cout << "Number of Points        : " << numPoints << std::endl;
  std::cout << "Dimensions              : " << dimensions << std::endl;
  std::cout << "Epsilon (eps)           : " << eps << std::endl;
  std::cout << "Minimum Points (minPts) : " << minPts << std::endl;
  std::cout << "Number of Jobs (njobs)  : " << n_jobs << std::endl;
  std::cout << "Output File             : " << outputFile << std::endl << std::endl;

  NDArray<float, 2> points({numPoints, dimensions});
  try {
    points = readCSV(csvFile, numPoints, dimensions);
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  std::cout << "Start!" << std::endl;

  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
  DBSCAN<float> dbscan(points, eps, minPts, n_jobs);
  dbscan.run();
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

  std::cout << "Elapsed = "
            << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()
            << "μs or "
            << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()
            << "ms" << std::endl;

  const auto &labels = dbscan.labels();
  std::cout << "labels size: " << labels.size() << std::endl;
  std::cout << "nClusters: " << dbscan.nClusters() << std::endl;

  std::ofstream out(outputFile);
  if (!out.is_open()) {
    std::cerr << "Unable to open output file: " << outputFile << std::endl;
    return 1;
  }

  for (auto &i: labels) {
    out << i << std::endl;
  }

  return 0;
}
