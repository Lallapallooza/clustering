cmake_minimum_required(VERSION 3.22)
project(clustering_cpm_smoke LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# CLUSTERING_SOURCE_DIR is the path to a clustering checkout. This project
# tests `CPMAddPackage` with the vendored `cmake/CPM.cmake`. It uses the local
# checkout, so it needs no published tag.
if(NOT DEFINED CLUSTERING_SOURCE_DIR)
    message(
        FATAL_ERROR
        "CLUSTERING_SOURCE_DIR must point at the clustering repo root."
    )
endif()

include("${CLUSTERING_SOURCE_DIR}/cmake/CPM.cmake")
cpmaddpackage(NAME clustering SOURCE_DIR "${CLUSTERING_SOURCE_DIR}")

add_executable(consumer ../consumer/main.cpp)
target_link_libraries(consumer PRIVATE clustering::clustering)
