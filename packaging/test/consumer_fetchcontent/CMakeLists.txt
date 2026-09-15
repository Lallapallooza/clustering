cmake_minimum_required(VERSION 3.22)
project(clustering_fetchcontent_smoke LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# CLUSTERING_SOURCE_DIR is the path to a clustering checkout. This project
# tests `FetchContent_MakeAvailable`. It uses the local checkout, so it needs
# no published tag.
if(NOT DEFINED CLUSTERING_SOURCE_DIR)
    message(
        FATAL_ERROR
        "CLUSTERING_SOURCE_DIR must point at the clustering repo root."
    )
endif()

include(FetchContent)
FetchContent_Declare(clustering SOURCE_DIR "${CLUSTERING_SOURCE_DIR}")
FetchContent_MakeAvailable(clustering)

add_executable(consumer ../consumer/main.cpp)
target_link_libraries(consumer PRIVATE clustering::clustering)
