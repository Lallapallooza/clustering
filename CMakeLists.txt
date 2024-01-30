cmake_minimum_required(VERSION 3.26)
project(clustering)

include(CheckCXXCompilerFlag)
include(cmake/CPM.cmake)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CLUSTERING_BUILD_BENCHMARK ON CACHE BOOL "Build benchmark")
set(CLUSTERING_USE_AVX2 OFF CACHE BOOL "Use AVX2")
set(CLUSTERING_BUILD_WITH_SANITIZER OFF CACHE BOOL "Build with sanitizer")

add_library(clustering_header_lib INTERFACE)
target_include_directories(clustering_header_lib INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(clustering_demo app/main.cpp)
target_link_libraries(clustering_demo PRIVATE clustering_header_lib)

if (CLUSTERING_USE_AVX2)
    CHECK_CXX_COMPILER_FLAG("-mavx2" COMPILER_SUPPORTS_AVX2)
    if(COMPILER_SUPPORTS_AVX2)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2")
    else()
        message(FATAL_ERROR "AVX2 not supported but requested")
    endif()
    target_compile_definitions(clustering_header_lib INTERFACE CLUSTERING_USE_AVX2)
endif()

if (CLUSTERING_BUILD_WITH_SANITIZER)
   target_compile_options(clustering_header_lib INTERFACE -fsanitize=thread -fno-omit-frame-pointer)
   target_link_options(clustering_header_lib INTERFACE -fsanitize=thread)
   message(STATUS "Build with sanitizer")
endif()

if (CLUSTERING_BUILD_BENCHMARK)
    CPMAddPackage(
        NAME benchmark
        GITHUB_REPOSITORY google/benchmark
        VERSION 1.8.3
        OPTIONS "BENCHMARK_ENABLE_GTEST_TESTS OFF;BENCHMARK_ENABLE_TESTING OFF"
    )
    add_executable(kdtree_benchmark benchmark/kdtree_benchmark.cpp)

    target_link_libraries(kdtree_benchmark benchmark::benchmark clustering_header_lib)
endif()

CPMAddPackage(
    NAME tsl
    GITHUB_REPOSITORY Tessil/hopscotch-map
    GIT_TAG v2.3.1
)
target_link_libraries(clustering_header_lib INTERFACE tsl::hopscotch_map)
























