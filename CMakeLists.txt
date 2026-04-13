cmake_minimum_required(VERSION 3.22)
project(clustering)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD_INCLUDE_DIRECTORIES
    ${CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES}
)
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

include(CheckCXXCompilerFlag)
include(cmake/CPM.cmake)
include(cmake/Tooling.cmake)
include(cmake/ThreadPoolDep.cmake)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CLUSTERING_BUILD_BENCHMARK ON CACHE BOOL "Build benchmark")
check_cxx_compiler_flag("-mavx2" CLUSTERING_COMPILER_SUPPORTS_AVX2)
set(CLUSTERING_USE_AVX2
    ${CLUSTERING_COMPILER_SUPPORTS_AVX2}
    CACHE BOOL
    "Use AVX2"
)
set(CLUSTERING_BUILD_WITH_SANITIZER OFF CACHE BOOL "Build with sanitizer")

add_library(clustering_header_lib INTERFACE)
target_include_directories(
    clustering_header_lib
    INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(clustering_header_lib INTERFACE bshoshany_thread_pool)

add_executable(clustering_demo app/main.cpp)
target_link_libraries(clustering_demo PRIVATE clustering_header_lib)
clustering_enable_tidy(clustering_demo)

if(CLUSTERING_USE_AVX2)
    if(NOT CLUSTERING_COMPILER_SUPPORTS_AVX2)
        message(FATAL_ERROR "AVX2 not supported but requested")
    endif()
    target_compile_options(clustering_header_lib INTERFACE -mavx2)
    target_compile_definitions(
        clustering_header_lib
        INTERFACE CLUSTERING_USE_AVX2
    )
endif()

if(CLUSTERING_BUILD_WITH_SANITIZER)
    target_compile_options(
        clustering_header_lib
        INTERFACE -fsanitize=thread -fno-omit-frame-pointer
    )
    target_link_options(clustering_header_lib INTERFACE -fsanitize=thread)
    message(STATUS "Build with sanitizer")
endif()

if(CLUSTERING_BUILD_BENCHMARK)
    cpmaddpackage(
        NAME benchmark
        GITHUB_REPOSITORY google/benchmark
        VERSION 1.9.4
        OPTIONS "BENCHMARK_ENABLE_GTEST_TESTS OFF;BENCHMARK_ENABLE_TESTING OFF"
    )
    add_executable(kdtree_benchmark benchmark/kdtree_benchmark.cpp)

    target_link_libraries(
        kdtree_benchmark
        benchmark::benchmark
        clustering_header_lib
    )
    clustering_enable_tidy(kdtree_benchmark)
endif()
