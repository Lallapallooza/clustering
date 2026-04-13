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

option(CLUSTERING_BUILD_BENCHMARK "Build benchmarks" ${CLUSTERING_IS_TOP_LEVEL})
option(CLUSTERING_BUILD_TESTS "Build unit tests" ${CLUSTERING_IS_TOP_LEVEL})
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

# Top-level-only: enforce -Wall -Wextra -Werror on our executables.
# Consumers that pull us in as a subdirectory do not inherit these flags.
add_library(clustering_warnings INTERFACE)
if(CLUSTERING_IS_TOP_LEVEL)
    target_compile_options(
        clustering_warnings
        INTERFACE
            $<$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>>:-Wall;-Wextra;-Werror>
    )
endif()

add_executable(clustering_demo app/main.cpp)
target_link_libraries(
    clustering_demo
    PRIVATE clustering_header_lib clustering_warnings
)
clustering_enable_tidy(clustering_demo)

if(CLUSTERING_USE_AVX2)
    if(NOT CLUSTERING_COMPILER_SUPPORTS_AVX2)
        message(FATAL_ERROR "AVX2 not supported but requested")
    endif()
    target_compile_options(clustering_header_lib INTERFACE -mavx2 -mfma)
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
        SYSTEM YES
        OPTIONS "BENCHMARK_ENABLE_GTEST_TESTS OFF;BENCHMARK_ENABLE_TESTING OFF"
    )
    add_executable(kdtree_benchmark benchmark/kdtree_benchmark.cpp)

    target_link_libraries(
        kdtree_benchmark
        benchmark::benchmark
        clustering_header_lib
        clustering_warnings
    )
    clustering_enable_tidy(kdtree_benchmark)
endif()

if(CLUSTERING_BUILD_TESTS)
    include(cmake/GTestDep.cmake)
    enable_testing()
    add_subdirectory(tests)
endif()
