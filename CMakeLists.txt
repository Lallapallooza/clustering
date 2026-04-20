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
            $<$<CXX_COMPILER_ID:GNU>:-Wno-ignored-attributes>
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

    add_executable(gemm_benchmark benchmark/gemm_benchmark.cpp)
    target_link_libraries(
        gemm_benchmark
        benchmark::benchmark
        clustering_header_lib
        clustering_warnings
    )

    add_executable(kmeans_benchmark benchmark/kmeans_benchmark.cpp)
    target_link_libraries(
        kmeans_benchmark
        benchmark::benchmark
        clustering_header_lib
        clustering_warnings
    )
    clustering_enable_tidy(kmeans_benchmark)

    add_executable(dbscan_benchmark benchmark/dbscan_benchmark.cpp)
    target_link_libraries(
        dbscan_benchmark
        benchmark::benchmark
        clustering_header_lib
        clustering_warnings
    )
    clustering_enable_tidy(dbscan_benchmark)

    # OpenBLAS is optional: when available the benchmark adds a comparative
    # cblas_sgemm fixture next to ours. pkg-config reliably reports the include
    # dir on every distro that ships openblas.pc (Nix, Debian, Arch, Homebrew),
    # avoiding the install-tree probe that find_package(BLAS) skips.
    find_package(PkgConfig QUIET)
    set(_clustering_openblas_target "")
    if(PkgConfig_FOUND)
        # NixOS's openblas package exports cblas.pc (not openblas.pc); Debian /
        # Homebrew / Arch all do the same. Try cblas first, fall back to
        # openblas for the rare distro that uses that name.
        pkg_check_modules(CLUSTERING_OPENBLAS QUIET IMPORTED_TARGET cblas)
        if(NOT CLUSTERING_OPENBLAS_FOUND)
            pkg_check_modules(
                CLUSTERING_OPENBLAS
                QUIET
                IMPORTED_TARGET
                openblas
            )
        endif()
        if(CLUSTERING_OPENBLAS_FOUND)
            set(_clustering_openblas_target PkgConfig::CLUSTERING_OPENBLAS)
        endif()
    endif()

    if(NOT _clustering_openblas_target)
        # Fallback for systems without pkg-config: find_package(BLAS) locates the
        # library, then probe common cblas.h locations.
        find_package(BLAS QUIET)
        if(BLAS_FOUND)
            find_path(
                CLUSTERING_CBLAS_INCLUDE_DIR
                cblas.h
                PATHS
                    /usr/include
                    /usr/include/openblas
                    /usr/local/include
                    $ENV{OPENBLAS_INCLUDE_DIR}
            )
            if(CLUSTERING_CBLAS_INCLUDE_DIR)
                add_library(_clustering_blas_fallback INTERFACE)
                target_link_libraries(
                    _clustering_blas_fallback
                    INTERFACE ${BLAS_LIBRARIES}
                )
                target_include_directories(
                    _clustering_blas_fallback
                    INTERFACE ${CLUSTERING_CBLAS_INCLUDE_DIR}
                )
                set(_clustering_openblas_target _clustering_blas_fallback)
            endif()
        endif()
    endif()

    if(_clustering_openblas_target)
        message(STATUS "OpenBLAS found; gemm benchmark will compare against it")
        target_compile_definitions(
            gemm_benchmark
            PRIVATE CLUSTERING_BENCHMARK_OPENBLAS
        )
        target_link_libraries(gemm_benchmark ${_clustering_openblas_target})
    else()
        message(
            STATUS
            "OpenBLAS not found; gemm benchmark will run our path only"
        )
    endif()

    clustering_enable_tidy(gemm_benchmark)
endif()

if(CLUSTERING_BUILD_TESTS)
    include(cmake/GTestDep.cmake)
    enable_testing()
    add_subdirectory(tests)
endif()
