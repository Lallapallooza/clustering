cmake_minimum_required(VERSION 3.22)
project(clustering_python)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(nanobind CONFIG REQUIRED)

include(CheckCXXCompilerFlag)
include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/CPM.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/ThreadPoolDep.cmake)
check_cxx_compiler_flag("-mavx2" CLUSTERING_COMPILER_SUPPORTS_AVX2)

nanobind_add_module(_clustering src/_bindings.cpp)

target_include_directories(
    _clustering
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include
)
target_link_libraries(_clustering PRIVATE bshoshany_thread_pool)
target_compile_features(_clustering PRIVATE cxx_std_20)

if(CLUSTERING_COMPILER_SUPPORTS_AVX2)
    target_compile_options(_clustering PRIVATE -mavx2)
    target_compile_definitions(_clustering PRIVATE CLUSTERING_USE_AVX2)
endif()

install(TARGETS _clustering LIBRARY DESTINATION .)
