cmake_minimum_required(VERSION 3.22)
project(clustering_python)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(nanobind CONFIG REQUIRED)

include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/CPM.cmake)

CPMAddPackage(
    NAME tsl
    GITHUB_REPOSITORY Tessil/hopscotch-map
    GIT_TAG v2.3.1
)

nanobind_add_module(_clustering src/_bindings.cpp)

target_include_directories(_clustering PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(_clustering PRIVATE tsl::hopscotch_map)
target_compile_features(_clustering PRIVATE cxx_std_20)

install(TARGETS _clustering LIBRARY DESTINATION .)
