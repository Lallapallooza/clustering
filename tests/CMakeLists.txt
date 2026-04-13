add_executable(
    clustering_tests
    kdtree_test.cpp
    dbscan_test.cpp
    ndarray_test.cpp
    matrix_desc_test.cpp
    math_equality_test.cpp
    math_thread_test.cpp
)
target_link_libraries(
    clustering_tests
    PRIVATE clustering_header_lib clustering_warnings GTest::gtest_main
)
clustering_enable_tidy(clustering_tests)

include(GoogleTest)
gtest_discover_tests(clustering_tests)
