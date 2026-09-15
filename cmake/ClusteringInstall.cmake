# Install rules for the headers and the CMake package config. After the
# install, consumers call `find_package(clustering X.Y.Z REQUIRED)` and link
# `clustering::clustering`.

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(TARGETS clustering_header_lib EXPORT clusteringTargets)
install(
    EXPORT clusteringTargets
    FILE clusteringTargets.cmake
    NAMESPACE clustering::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/clustering
)

configure_package_config_file(
    ${PROJECT_SOURCE_DIR}/cmake/clusteringConfig.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/clustering-config.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/clustering
)
# Before 1.0, a minor release can break the API.
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/clustering-config-version.cmake"
    COMPATIBILITY SameMinorVersion
    ARCH_INDEPENDENT
)
install(
    FILES
        "${CMAKE_CURRENT_BINARY_DIR}/clustering-config.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/clustering-config-version.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/clustering
)
