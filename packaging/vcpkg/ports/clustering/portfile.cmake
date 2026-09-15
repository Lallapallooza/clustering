# Overlay port for clustering. Pass the citor overlay port as well:
#   vcpkg install clustering \
#     --overlay-ports=path/to/clustering/packaging/vcpkg/ports \
#     --overlay-ports=path/to/citor/packaging/vcpkg/ports
#
# The port builds the clustering checkout that contains it, so a release
# needs no new commit SHA or tarball hash in this file. The vcpkg binary cache
# does not hash this source tree and can serve stale headers. For an untagged
# checkout, pass `--binarysource=clear`.

get_filename_component(
    SOURCE_PATH
    "${CMAKE_CURRENT_LIST_DIR}/../../../.."
    ABSOLUTE
)

set(VCPKG_BUILD_TYPE release)

# CPM_LOCAL_PACKAGES_ONLY makes CPM use the installed citor port. Without it,
# CPM downloads and installs a second copy of citor.
vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
    OPTIONS
        -DCLUSTERING_BUILD_TESTS=OFF
        -DCLUSTERING_BUILD_BENCHMARK=OFF
        -DCLUSTERING_BUILD_DEMO=OFF
        -DCLUSTERING_ENABLE_CLANG_TIDY=OFF
        -DCLUSTERING_INSTALL=ON
        -DCPM_LOCAL_PACKAGES_ONLY=ON
)

vcpkg_cmake_install()
vcpkg_cmake_config_fixup(PACKAGE_NAME clustering CONFIG_PATH lib/cmake/clustering)

file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/lib")
vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE")
