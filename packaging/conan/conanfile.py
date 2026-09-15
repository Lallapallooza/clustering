# Conan 2.x recipe for clustering. The library is header-only, so the package
# step copies only the headers and the license.
#
# Create the citor package first, then this one:
#   conan create <citor checkout>/packaging/conan
#   conan create packaging/conan -s compiler.cppstd=20
#   conan install --requires=clustering/<version> -s compiler.cppstd=20

from pathlib import Path

from conan import ConanFile
from conan.tools.build import check_min_cppstd
from conan.tools.files import copy
from conan.tools.layout import basic_layout


class ClusteringConan(ConanFile):
    name = "clustering"
    version = "0.10.1"
    license = "MIT"
    homepage = "https://github.com/Lallapallooza/clustering"
    url = "https://github.com/Lallapallooza/clustering"
    description = "Header-only C++20 clustering library: DBSCAN, HDBSCAN*, k-means."
    topics = ("clustering", "dbscan", "hdbscan", "kmeans", "header-only")

    settings = "os", "arch", "compiler", "build_type"
    package_type = "header-library"
    no_copy_source = True
    options = {"with_avx2": [True, False]}
    default_options = {"with_avx2": True}

    # This recipe is two levels below the repository root. `exports_sources`
    # cannot copy files from `../../include`, so this method copies them.
    def export_sources(self):
        src = Path(self.recipe_folder).resolve().parent.parent
        copy(
            self,
            "*.h",
            str(src / "include"),
            str(Path(self.export_sources_folder) / "include"),
        )
        copy(self, "LICENSE", str(src), str(self.export_sources_folder))

    def config_options(self):
        if str(self.settings.arch) not in ("x86", "x86_64"):
            del self.options.with_avx2

    def requirements(self):
        self.requires("citor/0.6.1", transitive_headers=True)

    def validate(self):
        check_min_cppstd(self, 20)

    def package_id(self):
        # Options and settings change only the consumer flags. The packaged
        # files are the same for all of them.
        self.info.clear()

    def layout(self):
        basic_layout(self, src_folder=".")

    def package(self):
        src = Path(self.source_folder)
        pkg = Path(self.package_folder)
        copy(self, "*.h", str(src / "include"), str(pkg / "include"))
        copy(self, "LICENSE", str(src), str(pkg / "licenses"))

    def package_info(self):
        self.cpp_info.set_property("cmake_file_name", "clustering")
        self.cpp_info.set_property("cmake_target_name", "clustering::clustering")
        self.cpp_info.bindirs = []
        self.cpp_info.libdirs = []
        # The CMake target adds these flags when CLUSTERING_USE_AVX2 is ON.
        # MSVC uses its own AVX2 flag.
        if self.options.get_safe("with_avx2"):
            self.cpp_info.defines = ["CLUSTERING_USE_AVX2"]
            if str(self.settings.compiler) == "msvc":
                self.cpp_info.cxxflags = ["/arch:AVX2"]
            else:
                self.cpp_info.cxxflags = ["-mavx2", "-mfma"]
