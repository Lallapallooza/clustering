# CPM consumption of the standalone citor thread pool. Citor is header-only;
# the INTERFACE target it exposes is `citor::citor`. Pinned to a tagged
# upstream release so configures are reproducible across machines. The Conan
# recipe under packaging/conan requires the same citor version.
cpmaddpackage(
    NAME citor
    GITHUB_REPOSITORY Lallapallooza/citor
    GIT_TAG v0.6.1
)
