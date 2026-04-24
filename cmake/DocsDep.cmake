include_guard(GLOBAL)

# Pin to a specific release tag so the theme cannot drift between machines.
cpmaddpackage(
    NAME doxygen-awesome-css
    GITHUB_REPOSITORY jothepro/doxygen-awesome-css
    GIT_TAG v2.4.2
    DOWNLOAD_ONLY YES
)
