cpmaddpackage(
    NAME googletest
    GITHUB_REPOSITORY google/googletest
    GIT_TAG v1.15.2
    SYSTEM YES
    OPTIONS "INSTALL_GTEST OFF" "gtest_force_shared_crt ON"
)
