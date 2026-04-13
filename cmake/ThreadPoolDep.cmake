cpmaddpackage(
    NAME bshoshany_thread_pool
    GITHUB_REPOSITORY bshoshany/thread-pool
    GIT_TAG v5.1.0
    DOWNLOAD_ONLY YES
)

if(bshoshany_thread_pool_ADDED AND NOT TARGET bshoshany_thread_pool)
    add_library(bshoshany_thread_pool INTERFACE)
    target_include_directories(
        bshoshany_thread_pool
        SYSTEM
        INTERFACE ${bshoshany_thread_pool_SOURCE_DIR}/include
    )
endif()
