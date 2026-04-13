set(CLUSTERING_IS_TOP_LEVEL OFF)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(CLUSTERING_IS_TOP_LEVEL ON)
endif()

option(
    CLUSTERING_ENABLE_CLANG_TIDY
    "Run clang-tidy during build"
    ${CLUSTERING_IS_TOP_LEVEL}
)

if(CLUSTERING_ENABLE_CLANG_TIDY)
    find_program(
        CLANG_TIDY_EXE
        NAMES clang-tidy clang-tidy-21 clang-tidy-20 clang-tidy-19 clang-tidy-18
    )
    if(CLANG_TIDY_EXE)
        message(STATUS "clang-tidy enabled: ${CLANG_TIDY_EXE}")
    else()
        message(
            WARNING
            "CLUSTERING_ENABLE_CLANG_TIDY=ON but clang-tidy not found; skipping"
        )
    endif()
endif()

function(clustering_enable_tidy target)
    if(CLUSTERING_ENABLE_CLANG_TIDY AND CLANG_TIDY_EXE)
        set_target_properties(
            ${target}
            PROPERTIES CXX_CLANG_TIDY "${CLANG_TIDY_EXE}"
        )
    endif()
endfunction()

if(CLUSTERING_IS_TOP_LEVEL)
    find_program(CLANG_FORMAT_EXE NAMES clang-format)
    if(CLANG_FORMAT_EXE)
        file(
            GLOB_RECURSE CLUSTERING_FORMAT_SOURCES
            CONFIGURE_DEPENDS
            "${CMAKE_SOURCE_DIR}/include/clustering/*.h"
            "${CMAKE_SOURCE_DIR}/app/*.cpp"
            "${CMAKE_SOURCE_DIR}/benchmark/*.cpp"
            "${CMAKE_SOURCE_DIR}/python/src/*.cpp"
        )

        add_custom_target(
            format
            COMMAND ${CLANG_FORMAT_EXE} -i ${CLUSTERING_FORMAT_SOURCES}
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            COMMENT "Running clang-format on project sources"
        )

        add_custom_target(
            check-format
            COMMAND
                ${CLANG_FORMAT_EXE} --dry-run --Werror
                ${CLUSTERING_FORMAT_SOURCES}
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            COMMENT "Checking clang-format compliance"
        )
    endif()
endif()
