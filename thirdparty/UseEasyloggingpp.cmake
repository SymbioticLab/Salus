# Library name, also the path
set(library_name easyloggingpp)

message("-- ${library_name} version: bundled")

# Set any options
set(build_static_lib ON CACHE BOOL "Build static lib" FORCE)
set(lib_utc_datetime ON CACHE BOOL "Use UTC datetime" FORCE)

# Fix policy warnings
cmake_policy(SET CMP0048 OLD)

add_submodule(${library_name}
    PATCHES
    vmodule-respect-vlevel-elsewhere.patch
    more-color-term.patch
    disable-warning.patch
    )
# Macros to enable extra features in Easylogging++
set(macros
    ELPP_DEBUG_ASSERT_FAILURE
    ELPP_THREAD_SAFE
    ELPP_UTC_DATETIME
    ELPP_STL_LOGGING
    ELPP_FEATURE_PERFORMANCE_TRACKING
    ELPP_PERFORMANCE_MICROSECONDS
    ELPP_NO_DEFAULT_LOG_FILE
    ELPP_DISABLE_LOGGING_FLAGS_FROM_ARG
    ELPP_DISABLE_LOG_FILE_FROM_ARG
    )

# Use CMAKE_CURRENT_LIST_DIR as this file will be included
set(third_party_dir ${CMAKE_CURRENT_LIST_DIR})

# Make the target suitable for directly use
set_property(TARGET easyloggingpp APPEND PROPERTY INCLUDE_DIRECTORIES "${third_party_dir}/${library_name}/src")
set_property(TARGET easyloggingpp APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${third_party_dir}/${library_name}/src")
set_property(TARGET easyloggingpp APPEND PROPERTY COMPILE_DEFINITIONS ${macros})
set_property(TARGET easyloggingpp APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS ${macros})
# Since the target is defined inside a nested project() call, global CMAKE_POSITION_INDENPENDENT_CODE has no effect on it.
# re-enable it here.
set_property(TARGET easyloggingpp PROPERTY POSITION_INDEPENDENT_CODE ON)
