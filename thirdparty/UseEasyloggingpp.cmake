
# Use CMAKE_CURRENT_LIST_DIR as this file will be included
set(third_party_dir ${CMAKE_CURRENT_LIST_DIR})

# Library name, also the path
set(library_name easyloggingpp)

message("-- ${library_name} version: bundled")

if(NOT EXISTS ${third_party_dir}/${library_name}/CMakeLists.txt)
    message(STATUS "--   Initialize submodule")
    execute_process(COMMAND "git" "submodule" "update" "--init" "${library_name}"
        WORKING_DIRECTORY "${third_party_dir}"
        RESULT_VARIABLE retcode
    )
    if(NOT "${retcode}" STREQUAL "0")
        message(FATAL_ERROR "Failed to checkout ${library_name} as submodule")
    endif(NOT "${retcode}" STREQUAL "0")
    execute_process(COMMAND "git" "apply" "../vmodule-respect-vlevel-elsewhere.patch"
        WORKING_DIRECTORY "${third_party_dir}/${library_name}"
        RESULT_VARIABLE retcode
    )
    if(NOT "${retcode}" STREQUAL "0")
        message(FATAL_ERROR "Failed to patch ${library_name} after submodule initialization")
    endif(NOT "${retcode}" STREQUAL "0")
    execute_process(COMMAND "git" "apply" "../more-color-term.patch"
        WORKING_DIRECTORY "${third_party_dir}/${library_name}"
        RESULT_VARIABLE retcode
    )
    if(NOT "${retcode}" STREQUAL "0")
        message(FATAL_ERROR "Failed to patch ${library_name} after submodule initialization")
    endif(NOT "${retcode}" STREQUAL "0")
endif(NOT EXISTS ${third_party_dir}/${library_name}/CMakeLists.txt)

# Set any options
set(build_static_lib ON CACHE BOOL "Build static lib" FORCE)
set(lib_utc_datetime ON CACHE BOOL "Use UTC datetime" FORCE)
add_subdirectory(${third_party_dir}/${library_name})

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
# Disable extensive logging in release
if(${CMAKE_BUILD_TYPE} STREQUAL Release OR ${CMAKE_BUILD_TYPE} STREQUAL RelWithDebugInfo)
    list(APPEND macros ELPP_DISABLE_VERBOSE_LOGS)
    list(APPEND macros ELPP_DISABLE_PERFORMANCE_TRACKING)
endif()


# Make the target suitable for directly use
set_property(TARGET easyloggingpp APPEND PROPERTY INCLUDE_DIRECTORIES "${third_party_dir}/${library_name}/src")
set_property(TARGET easyloggingpp APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${third_party_dir}/${library_name}/src")
set_property(TARGET easyloggingpp APPEND PROPERTY COMPILE_DEFINITIONS ${macros})
set_property(TARGET easyloggingpp APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS ${macros})
