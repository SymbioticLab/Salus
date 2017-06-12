cmake_minimum_required(VERSION 3.1.0)

project(executor)

#---------------------------------------------------------------------------------------
# Initial setups
#---------------------------------------------------------------------------------------
# Include utilities
include(cmake/Utilities.cmake)
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
# Setup defaults
include(cmake/Defaults.cmake)
# For feature_summary
include(FeatureSummary)

#---------------------------------------------------------------------------------------
# Available options
#---------------------------------------------------------------------------------------
option(WITH_TENSORFLOW "Build TensorFlow operation library" ON)

option(WITH_TESTS "Build test suite in default target" OFF)

option(WITH_BUNDLED_SPDLOG "Use bundled spdlog library instead of system provided version" OFF)

#---------------------------------------------------------------------------------------
# Find packages
#---------------------------------------------------------------------------------------
# Protobuf
find_package(Protobuf REQUIRED)
set_package_properties(Protobuf PROPERTIES TYPE REQUIRED PURPOSE "For message serialization")

# TensorFlow
# TensorFlow root must be passed in command line as -DTensorFlow_ROOT=/path/to/tensorflow
set(USE_TENSORFLOW OFF)
if(WITH_TENSORFLOW)
    find_package(TensorFlow REQUIRED)
    set(USE_TENSORFLOW ON)
else(WITH_TENSORFLOW)
    find_package(TensorFlow OPTIONAL)
    if(TensorFlow_FOUND)
        set(USE_TENSORFLOW ON)
    endif()
endif(WITH_TENSORFLOW)
set_package_properties(TensorFlow PROPERTIES TYPE RECOMMENDED PURPOSE "For TensorFlow operation library")

# ZeroMQ
find_package(ZeroMQ REQUIRED)
set_package_properties(ZeroMQ PROPERTIES TYPE REQUIRED PURPOSE "For communication")

# Boost
find_package(Boost 1.64 REQUIRED)
set_package_properties(Boost PROPERTIES TYPE REQUIRED PURPOSE "For lock free queue")

# Q
find_package(Q REQUIRED)
set_package_properties(Q PROPERTIES TYPE REQUIRED PURPOSE "For multithreaded execution")

# spdlog
set(USE_BUNDLED_SPDLOG ${WITH_BUNDLED_SPDLOG})
if(NOT WITH_BUNDLED_SPDLOG)
    find_package(spdlog QUIET)
    set_package_properties(spdlog PROPERTIES TYPE OPTIONAL PURPOSE "For logging")
    if(NOT spdlog_FOUND)
        ## fallback to bundled spdlog
        set(USE_BUNDLED_SPDLOG ON)
        message(WARNING "spdlog requested but not found, fallback to use bundled version.")
    endif(NOT spdlog_FOUND)
endif(NOT WITH_BUNDLED_SPDLOG)

if(USE_BUNDLED_SPDLOG)
    if(NOT EXISTS thirdparty/spdlog/CMakeLists.txt)
        message(STATUS "Initialize submodule")
        execute_process(COMMAND "git" "submodule" "update" "--init" "thirdparty/spdlog"
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
            RESULT_VARIABLE retcode
        )
        if(NOT "${retcode}" STREQUAL "0")
            message(FATAL_ERROR "Failed to checkout spdlog as submodule")
        endif(NOT "${retcode}" STREQUAL "0")
    endif(NOT EXISTS thirdparty/spdlog/CMakeLists.txt)
    SET(SPDLOG_BUILD_TESTING OFF CACHE BOOL "Build spdlog tests" FORCE)
    add_subdirectory(thirdparty/spdlog)
    # make a imported target from it to keep consistency with find_package
    add_library(spdlog::spdlog INTERFACE IMPORTED)
    get_target_property(spdlog_inc spdlog INTERFACE_INCLUDE_DIRECTORIES)
    set_target_properties(spdlog::spdlog PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${spdlog_inc}"
    )
endif(USE_BUNDLED_SPDLOG)

#---------------------------------------------------------------------------------------
# Print summary
#---------------------------------------------------------------------------------------
add_feature_info(WITH_TENSORFLOW USE_TENSORFLOW "build TensorFlow operation library")
add_feature_info(WITH_BUNDLED_SPDLOG USE_BUNDLED_SPDLOG "use bundled spdlog library")
add_feature_info(WITH_TESTS WITH_TESTS "build test suite with default target")
feature_summary(INCLUDE_QUIET_PACKAGES FATAL_ON_MISSING_REQUIRED_PACKAGES WHAT ALL)

#---------------------------------------------------------------------------------------
# Put code together
#---------------------------------------------------------------------------------------

# Only set compile options after any inclusion of third party code
include(cmake/CompileOptions.cmake)

add_subdirectory(protos)

add_subdirectory(src)

if(WITH_TESTS)
    add_subdirectory(tests)
else()
    add_subdirectory(tests EXCLUDE_FROM_ALL)
endif()
