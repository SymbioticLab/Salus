cmake_minimum_required(VERSION 3.1.0)

project(executor)

#---------------------------------------------------------------------------------------
# Initial setups
#---------------------------------------------------------------------------------------
# Include utilities
include(cmake/Utilities.cmake)
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
# Setup defaults
include(cmake/DefaultFlags.cmake)
# For feature_summary
include(FeatureSummary)

#---------------------------------------------------------------------------------------
# Available options
#---------------------------------------------------------------------------------------
option(WITH_TESTS "Build test suite in default target" OFF)

option(WITH_BUNDLED_SPDLOG "Use bundled spdlog library instead of system provided version" OFF)

#---------------------------------------------------------------------------------------
# Find packages
#---------------------------------------------------------------------------------------
# Protobuf
find_package(Protobuf REQUIRED)


# TensorFlow
# TensorFlow root must be passed in command line as -DTensorFlow_ROOT=/path/to/tensorflow
find_package(TensorFlow REQUIRED)

# ZeroMQ or gRPC
find_package(ZeroMQ REQUIRED)
# gRPC
if(NOT ZeroMQ_FOUND)
    find_package(GRPC REQUIRED)
endif()

# spdlog
if(NOT WITH_BUNDLED_SPDLOG)
    find_package(spdlog QUIET)
endif(NOT WITH_BUNDLED_SPDLOG)

## fallback to bundled spdlog
if(NOT TARGET spdlog::spdlog)
    SET(WITH_BUNDLED_SPDLOG ON CACHE BOOL "Use bundled spdlog library instead of system provided version" FORCE)
    message(WARNING "spdlog requested but not found, fallback to use bundled version.")

    SET(SPDLOG_BUILD_TESTING OFF CACHE BOOL "Build spdlog tests" FORCE)
    add_subdirectory(thirdparty/spdlog)
    add_library(spdlog::spdlog ALIAS spdlog)
endif(NOT TARGET spdlog::spdlog)

#---------------------------------------------------------------------------------------
# Print summary
#---------------------------------------------------------------------------------------
add_feature_info(TESTS WITH_TESTS "build test suite with default target")
add_feature_info("BUNDLED_SPDLOG" WITH_BUNDLED_SPDLOG
                 "use bundled spdlog library instead of system provided version")
feature_summary(INCLUDE_QUIET_PACKAGES FATAL_ON_MISSING_REQUIRED_PACKAGES WHAT ALL)

#---------------------------------------------------------------------------------------
# Put code together
#---------------------------------------------------------------------------------------
add_subdirectory(protos)

add_subdirectory(src)

if(WITH_TESTS)
    add_subdirectory(tests)
else()
    add_subdirectory(tests EXCLUDE_FROM_ALL)
endif()
