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
add_feature_info(Tests WITH_TESTS "build test suite with default target")

#---------------------------------------------------------------------------------------
# Find packages
#---------------------------------------------------------------------------------------
# Protobuf
find_package(Protobuf REQUIRED)

# gRPC
find_package(GRPC REQUIRED)

# TensorFlow
# TensorFlow root must be passed in command line as -DTensorFlow_ROOT=/path/to/tensorflow
find_package(TensorFlow REQUIRED)

#---------------------------------------------------------------------------------------
# Print summary
#---------------------------------------------------------------------------------------
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
