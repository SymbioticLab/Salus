# Version 3.8 required from CheckCXXFeature.cmake
# Version 3.5 required from Boost for imported target
cmake_minimum_required(VERSION 3.8.0)

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

option(WITH_BUNDLED_DOCOPT "Use bundled docopt.cpp library instead of system provided version" OFF)

#---------------------------------------------------------------------------------------
# Find packages
#---------------------------------------------------------------------------------------
# Protobuf
find_package(Protobuf 3.2.0 EXACT REQUIRED)
set_package_properties(Protobuf PROPERTIES TYPE REQUIRED PURPOSE
    "For message serialization, version must match the one used in TensorFlow"
)

# TensorFlow
# TensorFlow root must be passed in command line as -DTENSORFLOW_ROOT=/path/to/tensorflow
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
find_package(Boost 1.64 REQUIRED COMPONENTS
    thread
)
set_package_properties(Boost PROPERTIES TYPE REQUIRED PURPOSE "For lock free queue and some utilities")
add_definitions(-DBOOST_THREAD_VERSION=4)

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
    include(thirdparty/UseSpdlog.cmake)
endif(USE_BUNDLED_SPDLOG)

# docopt.cpp
set(USE_BUNDLED_DOCOPT ${WITH_BUNDLED_DOCOPT})
if(NOT WITH_BUNDLED_DOCOPT)
    find_package(docopt QUIET)
    set_package_properties(docopt PROPERTIES TYPE OPTIONAL PURPOSE "For command line interface")
    if(NOT docopt_FOUND)
        ## fallback to bundled spdlog
        set(USE_BUNDLED_DOCOPT ON)
        message(WARNING "docopt.cpp requested but not found, fallback to use bundled version.")
    endif(NOT docopt_FOUND)
endif(NOT WITH_BUNDLED_DOCOPT)
if(USE_BUNDLED_DOCOPT)
    include(thirdparty/UseDocopt.cmake)
endif(USE_BUNDLED_DOCOPT)

#---------------------------------------------------------------------------------------
# Print summary
#---------------------------------------------------------------------------------------
add_feature_info(WITH_TENSORFLOW USE_TENSORFLOW "build TensorFlow operation library")
add_feature_info(WITH_BUNDLED_SPDLOG USE_BUNDLED_SPDLOG "use bundled spdlog library")
add_feature_info(WITH_BUNDLED_DOCOPT USE_BUNDLED_DOCOPT "use bundled docopt.cpp library")
add_feature_info(WITH_TESTS WITH_TESTS "build test suite with default target")
feature_summary(INCLUDE_QUIET_PACKAGES FATAL_ON_MISSING_REQUIRED_PACKAGES WHAT ALL)

#---------------------------------------------------------------------------------------
# Set compiler options and detect features
#---------------------------------------------------------------------------------------

# Only set compile options after any inclusion of third party code
include(cmake/CompileOptions.cmake)

# Detect features
include(cmake/CheckCXXFeature.cmake)
check_cxx_feature(cxx_enum_hash HAS_CXX_ENUM_HASH)
check_cxx_feature(cxx_aligned_alloc HAS_CXX_ALIGNED_ALLOC)

configure_file(src/config.h.in ${CMAKE_CURRENT_BINARY_DIR}/config.h)
include_directories(${CMAKE_CURRENT_BINARY_DIR})

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
