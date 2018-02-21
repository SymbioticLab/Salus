# Version 3.8 required from CheckCXXFeature.cmake
# Version 3.8 required from Boost 1.64.0 for imported target
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

#---------------------------------------------------------------------------------------
# Find packages
#---------------------------------------------------------------------------------------
list(APPEND CMAKE_PREFIX_PATH ${PROJECT_SOURCE_DIR}/spack-packages)
list(REMOVE_DUPLICATES CMAKE_PREFIX_PATH)

# Protobuf
#set(protobuf_MODULE_COMPATIBLE ON CACHE BOOL "CMake build-in FindProtobuf.cmake module compatible" FORCE)
find_package(Protobuf 3.4.0 EXACT REQUIRED)
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
find_package(Boost 1.66 EXACT REQUIRED COMPONENTS
    thread
)
set_package_properties(Boost PROPERTIES TYPE REQUIRED PURPOSE "For lock free queue and some utilities")
add_definitions(-DBOOST_THREAD_VERSION=4)

# Easylogging++
include(thirdparty/UseEasyloggingpp.cmake)

# docopt.cpp
include(thirdparty/UseDocopt.cmake)

#---------------------------------------------------------------------------------------
# Print summary
#---------------------------------------------------------------------------------------
add_feature_info(WITH_TENSORFLOW USE_TENSORFLOW "build TensorFlow operation library")
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
check_cxx_feature(cxx_builtin_expect HAS_CXX_BUILTIN_EXPECT)

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
