# Version 3.8 required from CheckCXXFeature.cmake
# Version 3.10 required from Boost 1.66.0 for imported target
cmake_minimum_required(VERSION 3.10.0 FATAL_ERROR)

#---------------------------------------------------------------------------------------
# Setup project with version
#---------------------------------------------------------------------------------------
project(executor VERSION 2.0.0)

#---------------------------------------------------------------------------------------
# Initial setups
#---------------------------------------------------------------------------------------
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
# Include utilities
include(Utilities)
# Setup defaults
include(SetDefaults)
# For feature summary
include(FeatureSummary)

#---------------------------------------------------------------------------------------
# Available options
#---------------------------------------------------------------------------------------
option(WITH_TENSORFLOW "Build TensorFlow operation library" ON)

option(WITH_TESTS "Build test suite in default target" OFF)

#---------------------------------------------------------------------------------------
# Find dependencies
#---------------------------------------------------------------------------------------
include(Dependencies)

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

configure_file(src/saluslib/include/salus/config.h.in ${CMAKE_CURRENT_BINARY_DIR}/config.h)
include_directories(${CMAKE_CURRENT_BINARY_DIR})

#---------------------------------------------------------------------------------------
# Put code together
#---------------------------------------------------------------------------------------
add_subdirectory(src/saluslib)
add_subdirectory(src/salusbin)
add_subdirectory(src/frameworks)

if(WITH_TESTS)
    add_subdirectory(tests EXCLUDE_FROM_ALL)
endif()
