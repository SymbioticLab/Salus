# Version 3.8 required from CheckCXXFeature.cmake
# Version 3.10 required from Boost 1.66.0 for imported target
cmake_minimum_required(VERSION 3.10.0)

project(executor VERSION 1.0.0 LANGUAGES C CXX)

#---------------------------------------------------------------------------------------
# Initial setups
#---------------------------------------------------------------------------------------
# Include utilities
include(cmake/Utilities.cmake)
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
# For feature_summary
include(FeatureSummary)

# Setup defaults
include(cmake/Defaults.cmake)
# Extra build types
include(cmake/BuildTypes.cmake)

#---------------------------------------------------------------------------------------
# Available options
#---------------------------------------------------------------------------------------
option(WITH_TENSORFLOW "Build TensorFlow operation library" ON)

option(WITH_TESTS "Build test suite in default target" OFF)

option(WITH_TCMALLOC "Build with tcmalloc" ON)

option(WITH_TF_REFINER "Enable ShapeRefiner in TF oplibrary" OFF)

option(WITH_PARALLEL_SCHED "Enable parallel processing in scheduler" OFF)

option(WITH_MULTI_DEVICE "Enable multi-device scheduling support" OFF)

option(DISABLE_LOGGING "Disable all logging except INFO level" OFF)

option(WITH_STATIC_STREAM "Use static GPU stream assignment, for debug only" ON)

option(WITH_EXCLUSIVE_ITER "Each iteration runs exclusively" ON)

option(WITH_TIMEOUT_WARNING "Enable timeout warning. Note that the logging function should be enabled seperately" OFF)

#---------------------------------------------------------------------------------------
# Find packages
#---------------------------------------------------------------------------------------
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

if(WITH_TCMALLOC)
    find_package(Gperftools REQUIRED)
    set_package_properties(Gperftools PROPERTIES TYPE REQUIRED PURPOSE "For gperftools cpu profiler and/or tcmalloc")
endif(WITH_TCMALLOC)

# Bundled third party library
add_subdirectory(thirdparty)
#---------------------------------------------------------------------------------------
# Print summary
#---------------------------------------------------------------------------------------
add_feature_info(WITH_TENSORFLOW USE_TENSORFLOW "build TensorFlow operation library")
add_feature_info(WITH_TESTS WITH_TESTS "build test suite with default target")
add_feature_info(WITH_TCMALLOC WITH_TCMALLOC "build with tcmalloc")
add_feature_info(WITH_TF_REFINER WITH_TF_REFINER "enable ShapeRefiner in TF oplibrary")
add_feature_info(WITH_PARALLEL_SCHED WITH_WITH_PARALLEL_SCHED "enable parallel processing in scheduler")
add_feature_info(WITH_MULTI_DEVICE WITH_MULTI_DEVICE "enable multi-device scheduling support")
add_feature_info(DISABLE_LOGGING DISABLE_LOGGING "disable all logging except INFO level")
add_feature_info(WITH_STATIC_STREAM WITH_STATIC_STREAM "use static GPU stream assignment, for debug only")
add_feature_info(WITH_EXCLUSIVE_ITER WITH_EXCLUSIVE_ITER "Each iteration runs exclusively")
add_feature_info(WITH_TIMEOUT_WARNING WITH_TIMEOUT_WARNING "Enable timeout warning")
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

# Set feature macros
if(WITH_TF_REFINER)
    set(SALUS_ENABLE_REFINER 1)
endif(WITH_TF_REFINER)

if(WITH_PARALLEL_SCHED)
    set(SALUS_ENABLE_PARALLEL_SCHED 1)
endif(WITH_PARALLEL_SCHED)

if(WITH_MULTI_DEVICE)
    set(SALUS_ENABLE_MULTI_DEVICE 1)
endif(WITH_MULTI_DEVICE)

if(WITH_STATIC_STREAM)
    set(SALUS_ENABLE_STATIC_STREAM 1)
endif(WITH_STATIC_STREAM)

if(WITH_EXCLUSIVE_ITER)
    set(SALUS_ENABLE_EXCLUSIVE_ITER 1)
endif(WITH_EXCLUSIVE_ITER)

if(WITH_TIMEOUT_WARNING)
    set(SALUS_ENABLE_TIMEOUT_WARNING 1)
endif(WITH_TIMEOUT_WARNING)

configure_file(src/config.h.in ${CMAKE_CURRENT_BINARY_DIR}/config.h)
include_directories(${CMAKE_CURRENT_BINARY_DIR})

# Disable verbose logging in non debug builds
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    #add_definitions(-DELPP_DISABLE_DEBUG_LOGS)
    #add_definitions(-DELPP_DISABLE_TRACE_LOGS)
    #add_definitions(-DELPP_DISABLE_VERBOSE_LOGS)
elseif(CMAKE_BUILD_TYPE STREQUAL "OpTracing")
    add_definitions(-DELPP_DISABLE_DEBUG_LOGS)
    #add_definitions(-DELPP_DISABLE_TRACE_LOGS)
    add_definitions(-DELPP_DISABLE_VERBOSE_LOGS)
elseif(CMAKE_BUILD_TYPE MATCHES "San$")
    #add_definitions(-DELPP_DISABLE_DEBUG_LOGS)
    add_definitions(-DELPP_DISABLE_TRACE_LOGS)
    #add_definitions(-DELPP_DISABLE_VERBOSE_LOGS)
else()
    add_definitions(-DELPP_DISABLE_DEBUG_LOGS)
    add_definitions(-DELPP_DISABLE_TRACE_LOGS)
    add_definitions(-DELPP_DISABLE_VERBOSE_LOGS)
endif()

if(DISABLE_LOGGING)
    add_definitions(-DELPP_DISABLE_DEBUG_LOGS)
    add_definitions(-DELPP_DISABLE_INFO_LOGS)
    #add_definitions(-DELPP_DISABLE_WARNING_LOGS)
    #add_definitions(-DELPP_DISABLE_ERROR_LOGS)
    #add_definitions(-DELPP_DISABLE_FATAL_LOGS)
    add_definitions(-DELPP_DISABLE_TRACE_LOGS)
    add_definitions(-DELPP_DISABLE_VERBOSE_LOGS)
endif()

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
