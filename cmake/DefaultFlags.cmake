# Ensure out-of-source build
if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
    message(FATAL_ERROR "Usage: mkdir build; cmake ..")
endif()

# Default build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug")
endif()

# Use C++14 standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set compiler flags
set(CLOSS_COMPILER_FLAGS)
if(${CMAKE_BUILD_TYPE} STREQUAL Debug)
  message(STATUS "Debug configuration")
  if(CMAKE_COMPILER_IS_GNUCXX)
    compiler_check_add_flag("-ggdb")
    compiler_check_add_flag("-fno-default-inline")
  endif()
elseif(${CMAKE_BUILD_TYPE} STREQUAL Release)
  message(STATUS "Release configuration")
  if(CMAKE_COMPILER_IS_GNUCXX)
    compiler_check_add_flag("-g0")
    compiler_check_add_flag("-s")
    compiler_check_add_flag("-O3")
    compiler_check_add_flag("-msse")
    compiler_check_add_flag("-msse2")
    compiler_check_add_flag("-msse3")
    compiler_check_add_flag("-mssse3")
    compiler_check_add_flag("-msse4.1")
    compiler_check_add_flag("-msse4.2")
  endif()
else()
  message(FATAL_ERROR "Unknown configuration, set CMAKE_BUILD_TYPE to Debug or Release")
endif()
if(CMAKE_COMPILER_IS_GNUCXX)
  set(COMPILER_WARNING_FLAGS "-Wall -Wextra -pedantic -Wno-long-long -Wno-enum-compare")
endif()


