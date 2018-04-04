# Ensure out-of-source build
if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
    message(FATAL_ERROR "Usage: mkdir build; cmake ..")
endif()

# Default build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug")
endif()

# Default third party package path
if(NOT CMAKE_PREFIX_PATH)
    set(CMAKE_PREFIX_PATH spack-packages)
endif()
get_filename_component(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH}" ABSOLUTE BASE_DIR ${PROJECT_SOURCE_DIR})

# Use C++14 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
