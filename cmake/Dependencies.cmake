list(APPEND CMAKE_PREFIX_PATH ${PROJECT_SOURCE_DIR}/spack-packages)
list(REMOVE_DUPLICATES CMAKE_PREFIX_PATH)

# Prefer pthreads
set(CMAKE_THREAD_PREFER_PTHREAD ON)
find_package(Threads REQUIRED)

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

# Bundled third party library
add_subdirectory(${PROJECT_SOURCE_DIR}/thirdparty)
