# Add new build types for sanitizer and profiler
# Address sanitizer and undefined beheavior sanitizer
set(CMAKE_CXX_FLAGS_ASAN "-DNDEBUG -g -O1 -fno-omit-frame-pointer -fsanitize=address -fsanitize=undefined"
    CACHE STRING "Flags used by the C++ compiler during ASan builds." FORCE)

set(CMAKE_C_FLAGS_ASAN "${CMAKE_CXX_FLAGS_ASAN}"
    CACHE STRING "Flags used by the C compiler during ASan builds." FORCE)

set(CMAKE_EXE_LINKER_FLAGS_ASAN "-fsanitize=address -fsanitize=undefined"
    CACHE STRING "Flags used for linking binaries during ASan builds." FORCE)

set(CMAKE_SHARED_LINKER_FLAGS_ASAN "-fsanitize=address -fsanitize=undefined"
    CACHE STRING "Flags used by the shared libraries linker during ASan builds." FORCE)

# Thread sanitizer
set(CMAKE_CXX_FLAGS_TSAN "-DNDEBUG -g -O0 -fno-omit-frame-pointer -fsanitize=thread"
    CACHE STRING "Flags used by the C++ compiler during TSan builds." FORCE)

set(CMAKE_C_FLAGS_TSAN "${CMAKE_CXX_FLAGS_TSAN}"
    CACHE STRING "Flags used by the C compiler during TSan builds." FORCE)

set(CMAKE_EXE_LINKER_FLAGS_TSAN "-fsanitize=thread"
    CACHE STRING "Flags used for linking binaries during TSan builds." FORCE)

set(CMAKE_SHARED_LINKER_FLAGS_TSAN "-fsanitize=thread"
    CACHE STRING "Flags used by the shared libraries linker during TSan builds." FORCE)

# Memory sanitizer
set(CMAKE_CXX_FLAGS_MSAN "-DNDEBUG -g -O1 -fno-omit-frame-pointer -fsanitize=memory"
    CACHE STRING "Flags used by the C++ compiler during MSan builds." FORCE)

set(CMAKE_C_FLAGS_MSAN "${CMAKE_CXX_FLAGS_MSAN}"
    CACHE STRING "Flags used by the C compiler during MSan builds." FORCE)

set(CMAKE_EXE_LINKER_FLAGS_MSAN "-fsanitize=memory"
    CACHE STRING "Flags used for linking binaries during MSan builds." FORCE)

set(CMAKE_SHARED_LINKER_FLAGS_MSAN "-fsanitize=memory"
    CACHE STRING "Flags used by the shared libraries linker during MSan builds." FORCE)

# CPU Profiler
set(CMAKE_CXX_FLAGS_GPERF "-DNDEBUG -g -O1 -fno-omit-frame-pointer"
    CACHE STRING "Flags used by the C++ compiler during Gperf builds." FORCE)

set(CMAKE_C_FLAGS_GPERF "${CMAKE_CXX_FLAGS_GPERF}"
    CACHE STRING "Flags used by the C compiler during Gperf builds." FORCE)

set(CMAKE_EXE_LINKER_FLAGS_GPERF ""
    CACHE STRING "Flags used for linking binaries during Gperf builds." FORCE)

set(CMAKE_SHARED_LINKER_FLAGS_GPERF ""
    CACHE STRING "Flags used by the shared libraries linker during Gperf builds." FORCE)

# Optracing
set(CMAKE_CXX_FLAGS_OPTRACING "-DNDEBUG -O3"
    CACHE STRING "Flags used by the C++ compiler during OpTracing builds." FORCE)

set(CMAKE_C_FLAGS_OPTRACING "${CMAKE_CXX_FLAGS_OPTRACING}"
    CACHE STRING "Flags used by the C compiler during OpTracing builds." FORCE)

set(CMAKE_EXE_LINKER_FLAGS_OPTRACING ""
    CACHE STRING "Flags used for linking binaries during OpTracing builds." FORCE)

set(CMAKE_SHARED_LINKER_FLAGS_OPTRACING ""
    CACHE STRING "Flags used by the shared libraries linker during OpTracing builds." FORCE)

foreach(build IN ITEMS ASAN TSAN MSAN OPTRACING GPERF)
    mark_as_advanced(
        CMAKE_CXX_FLAGS_${build}
        CMAKE_C_FLAGS_${build}
        CMAKE_EXE_LINKER_FLAGS_${build}
        CMAKE_SHARED_LINKER_FLAGS_${build}
    )
endforeach()

# Update the documentation string of CMAKE_BUILD_TYPE for GUIs
set(CMAKE_BUILD_TYPE "${CMAKE_BUILD_TYPE}"
    CACHE STRING
    "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel ASan TSan MSan OpTracing Gperf."
    FORCE)

message(STATUS "Using build type: ${CMAKE_BUILD_TYPE}")
if(CMAKE_BUILD_TYPE STREQUAL "Gperf")
    find_package(Gperftools REQUIRED)
    set_package_properties(Gperftools PROPERTIES TYPE REQUIRED PURPOSE "For gperftools cpu profiler")
    set(WITH_GPERFTOOLS ON)
endif(CMAKE_BUILD_TYPE STREQUAL "Gperf")
