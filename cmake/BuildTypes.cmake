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
set(CMAKE_CXX_FLAGS_TSAN "-DNDEBUG -g -O1 -fno-omit-frame-pointer -fsanitize=thread"
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
set(CMAKE_CXX_FLAGS_PROFILING "-DNDEBUG -g -O1 -fno-omit-frame-pointer"
    CACHE STRING "Flags used by the C++ compiler during Profiling builds." FORCE)

set(CMAKE_C_FLAGS_PROFILING "${CMAKE_CXX_FLAGS_PROFILING}"
    CACHE STRING "Flags used by the C compiler during Profiling builds." FORCE)

set(CMAKE_EXE_LINKER_FLAGS_PROFILING ""
    CACHE STRING "Flags used for linking binaries during Profiling builds." FORCE)

set(CMAKE_SHARED_LINKER_FLAGS_PROFILING ""
    CACHE STRING "Flags used by the shared libraries linker during Profiling builds." FORCE)

foreach(build IN ITEMS ASAN TSAN MSAN PROFILING)
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
    "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel ASan TSan MSan Profiling."
    FORCE)

message(STATUS "Using build type: ${CMAKE_BUILD_TYPE}")
if(CMAKE_BUILD_TYPE EQUAL "Profiling")
    find_package(Gperftools REQUIRED)
    set(WITH_GPERFTOOLS ON)
endif(CMAKE_BUILD_TYPE EQUAL "Profiling")
