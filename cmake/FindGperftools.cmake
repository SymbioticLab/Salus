# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

#.rst:
# FindGperftools
# -------
#
# Finds the gperftools library
#
# This will define the following variables::
#
#   Gperftools_FOUND    - True if the system has the gperftools library
#   Gperftools_VERSION  - The version of the gperftools library which was found
#
# and the following imported targets::
#
#   gperftools::profiler                  - The cpu profiler library
#   gperftools::tcmalloc                  - The tcmalloc library
#   gperftools::tcmalloc_debug            - The tcmalloc library
#   gperftools::tcmalloc_minimal          - The tcmalloc_minimal library
#   gperftools::tcmalloc_minimal_debug    - The tcmalloc_minimal library
#   gperftools::tcmalloc_and_profiler     - The tcmalloc_and_profiler library
#

find_package(PkgConfig)
pkg_check_modules(PC_profiler QUIET libprofiler)
pkg_check_modules(PC_tcmalloc QUIET libtcmalloc)
pkg_check_modules(PC_tcmalloc_debug QUIET libtcmalloc_debug)
pkg_check_modules(PC_tcmalloc_minimal QUIET libtcmalloc)
pkg_check_modules(PC_tcmalloc_minimal_debug QUIET libtcmalloc_debug)


# Find libprofiler
find_path(profiler_INCLUDE_DIR
    NAMES profiler.h
    PATHS ${PC_profiler_INCLUDE_DIRS}
    PATH_SUFFIXES gperftools
)
find_library(profiler_LIBRARY
    NAMES profiler
    PATHS ${PC_profiler_LIBRARY_DIRS}
)
set(profiler_VERSION ${PC_profiler_VERSION})

# Find tcmalloc
find_path(tcmalloc_INCLUDE_DIR
    NAMES tcmalloc.h
    PATHS
        ${PC_tcmalloc_INCLUDE_DIRS}
        ${PC_tcmalloc_debug_INCLUDE_DIRS}
        ${PC_tcmalloc_minimal_INCLUDE_DIRS}
        ${PC_tcmalloc_minimal_debug_INCLUDE_DIRS}
    PATH_SUFFIXES gperftools
)

foreach(lib IN tcmalloc tcmalloc_debug tcmalloc_minimal tcmalloc_debug)
    find_library(${lib}_LIBRARY
        NAMES ${lib}
        PATHS ${PC_${lib}_LIBRARY_DIRS}
    )
    set(${lib}_VERSION ${PC_${lib}_VERSION})
endforeach()

# Find tcmalloc_and_profiler
find_library(tcmalloc_and_profiler_LIBRARY
    NAMES tcmalloc_and_profiler
    PATHS ${PC_tcmalloc_and_profiler_LIBRARY_DIRS}
)

# Merge versions
set(Gperftools_VERSION ${profiler_VERSION})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Foo
    FOUND_VAR Gperftools_FOUND
    REQUIRED_VARS
        profiler_LIBRARY
        profiler_INCLUDE_DIR
        tcmalloc_LIBRARY
        tcmalloc_INCLUDE_DIR
        tcmalloc_minimal_LIBRARY
        tcmalloc_debug_LIBRARY
        tcmalloc_minimal_debug_LIBRARY
    VERSION_VAR Gperftools_VERSION
)

if(Gperftools_FOUND)
    foreach(lib IN profiler tcmalloc)
        if(NOT TARGET gperftools::${lib})
            add_library(gperftools::${lib} UNKNOWN IMPORTED)
            set_target_properties(gperftools::${lib}
                IMPORTED_LOCATION "${${lib}_LIBRARY}"
                INTERFACE_COMPILE_OPTIONS "${PC_${lib}_CFLAGS_OTHER}"
                INTERFACE_INCLUDE_DIRECTORIES "${${lib}_INCLUDE_DIR}"
            )
        endif()
    endforeach()
    foreach(lib IN tcmalloc_debug tcmalloc_minimal tcmalloc_minimal_debug)
        if(NOT TARGET gperftools::${lib})
            add_library(gperftools::${lib} UNKNOWN IMPORTED)
            set_target_properties(gperftools::${lib}
                IMPORTED_LOCATION "${${lib}_LIBRARY}"
                INTERFACE_COMPILE_OPTIONS "${PC_${lib}_CFLAGS_OTHER}"
                INTERFACE_INCLUDE_DIRECTORIES "${tcmalloc_INCLUDE_DIR}"
                )
        endif()
    endforeach()
endif()

mark_as_advanced(
    profiler_INCLUDE_DIR
    profiler_LIBRARY
    tcmalloc_INCLUDE_DIR
    tcmalloc_LIBRARY
    tcmalloc_debug_LIBRARY
    tcmalloc_minimal_LIBRARY
    tcmalloc_minimal_debug_LIBRARY
)
