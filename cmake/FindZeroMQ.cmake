# Locates the ZeroMQ library and include directories.
# Copyright (C) 2017 Peifeng Yu <peifeng@umich.edu>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# Usage Documentation
#
# Cache Variables: (not for direct use in CMakeLists.txt)
#  ZeroMQ_LIBRARY
#  ZeroMQ_STATIC_LIBRARY
#  ZeroMQ_INCLUDE_DIR
#
# Non-cache variables you might use in your CMakeLists.txt:
#  ZeroMQ_FOUND
#  ZeroMQ_LIBRARIES - Libraries to link for consumer targets
#  ZeroMQ_STATIC_LIBRARIES - Libraries to link for consumer targets
#  ZeroMQ_INCLUDE_DIRS - Include directories
#
# Adds the following targets:
#  ZeroMQ::zmq - alias to dynamic linked library
#  ZeroMQ::zmq-shared - dynamic linked library
#  ZeroMQ::zmq-static - static library
#
# Use this module this way:
#  find_package(ZeroMQ)
#  add_executable(myapp ${SOURCES})
#  target_link_libraries(myapp ZeroMQ::zmq)
#
# Requires these CMake modules:
#  FindPackageHandleStandardArgs (CMake standard module)
include(FindPackageHandleStandardArgs)
unset(ZeroMQ_FOUND)

find_path(ZeroMQ_INCLUDE_DIR NAMES zmq.h)
find_library(ZeroMQ_LIBRARY NAMES libzmq.so)
find_library(ZeroMQ_STATIC_LIBRARY NAMES libzmq.a)

if(ZeroMQ_INCLUDE_DIR)
    set(_ZeroMQ_H ${ZeroMQ_INCLUDE_DIR}/zmq.h)

    function(_zmqver_EXTRACT _ZeroMQ_VER_COMPONENT _ZeroMQ_VER_OUTPUT)
        set(CMAKE_MATCH_1 "0")
        set(_ZeroMQ_expr "^[ \\t]*#define[ \\t]+${_ZeroMQ_VER_COMPONENT}[ \\t]+([0-9]+)$")
        file(STRINGS "${_ZeroMQ_H}" _ZeroMQ_ver REGEX "${_ZeroMQ_expr}")
        string(REGEX MATCH "${_ZeroMQ_expr}" ZeroMQ_ver "${_ZeroMQ_ver}")
        set(${_ZeroMQ_VER_OUTPUT} "${CMAKE_MATCH_1}" PARENT_SCOPE)
    endfunction()

    _zmqver_EXTRACT("ZMQ_VERSION_MAJOR" ZeroMQ_VERSION_MAJOR)
    _zmqver_EXTRACT("ZMQ_VERSION_MINOR" ZeroMQ_VERSION_MINOR)
    _zmqver_EXTRACT("ZMQ_VERSION_PATCH" ZeroMQ_VERSION_PATCH)

    set(ZeroMQ_LIBRARIES ${ZeroMQ_LIBRARY})
    set(ZeroMQ_STATIC_LIBRARIES ${ZeroMQ_STATIC_LIBRARY})
    set(ZeroMQ_INCLUDE_DIRS ${ZeroMQ_INCLUDE_DIR})

    # We should provide version to find_package_handle_standard_args in the same format as it was requested,
    # otherwise it can't check whether version matches exactly.
    if (ZeroMQ_FIND_VERSION_COUNT GREATER 2)
        set(ZeroMQ_VERSION "${ZeroMQ_VERSION_MAJOR}.${ZeroMQ_VERSION_MINOR}.${ZeroMQ_VERSION_PATCH}")
    else()
        # User has requested ZeroMQ version without patch part => user is not interested in specific patch =>
        # any patch should be an exact match.
        set(ZeroMQ_VERSION "${ZeroMQ_VERSION_MAJOR}.${ZeroMQ_VERSION_MINOR}")
    endif()

    # Add imported targets
    add_library(ZeroMQ::zmq-shared SHARED IMPORTED)
    set_target_properties(ZeroMQ::zmq-shared PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${ZeroMQ_INCLUDE_DIRS}"
        IMPORTED_LOCATION "${ZeroMQ_LIBRARY}"
    )

    add_library(ZeroMQ::zmq-static STATIC IMPORTED)
    set_target_properties(ZeroMQ::zmq-static PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${ZeroMQ_INCLUDE_DIRS}"
        IMPORTED_LOCATION "${ZeroMQ_STATIC_LIBRARY}"
    )

    add_library(ZeroMQ::zmq INTERFACE IMPORTED)
    set_target_properties(ZeroMQ::zmq PROPERTIES
        INTERFACE_LINK_LIBRARIES ZeroMQ::zmq-shared
    )
endif(ZeroMQ_INCLUDE_DIR)

find_package_handle_standard_args(ZeroMQ FOUND_VAR ZeroMQ_FOUND
    REQUIRED_VARS ZeroMQ_INCLUDE_DIRS ZeroMQ_LIBRARIES ZeroMQ_STATIC_LIBRARIES
    VERSION_VAR ZeroMQ_VERSION)

if (ZeroMQ_FOUND)
    mark_as_advanced(ZeroMQ_INCLUDE_DIRS ZeroMQ_LIBRARIES ZeroMQ_VERSION
        ZeroMQ_VERSION_MAJOR ZeroMQ_VERSION_MINOR ZeroMQ_VERSION_PATCH)
endif()
