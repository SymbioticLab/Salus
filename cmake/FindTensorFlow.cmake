# Locates the tensorFlow library and include directories.
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
#  TENSORFLOW_ROOT
#  TensorFlow_LIBRARY
#  TensorFlow_INCLUDE_DIR
#
# Non-cache variables you might use in your CMakeLists.txt:
#  TensorFlow_FOUND
#
#  TensorFlow_LIBRARIES - Libraries to link for consumer targets
#  TensorFlow_INCLUDE_DIRS - Include directories
#  TensorFlow_PROTO_DIRS - Include directories for protobuf imports
#
# Adds the following targets:
#  tensorflow::headers - A header only interface library
#  tensorflow::all - The whole tensorflow library
#  tensorflow::kernels - All kernels
#
# Use this module this way:
#  find_package(TensorFlow)
#  add_executable(myapp ${SOURCES})
#  target_link_libraries(myapp tensorflow::all)
#
# Requires these CMake modules:
#  FindPackageHandleStandardArgs (CMake standard module)

include(FindPackageHandleStandardArgs)
unset(TensorFlow_FOUND)

set(TENSORFLOW_ROOT "${TENSORFLOW_ROOT}" CACHE PATH "Root directory to look in")

find_path(TensorFlow_INCLUDE_DIR
    NAMES
    tensorflow/core
    tensorflow/cc
    third_party
    PATHS ${TENSORFLOW_ROOT}
    NO_DEFAULT_PATH
)
# fall back to system paths
find_path(TensorFlow_INCLUDE_DIR
    NAMES
    tensorflow/core
    tensorflow/cc
    third_party
)

find_library(TensorFlow_LIBRARY NAMES tensorflow
    PATHS ${TENSORFLOW_ROOT}
    PATH_SUFFIXES bazel-bin/tensorflow
    NO_DEFAULT_PATH
)
# fall back to system paths
find_library(TensorFlow_LIBRARY NAMES tensorflow)

find_library(TensorFlow_Kernel_LIBRARY NAMES tensorflow_kernels
    PATHS ${TENSORFLOW_ROOT}
    PATH_SUFFIXES bazel-bin/tensorflow
    NO_DEFAULT_PATH
)
find_library(TensorFlow_Kernel_LIBRARY NAMES tensorflow_kernels)

# set TensorFlow_FOUND
find_package_handle_standard_args(TensorFlow DEFAULT_MSG
    TensorFlow_INCLUDE_DIR
    TensorFlow_LIBRARY
    TensorFlow_Kernel_LIBRARY
)

# set external variables for usage in CMakeLists.txt
if(TensorFlow_FOUND)
    set(TensorFlow_LIBRARIES ${TensorFlow_LIBRARY})
    get_filename_component(tf_repo_name ${TensorFlow_INCLUDE_DIR} NAME)
    set(TensorFlow_INCLUDE_DIRS
        ${TensorFlow_INCLUDE_DIR}
        ${TensorFlow_INCLUDE_DIR}/bazel-genfiles
        ${TensorFlow_INCLUDE_DIR}/bazel-${tf_repo_name}/external/eigen_archive
    )
    # This is the same as the include dir
    set(TensorFlow_PROTO_DIRS ${TensorFlow_INCLUDE_DIR})

    # Add imported targets
    add_library(tensorflow::headers INTERFACE IMPORTED)
    set_target_properties(tensorflow::headers PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${TensorFlow_INCLUDE_DIRS}"
    )

    add_library(tensorflow::all SHARED IMPORTED)
    set_target_properties(tensorflow::all PROPERTIES
        INTERFACE_LINK_LIBRARIES tensorflow::headers
        IMPORTED_LOCATION ${TensorFlow_LIBRARIES}
    )

    add_library(tensorflow::kernels SHARED IMPORTED)
    set_target_properties(tensorflow::kernels PROPERTIES
        INTERFACE_LINK_LIBRARIES tensorflow::headers
        IMPORTED_LOCATION ${TensorFlow_Kernel_LIBRARY}
    )

endif()

# hide locals from GUI
mark_as_advanced(TensorFlow_INCLUDE_DIR TensorFlow_LIBRARY)

