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
# User-defined variables:
#  TENSORFLOW_ROOT - Where to find Tensorflow
#
# Cache variables: (not for direct use in CMakeLists.txt)
#  TensorFlow_HOME
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

set(TENSORFLOW_ROOT "$ENV{TensorFlow_DIR}" CACHE PATH "Root directory to look in")

find_path(TensorFlow_HOME
    NAMES
    bazel-tensorflow/tensorflow/core
    bazel-tensorflow/tensorflow/cc
    bazel-tensorflow/third_party
    PATHS ${TENSORFLOW_ROOT}
    NO_DEFAULT_PATH
)
# fall back to system paths
find_path(TensorFlow_HOME
    NAMES
    bazel-tensorflow/tensorflow/core
    bazel-tensorflow/tensorflow/cc
    bazel-tensorflow/third_party
)

find_library(TensorFlow_LIBRARY NAMES tensorflow_framework
    PATHS ${TensorFlow_HOME}
    PATH_SUFFIXES bazel-bin/tensorflow
    NO_DEFAULT_PATH
)
# fall back to system paths
find_library(TensorFlow_LIBRARY NAMES tensorflow_framework)

find_library(TensorFlow_Kernel_LIBRARY NAMES tensorflow_kernels
    PATHS ${TensorFlow_HOME}
    PATH_SUFFIXES bazel-bin/tensorflow
    NO_DEFAULT_PATH
)
find_library(TensorFlow_Kernel_LIBRARY NAMES tensorflow_kernels)

# set TensorFlow_FOUND
find_package_handle_standard_args(TensorFlow DEFAULT_MSG
    TensorFlow_HOME
    TensorFlow_LIBRARY
    TensorFlow_Kernel_LIBRARY
)

# set external variables for usage in CMakeLists.txt
if(TensorFlow_FOUND)
    get_filename_component(tf_repo_name ${TensorFlow_HOME} NAME)
    set(tf_binary_path ${TensorFlow_HOME}/bazel-bin/tensorflow)
    set(tf_src_path ${TensorFlow_HOME}/bazel-${tf_repo_name})

    message("-- Found TensorFlow: ${TensorFlow_HOME}")

    set(TensorFlow_LIBRARIES ${TensorFlow_LIBRARY} ${TensorFlow_Kernel_LIBRARY})
    set(TensorFlow_INCLUDE_DIRS
        ${tf_src_path}
        ${TensorFlow_HOME}/bazel-genfiles
        ${tf_src_path}/external/eigen_archive
        ${tf_src_path}/external/nsync/public
    )
    # This is the same as the include dir
    set(TensorFlow_PROTO_DIRS ${tf_src_path})

    # locate cuda in tensorflow
    if(EXISTS ${tf_src_path}/external/local_config_cuda/cuda/cuda/cuda_config.h)
        file(STRINGS ${tf_src_path}/external/local_config_cuda/cuda/cuda/cuda_config.h
             tf_cuda_config REGEX "TF_CUDA_TOOLKIT_PATH")
        string(REGEX REPLACE "^#define TF_CUDA_TOOLKIT_PATH \"(.+)\"$" "\\1" tf_cuda_path ${tf_cuda_config})
        message("-- Found TF CUDA: ${tf_cuda_path}")

        set(tf_cuda_link_path_flag -L${tf_cuda_path}/lib64)
        list(APPEND TensorFlow_INCLUDE_DIRS ${TensorFlow_HOME}/bazel-${tf_repo_name}/external/local_config_cuda/cuda/)
    else()
        message("-- Found TF CUDA: NotFound")
        set(tf_cuda_link_path_flag "")
    endif()

    # Add imported targets
    add_library(tensorflow::headers INTERFACE IMPORTED)
    set_property(TARGET tensorflow::headers PROPERTY INTERFACE_INCLUDE_DIRECTORIES
        ${TensorFlow_INCLUDE_DIRS}
    )

    add_library(tensorflow::framework SHARED IMPORTED)
    file(STRINGS ${tf_binary_path}/libtensorflow_framework.so-2.params FrameworkLinkLibraries REGEX "^-l")
    set_property(TARGET tensorflow::framework PROPERTY INTERFACE_LINK_LIBRARIES
        tensorflow::headers
        ${tf_cuda_link_path_flag}
        ${FrameworkLinkLibraries}
    )
    set_property(TARGET tensorflow::framework PROPERTY IMPORTED_LOCATION ${TensorFlow_LIBRARY})

    add_library(tensorflow::kernels SHARED IMPORTED)
    file(STRINGS ${tf_binary_path}/libtensorflow_kernels.so-2.params KernelLibraries REGEX "^-l")
    set_property(TARGET tensorflow::kernels PROPERTY INTERFACE_LINK_LIBRARIES
        tensorflow::headers
        ${tf_cuda_link_path_flag}
        ${KernelLinkLibraries}
    )
    set_property(TARGET tensorflow::kernels PROPERTY IMPORTED_LOCATION ${TensorFlow_Kernel_LIBRARY})

endif()

# hide locals from GUI
mark_as_advanced(TensorFlow_INCLUDE_DIR TensorFlow_LIBRARY)

