# Locates the tensorFlow library and include directories.
#
# Cache Variables: (not for direct use in CMakeLists.txt)
#  TensorFlow_ROOT
#  TensorFlow_LIBRARY
#  TensorFlow_INCLUDE_DIR
#
# Non-cache variables you might use in your CMakeLists.txt:
#  TensorFlow_FOUND
#
#  TensorFlow_LIBRARIES
#  TensorFlow_INCLUDE_DIRS
#  TensorFlow_LINKER_FLAGS
#
# Adds the following targets:
#  TensorFlow - The TensorFlow library as a whole
#
# Use this module this way:
#  find_package(TensorFlow)
#  add_executable(myapp ${SOURCES})
#  target_link_libraries(myapp TensorFlow)
#
# Requires these CMake modules:
#  FindPackageHandleStandardArgs (CMake standard module)

include(FindPackageHandleStandardArgs)
unset(TensorFlow_FOUND)

set(TensorFlow_ROOT "${TensorFlow_ROOT}" CACHE PATH "Root directory to look in")

find_path(TensorFlow_INCLUDE_DIR
    NAMES
    tensorflow/core
    tensorflow/cc
    third_party
    PATHS ${TensorFlow_ROOT}
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
    PATHS ${TensorFlow_ROOT}
    PATH_SUFFIXES bazel-bin/tensorflow
    NO_DEFAULT_PATH
)
# fall back to system paths
find_library(TensorFlow_LIBRARY NAMES tensorflow)

# set TensorFlow_FOUND
find_package_handle_standard_args(TensorFlow DEFAULT_MSG TensorFlow_INCLUDE_DIR TensorFlow_LIBRARY)

# set external variables for usage in CMakeLists.txt
if(TensorFlow_FOUND)
    add_library(TensorFlow SHARED IMPORTED)
    set_target_properties(TensorFlow PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES ${TensorFlow_INCLUDE_DIR}
        IMPORTED_LOCATION ${TensorFlow_LIBRARY}
    )

    set(TensorFlow_LIBRARIES ${TensorFlow_LIBRARY})
    set(TensorFlow_INCLUDE_DIRS ${TensorFlow_INCLUDE_DIR})
endif()

# hide locals from GUI
mark_as_advanced(TensorFlow_INCLUDE_DIR TensorFlow_LIBRARY)

