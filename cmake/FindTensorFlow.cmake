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
#  TensorFlow_LIBRARIES - Libraries to link for consumer targets
#  TensorFlow_INCLUDE_DIRS - Include directories
#  TensorFlow_PROTO_DIRS - Include directories for protobuf imports
#
# Adds the following targets:
#  tensorflow::headers - A header only interface library
#  tensorflow::all - The whole tensorflow library
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

# include dir for generated files
find_path(TensorFlow_GEN_INCLUDE_DIR
    NAMES
    tensorflow/core/framework/node_def.pb.h
    PATHS ${TensorFlow_ROOT}
    PATH_SUFFIXES bazel-genfiles
    NO_DEFAULT_PATH
)
find_path(TensorFlow_GEN_INCLUDE_DIR
    NAMES
    tensorflow/core/framework/node_def.pb.h
)

find_library(TensorFlow_LIBRARY NAMES tensorflow
    PATHS ${TensorFlow_ROOT}
    PATH_SUFFIXES bazel-bin/tensorflow
    NO_DEFAULT_PATH
)
# fall back to system paths
find_library(TensorFlow_LIBRARY NAMES tensorflow)

# set TensorFlow_FOUND
find_package_handle_standard_args(TensorFlow DEFAULT_MSG
    TensorFlow_INCLUDE_DIR
    TensorFlow_GEN_INCLUDE_DIR
    TensorFlow_LIBRARY
)

# set external variables for usage in CMakeLists.txt
if(TensorFlow_FOUND)
    set(TensorFlow_LIBRARIES ${TensorFlow_LIBRARY})
    set(TensorFlow_INCLUDE_DIRS ${TensorFlow_INCLUDE_DIR} ${TensorFlow_GEN_INCLUDE_DIR})
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

endif()

# hide locals from GUI
mark_as_advanced(TensorFlow_INCLUDE_DIR TensorFlow_LIBRARY)

