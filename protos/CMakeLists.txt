# Proto file
set(proto_files
    "executor.proto"
)

if(USE_TENSORFLOW)
    list(APPEND proto_files "tfoplibrary.proto")
endif(USE_TENSORFLOW)

# Generated sources
set(PROTO_SRC_DIR ${CMAKE_CURRENT_BINARY_DIR})

set(PROTOBUF_IMPORT_DIRS ${TensorFlow_PROTO_DIRS})
protobuf_generate_cpp(proto_srcs proto_hdrs ${PROTO_SRC_DIR} ${proto_files})
if(GRPC_FOUND)
    grpc_generate_cpp(grpc_srcs grpc_hdrs ${PROTO_SRC_DIR} ${proto_files})
endif()

# Compile to a static lib
add_library(protos_gen STATIC ${proto_srcs} ${grpc_srcs})
target_link_libraries(protos_gen protobuf::libprotobuf)
target_include_directories(protos_gen
    INTERFACE ${PROTO_SRC_DIR}
)

if(USE_TENSORFLOW)
    # HACK: we need the proto definition from TensorFlow
        message(${TensorFlow_INCLUDE_DIRS})
    target_link_libraries(protos_gen tensorflow::headers)
endif(USE_TENSORFLOW)
