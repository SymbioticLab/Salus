# Proto file
set(proto_files
    "helloworld.proto"
    "route_guide.proto"
    "executor.proto"
)

# Generated sources
set(PROTO_SRC_DIR ${CMAKE_CURRENT_BINARY_DIR}/proto-gen)
file(MAKE_DIRECTORY ${PROTO_SRC_DIR})

set(PROTOBUF_IMPORT_DIRS ${TensorFlow_PROTO_DIRS})
protobuf_generate_cpp(proto_srcs proto_hdrs ${PROTO_SRC_DIR} ${proto_files})
if(GRPC_FOUND)
    grpc_generate_cpp(grpc_srcs grpc_hdrs ${PROTO_SRC_DIR} ${proto_files})
endif()

# Compile to a static lib
add_library(protos_gen STATIC ${proto_srcs} ${grpc_srcs})
target_include_directories(protos_gen
    PUBLIC ${PROTO_SRC_DIR}
)

# HACK: we need the proto definition from TensorFlow
target_link_libraries(protos_gen tensorflow::headers)
