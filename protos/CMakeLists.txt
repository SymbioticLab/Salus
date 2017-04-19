# Proto file
set(proto_files
    "helloworld.proto"
)

# Generated sources
set(PROTO_SRC_DIR ${CMAKE_CURRENT_BINARY_DIR}/proto-gen)
file(MAKE_DIRECTORY ${PROTO_SRC_DIR})

protobuf_generate_cpp(proto_srcs proto_hdrs ${PROTO_SRC_DIR} ${proto_files})
grpc_generate_cpp(grpc_srcs grpc_hdrs ${PROTO_SRC_DIR} ${proto_files})

# Compile to a static lib
add_library(protos_gen STATIC ${proto_srcs} ${grpc_srcs})
target_include_directories(protos_gen
    PUBLIC ${PROTO_SRC_DIR}
)
