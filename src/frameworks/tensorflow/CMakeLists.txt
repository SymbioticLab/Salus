set(SRC_LIST
    "tfoplibraryv2.cpp"
    "tfexception.cpp"
    "tfinstance.cpp"
    "tfsession.cpp"
    "tfutils.cpp"
    "worker/rendezvousmgr.cpp"
    "worker/rendezvouswithhook.cpp"
    "worker/devicecontextwithdevice.cpp"
    "worker/dummyworkercache.cpp"
    "worker/dummysessionmgr.cpp"
    "worker/mdgraphmgr.cpp"

    "v2/md_executor.cpp"
    "v2/tfallocator.cpp"
    "v2/peropallocdevice.cpp"
    "v2/exectask.cpp"
    "v2/tensorutils.cpp"
    "v2/md_executor_impl.cpp"
)

set(proto_files
    protos/tfoplibrary.proto
)

# Generated sources
set(PROTO_SRC_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(PROTOBUF_IMPORT_DIRS ${TensorFlow_PROTO_DIRS})
protobuf_generate_cpp(proto_srcs proto_hdrs ${PROTO_SRC_DIR} ${proto_files})

salus_add_oplib(tfoplibrary SRCS ${SRC_LIST} ${proto_srcs})
target_include_directories(tfoplibrary
    PRIVATE
        ${PROTO_SRC_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/protos
)
target_link_libraries(tfoplibrary
    PRIVATE
        protobuf::libprotobuf
        tensorflow::kernels
)
