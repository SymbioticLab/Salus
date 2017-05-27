include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set(SRC_LIST
    "oplibraries/ioplibrary.cpp"
    "oplibraries/tfoplibrary.cpp"
    "rpcserver/irpcserver.cpp"
    "rpcserver/rpcservercore.cpp"
    "rpcserver/zmqserver.cpp"
    "platform/memory.cpp"
    "platform/logging.cpp"
    "platform/crashhandler/logcapture.cpp"
    "utils/protoutils.cpp"
    "utils/pointerutils.cpp"
    "utils/stringutils.cpp"
    "executionengine.cpp"
    "executor.cpp"
)

if(WIN32)
    list(APPEND SRC_LIST
        "platform/crashhandler/crashhandler_windows.cpp"
        "platform/crashhandler/stacktrace_windows.cpp"
    )
else(WIN32)
    list(APPEND SRC_LIST
        "platform/crashhandler/crashhandler_unix.cpp"
    )
endif(WIN32)

add_executable(executor ${SRC_LIST})
target_link_libraries(executor
    protos_gen

    spdlog::spdlog

    protobuf::libprotobuf
    #gRPC::grpc++
    ZeroMQ::zmq
    tensorflow::kernels
)
