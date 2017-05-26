include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_executable(executor
    "oplibraries/ioplibrary.cpp"
    "oplibraries/tfoplibrary.cpp"
    "rpcserver/irpcserver.cpp"
    "rpcserver/rpcservercore.cpp"
    "rpcserver/zmqserver.cpp"
    "platform/memory.cpp"
    "executionengine.cpp"
    "executor.cpp"
)
target_link_libraries(executor
    protos_gen

    spdlog::spdlog

    protobuf::libprotobuf
    #gRPC::grpc++
    ZeroMQ::zmq
    tensorflow::kernels
)
