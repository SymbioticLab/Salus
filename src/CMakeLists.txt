add_executable(executor
    "executor.cpp"
    "executionengine.cpp"
    "oplibraries/ioplibrary.cpp"
    "oplibraries/tfoplibrary.cpp"
)
target_link_libraries(executor
    protos_gen

    protobuf::libprotobuf
    gRPC::grpc++
    tensorflow::all
)
