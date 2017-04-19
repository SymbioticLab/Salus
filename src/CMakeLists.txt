add_executable(client "greeter_client.cc")
target_link_libraries(client
    protos_gen

    protobuf::libprotobuf
    gRPC::grpc++
)

add_executable(server "greeter_server.cc")
target_link_libraries(server
    protos_gen

    protobuf::libprotobuf
    gRPC::grpc++_reflection
)
