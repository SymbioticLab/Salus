add_executable(hw_client "greeter_client.cc")
target_link_libraries(hw_client
    protos_gen

    protobuf::libprotobuf
    gRPC::grpc++
)

add_executable(hw_server "greeter_server.cc")
target_link_libraries(hw_server
    protos_gen

    protobuf::libprotobuf
    gRPC::grpc++_reflection
)

add_executable(rg_server "route_guide_server.cpp" "helper.cpp")
target_link_libraries(rg_server
    protos_gen

    protobuf::libprotobuf
    gRPC::grpc++_reflection
)

add_executable(rg_client "route_guide_client.cpp" "helper.cpp")
target_link_libraries(rg_client
    protos_gen

    protobuf::libprotobuf
    gRPC::grpc++
)
