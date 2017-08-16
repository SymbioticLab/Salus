include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set(SRC_LIST
    "oplibraries/ioplibrary.cpp"

    "memorymgr/memorymgr.cpp"

    "execution/executionengine.cpp"
    "execution/devices.cpp"

    "rpcserver/rpcservercore.cpp"
    "rpcserver/zmqserver.cpp"

    "utils/protoutils.cpp"
    "utils/pointerutils.cpp"
    "utils/stringutils.cpp"
    "utils/threadutils.cpp"
    "utils/envutils.cpp"

    "executor.cpp"
)

add_subdirectory(platform)

if(USE_TENSORFLOW)
    list(APPEND SRC_LIST
        "oplibraries/tfoplibrary.cpp"
        "oplibraries/tfmocks/tfdevice.cpp"
        "oplibraries/tfmocks/tfallocator.cpp"
        "oplibraries/tfmocks/tfsession.cpp"
        "oplibraries/tfmocks/tfrendezvous.cpp"
    )
endif(USE_TENSORFLOW)

add_executable(executor ${SRC_LIST})
target_link_libraries(executor
    protos_gen
    platform

    spdlog::spdlog
    protobuf::libprotobuf
    ZeroMQ::zmq
    Boost::boost
    q
    docopt_s
)

if(USE_TENSORFLOW)
    target_link_libraries(executor
        tensorflow::kernels
    )
endif(USE_TENSORFLOW)
