include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set(SRC_LIST
    "oplibraries/ioplibrary.cpp"
    "oplibraries/tfoplibrary.cpp"

    "oplibraries/tfmocks/tfdevice.cpp"
    "oplibraries/tfmocks/tfallocator.cpp"
    "oplibraries/tfmocks/tfsession.cpp"
    "oplibraries/tfmocks/tfrendezvous.cpp"

    "memorymgr/memorymgr.cpp"

    "execution/executionengine.cpp"
    "execution/devices.cpp"

    "rpcserver/rpcservercore.cpp"
    "rpcserver/zmqserver.cpp"

    "platform/memory.cpp"
    "platform/logging.cpp"
    "platform/crashhandler/logcapture.cpp"

    "utils/protoutils.cpp"
    "utils/pointerutils.cpp"
    "utils/stringutils.cpp"

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
    ZeroMQ::zmq
    tensorflow::kernels
    Boost::thread
    Boost::boost
    q
)
