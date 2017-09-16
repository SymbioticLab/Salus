include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set(SRC_LIST
    "oplibraries/ioplibrary.cpp"

    "memorymgr/memorymgr.cpp"

    "execution/executionengine.cpp"
    "execution/devices.cpp"
    "execution/itask.cpp"
    "execution/operationtask.cpp"
    "execution/resources.cpp"

    "rpcserver/rpcservercore.cpp"
    "rpcserver/zmqserver.cpp"

    "utils/protoutils.cpp"
    "utils/pointerutils.cpp"
    "utils/stringutils.cpp"
    "utils/threadutils.cpp"
    "utils/envutils.cpp"
    "utils/zmqutils.cpp"
    "utils/containerutils.cpp"
    "utils/cpp17.cpp"
    "utils/debugging.cpp"

    "executor.cpp"
)

add_subdirectory(platform)

if(USE_TENSORFLOW)
    list(APPEND SRC_LIST
        "oplibraries/tensorflow/tfoplibraryv2.cpp"
        "oplibraries/tensorflow/v2/md_executor.cpp"
        "oplibraries/tensorflow/v2/md_rendezvous.cpp"
        "oplibraries/tensorflow/v2/tfallocator.cpp"
        "oplibraries/tensorflow/v2/peropallocdevice.cpp"
        "oplibraries/tensorflow/v2/exectask.cpp"
        "oplibraries/tensorflow/v2/tensorutils.cpp"
        "oplibraries/tensorflow/v2/md_executor_impl.cpp"
#         "oplibraries/tensorflow/tfoplibrary.cpp"
#         "oplibraries/tensorflow/tfmocks/tfdevice.cpp"
#         "oplibraries/tensorflow/tfmocks/tfsession.cpp"
#         "oplibraries/tensorflow/tfmocks/tfrendezvous.cpp"
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
        tensorflow::all
    )
endif(USE_TENSORFLOW)
