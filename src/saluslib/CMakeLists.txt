set(SRC_LIST
    "platform/logging.cpp"

    "oplibraries/ioplibrary.cpp"

    "memorymgr/memorymgr.cpp"

    "execution/scheduler/operationitem.cpp"
    "execution/scheduler/sessionitem.cpp"
    "execution/scheduler/ischeduler.cpp"
    "execution/scheduler/impl/fair.cpp"
    "execution/scheduler/impl/pack.cpp"
    "execution/scheduler/impl/preempt.cpp"
    "execution/executionengine.cpp"
    "execution/devices.cpp"
    "execution/operationtask.cpp"
    "execution/resources.cpp"
    "execution/threadpool/nonblockingthreadpool.cpp"
    "execution/scheduler/ischeduler.cpp"

    "rpcserver/iothreadpool.cpp"
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
)
if(WIN32)
    list(APPEND SRC_LIST
        "platform/windows/memory.cpp"
        "platform/windows/signals.cpp"
    )
else(WIN32)
    list(APPEND SRC_LIST
        "platform/posix/memory.cpp"
        "platform/posix/signals.cpp"
    )
endif(WIN32)
prepend_each(SRC_LIST src ${SRC_LIST})

# Generated sources
set(proto_files
    "src/protos/executor.proto"
)
set(PROTO_SRC_DIR ${CMAKE_CURRENT_BINARY_DIR})
protobuf_generate_cpp(proto_srcs proto_hdrs ${PROTO_SRC_DIR} ${proto_files})
list(APPEND SRC_LIST
    src/protos/executor.protos.cpp
    ${proto_srcs}
)

add_library(salus SHARED ${SRC_LIST})
target_include_directories(salus
    PUBLIC
        # TODO: namespace public headers
        ${CMAKE_CURRENT_SOURCE_DIR}/include/salus
        ${PROTO_SRC_DIR}
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)
# TODO: clean interface
target_link_libraries(salus
    PUBLIC
        Boost::thread
        ZeroMQ::zmq
        easyloggingpp
        protobuf::libprotobuf
        Boost::boost
    PRIVATE
)
set_target_properties(salus PROPERTIES VERSION ${PROJECT_VERSION})
set_target_properties(salus PROPERTIES SOVERSION ${PROJECT_VERSION_MAJOR})
generate_export_header(salus)
