set(SRC_LIST
    "src/platform/logging.cpp"

    "src/oplibraries/ioplibrary.cpp"

    "src/memorymgr/memorymgr.cpp"

    "src/execution/scheduler/operationitem.cpp"
    "src/execution/scheduler/sessionitem.cpp"
    "src/execution/scheduler/ischeduler.cpp"
    "src/execution/scheduler/impl/fair.cpp"
    "src/execution/scheduler/impl/pack.cpp"
    "src/execution/scheduler/impl/preempt.cpp"
    "src/execution/executionengine.cpp"
    "src/execution/devices.cpp"
    "src/execution/operationtask.cpp"
    "src/execution/resources.cpp"
    "src/execution/threadpool/nonblockingthreadpool.cpp"
    "src/execution/scheduler/ischeduler.cpp"

    "src/rpcserver/iothreadpool.cpp"
    "src/rpcserver/rpcservercore.cpp"
    "src/rpcserver/zmqserver.cpp"

    "src/utils/protoutils.cpp"
    "src/utils/pointerutils.cpp"
    "src/utils/stringutils.cpp"
    "src/utils/threadutils.cpp"
    "src/utils/envutils.cpp"
    "src/utils/zmqutils.cpp"
    "src/utils/containerutils.cpp"
    "src/utils/cpp17.cpp"
    "src/utils/debugging.cpp"
    "src/utils/plugins.cpp"
)
if(WIN32)
    list(APPEND SRC_LIST
        "src/platform/windows/memory.cpp"
        "src/platform/windows/signals.cpp"
    )
else(WIN32)
    list(APPEND SRC_LIST
        "src/platform/posix/memory.cpp"
        "src/platform/posix/signals.cpp"
    )
endif(WIN32)

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
target_compile_options(salus
    INTERFACE
        -fPIC # while CMAKE_POSITION_INDEPENDENT_CODE enables -fPIC for libraries, we need to propogate this to executables
)
# TODO: clean interface
target_link_libraries(salus
    PUBLIC
        ZeroMQ::zmq
        easyloggingpp
        protobuf::libprotobuf
        Boost::boost
        Boost::thread
        Boost::filesystem
    PRIVATE
)
set_target_properties(salus PROPERTIES VERSION ${PROJECT_VERSION})
set_target_properties(salus PROPERTIES SOVERSION ${PROJECT_VERSION_MAJOR})
generate_export_header(salus)
