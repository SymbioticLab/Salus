include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set(SRC_LIST
    "oplibraries/ioplibrary.cpp"

    "resources/memorymgr.cpp"
    "resources/iteralloctracker.cpp"
    "resources/resources.cpp"

    "execution/scheduler/operationitem.cpp"
    "execution/scheduler/sessionitem.cpp"
    "execution/scheduler/basescheduler.cpp"
    "execution/scheduler/schedulingparam.cpp"
    "execution/scheduler/impl/fair.cpp"
    "execution/scheduler/impl/pack.cpp"
    "execution/scheduler/impl/preempt.cpp"

    "execution/executionengine.cpp"
    "execution/engine/taskexecutor.cpp"
    "execution/engine/iterationcontext.cpp"
    "execution/engine/resourcecontext.cpp"
    "execution/engine/allocationlistener.cpp"

    "execution/devices.cpp"
    "execution/operationtask.cpp"
    "execution/iterationtask.cpp"
    "execution/threadpool/nonblockingthreadpool.cpp"

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
    "utils/objectpool.cpp"

    "main.cpp"
)

add_subdirectory(platform)

if(USE_TENSORFLOW)
    list(APPEND SRC_LIST
        "oplibraries/tensorflow/tfoplibraryv2.cpp"
        "oplibraries/tensorflow/tfexception.cpp"
        "oplibraries/tensorflow/tfinstance.cpp"
        "oplibraries/tensorflow/tfsession.cpp"
        "oplibraries/tensorflow/tfutils.cpp"
        "oplibraries/tensorflow/handlercallback.cpp"
        "oplibraries/tensorflow/worker/rendezvousmgr.cpp"
        "oplibraries/tensorflow/worker/rendezvouswithhook.cpp"
        "oplibraries/tensorflow/worker/devicecontextwithdevice.cpp"
        "oplibraries/tensorflow/worker/dummyworkercache.cpp"
        "oplibraries/tensorflow/worker/dummysessionmgr.cpp"

        "oplibraries/tensorflow/v3/sigraphmgr.cpp"
        "oplibraries/tensorflow/v3/tf_executor.cpp"
        "oplibraries/tensorflow/v3/smblocker.cpp"

        "oplibraries/tensorflow/device/shadowdevices.cpp"
        "oplibraries/tensorflow/device/salusdevices.cpp"
        "oplibraries/tensorflow/device/cpu.cpp"
        "oplibraries/tensorflow/device/gpu/gpu.cpp"
        "oplibraries/tensorflow/device/gpu/lane/lanemgr.cpp"
        "oplibraries/tensorflow/device/gpu/sessiondevice.cpp"
        "oplibraries/tensorflow/device/sessionallocator.cpp"
    )
endif(USE_TENSORFLOW)

add_executable(salus-server ${SRC_LIST})
target_link_libraries(salus-server
    protos_gen
    platform

    protobuf::libprotobuf
    ZeroMQ::zmq
    Boost::boost
    Boost::thread
    docopt_s
    moodycamel::concurrentqueue
)

if(USE_TENSORFLOW)
    target_link_libraries(salus-server
        tensorflow::kernels
    )
    target_compile_definitions(salus-server
        PRIVATE
            GOOGLE_CUDA=1
    )
endif(USE_TENSORFLOW)

target_link_options(salus-server
    PRIVATE
        "LINKER:--dynamic-list=${CMAKE_CURRENT_SOURCE_DIR}/salus-server.list"
)
set_target_properties(salus-server PROPERTIES
    LINK_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/salus-server.list
)

#---------------------------------------------------------------------------------------
# Instrucment
#---------------------------------------------------------------------------------------
if(WITH_GPERFTOOLS AND WITH_TCMALLOC)
    target_link_libraries(salus-server gperftools::tcmalloc_and_profiler)
elseif(WITH_GPERFTOOLS)
    target_link_libraries(salus-server gperftools::profiler)
elseif(WITH_TCMALLOC)
    target_link_libraries(salus-server gperftools::tcmalloc)
endif()

#---------------------------------------------------------------------------------------
# CUDA Hooker
#---------------------------------------------------------------------------------------
add_subdirectory(cudahook)

#---------------------------------------------------------------------------------------
# Installation
#---------------------------------------------------------------------------------------
#set_target_properties(salus-server PROPERTIES
#    INSTALL_RPATH "${SALUS_DEPS_PATH}/lib"
#)

install(TARGETS salus-server
    RUNTIME DESTINATION bin
)

# also install dependencies in spack-packages
#if (DEFINED SALUS_DEPS_PATH)
#    if (EXISTS ${SALUS_DEPS_PATH}/lib)
#        install(DIRECTORY ${SALUS_DEPS_PATH}/lib/ DESTINATION DESTINATION lib
#            USE_SOURCE_PERMISSIONS
#            FILES_MATCHING
#            PATTERN "*.so.*"
#            PATTERN "*.so"
#            PATTERN "pkgconfig" EXCLUDE
#        )
#    endif()
#endif()
