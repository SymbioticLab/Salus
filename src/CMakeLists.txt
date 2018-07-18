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
        "oplibraries/tensorflow/worker/rendezvousmgr.cpp"
        "oplibraries/tensorflow/worker/rendezvouswithhook.cpp"
        "oplibraries/tensorflow/worker/devicecontextwithdevice.cpp"
        "oplibraries/tensorflow/worker/dummyworkercache.cpp"
        "oplibraries/tensorflow/worker/dummysessionmgr.cpp"
        "oplibraries/tensorflow/worker/mdgraphmgr.cpp"

        "oplibraries/tensorflow/v2/md_executor.cpp"
        "oplibraries/tensorflow/v2/tfallocator.cpp"
        "oplibraries/tensorflow/v2/exectask.cpp"
        "oplibraries/tensorflow/v2/tensorutils.cpp"
        "oplibraries/tensorflow/v2/md_executor_impl.cpp"
        "oplibraries/tensorflow/v2/graphview.cpp"
        "oplibraries/tensorflow/v2/itertask.cpp"
        "oplibraries/tensorflow/v2/costmgr.cpp"

        "oplibraries/tensorflow/v3/sigraphmgr.cpp"
        "oplibraries/tensorflow/v3/si_executor.cpp"
        "oplibraries/tensorflow/v3/si_executor_impl.cpp"
        "oplibraries/tensorflow/v3/si_executortask.cpp"

        "oplibraries/tensorflow/device/salusdevices.cpp"
        "oplibraries/tensorflow/device/cpu.cpp"
        "oplibraries/tensorflow/device/gpu.cpp"
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

    protobuf::libprotobuf
    ZeroMQ::zmq
    Boost::boost
    Boost::thread
    docopt_s
    moodycamel::concurrentqueue
)

if(USE_TENSORFLOW)
    target_link_libraries(executor
        tensorflow::kernels
    )
    target_compile_definitions(executor
        PRIVATE
            GOOGLE_CUDA=1
    )
endif(USE_TENSORFLOW)

#---------------------------------------------------------------------------------------
# Instrucment
#---------------------------------------------------------------------------------------
if(WITH_GPERFTOOLS AND WITH_TCMALLOC)
    target_link_libraries(executor gperftools::tcmalloc_and_profiler)
elseif(WITH_GPERFTOOLS)
    target_link_libraries(executor gperftools::profiler)
elseif(WITH_TCMALLOC)
    target_link_libraries(executor gperftools::tcmalloc)
endif()
