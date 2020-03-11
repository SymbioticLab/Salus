set(SRC_LIST
    "protoutils.cpp"
    "pointerutils.cpp"
    "stringutils.cpp"
    "threadutils.cpp"
    "envutils.cpp"
    "zmqutils.cpp"
    "containerutils.cpp"
    "cpp17.cpp"
    "debugging.cpp"
    "objectpool.cpp"
)

add_library(utils STATIC ${SRC_LIST})
target_link_libraries(utils
    PUBLIC
        Boost::boost
        protobuf::libprotobuf
        Boost::thread
        CONAN_PKG::cppzmq
        moodycamel::concurrentqueue
)
