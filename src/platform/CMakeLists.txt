set(SRC_LIST
    "logging.cpp"
    "crashhandler/logcapture.cpp"
)

if(WIN32)
    list(APPEND SRC_LIST
        "windows/crashhandler.cpp"
        "windows/memory.cpp"
    )
else(WIN32)
    list(APPEND SRC_LIST
        "posix/crashhandler.cpp"
        "posix/memory.cpp"
    )
endif(WIN32)

add_library(platform STATIC ${SRC_LIST})
target_link_libraries(platform
    PUBLIC
    spdlog::spdlog

    PRIVATE
    protos_gen
)
