set(SRC_LIST
    "logging.cpp"
)

if(WIN32)
    list(APPEND SRC_LIST
        "windows/memory.cpp"
    )
else(WIN32)
    list(APPEND SRC_LIST
        "posix/memory.cpp"
    )
endif(WIN32)

add_library(platform STATIC ${SRC_LIST})
target_link_libraries(platform
    PUBLIC
    easyloggingpp

    PRIVATE
    protos_gen
)
