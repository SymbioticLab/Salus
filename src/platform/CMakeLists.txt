set(SRC_LIST
    "logging.cpp"
    "profiler.cpp"
)

if(WIN32)
    list(APPEND SRC_LIST
        "windows/memory.cpp"
        "windows/signals.cpp"
    )
else() # POSIX
    list(APPEND SRC_LIST
        "posix/memory.cpp"
        "posix/signals.cpp"
        "posix/thread_annotations.cpp"
    )
endif(WIN32)

add_library(platform STATIC ${SRC_LIST})
target_link_libraries(platform
    PUBLIC
    easyloggingpp

    PRIVATE
    protos_gen
    Threads::Threads
)

if(NOT WIN32)
    # POSIX
    target_compile_definitions(platform
        PRIVATE
            _GNU_SOURCE=1
    )
endif()
