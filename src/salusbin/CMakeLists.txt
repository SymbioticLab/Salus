add_executable(executor src/main.cpp)
target_link_libraries(executor
    salus

    docopt_s
)
