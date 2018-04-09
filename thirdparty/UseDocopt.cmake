# Use CMAKE_CURRENT_LIST_DIR as this file will be included
set(third_party_dir ${CMAKE_CURRENT_LIST_DIR})

# Library name, also the path
set(library_name docopt.cpp)

message("-- ${library_name} version: bundled")

# Set any options
#set(SPDLOG_BUILD_TESTING OFF CACHE BOOL "Build spdlog tests" FORCE)

add_submodule(${library_name})
