# Use CMAKE_CURRENT_LIST_DIR as this file will be included
set(third_party_dir ${CMAKE_CURRENT_LIST_DIR})

# Library name, also the path
set(library_name docopt.cpp)

message("-- ${library_name} version: bundled")

if(NOT EXISTS ${third_party_dir}/${library_name}/CMakeLists.txt)
    message(STATUS "--   Initialize submodule")
    execute_process(COMMAND "git" "submodule" "update" "--init" "${library_name}"
        WORKING_DIRECTORY "${third_party_dir}"
        RESULT_VARIABLE retcode
    )
    if(NOT "${retcode}" STREQUAL "0")
        message(FATAL_ERROR "Failed to checkout ${library_name} as submodule")
    endif(NOT "${retcode}" STREQUAL "0")
endif(NOT EXISTS ${third_party_dir}/${library_name}/CMakeLists.txt)

# Set any options
#set(SPDLOG_BUILD_TESTING OFF CACHE BOOL "Build spdlog tests" FORCE)
add_subdirectory(${third_party_dir}/${library_name})
