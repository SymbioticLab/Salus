
# Use CMAKE_CURRENT_LIST_DIR as this file will be included
set(third_party_dir ${CMAKE_CURRENT_LIST_DIR})

if(NOT EXISTS ${third_party_dir}/spdlog/CMakeLists.txt)
    message(STATUS "Initialize submodule")
    execute_process(COMMAND "git" "submodule" "update" "--init" "spdlog"
        WORKING_DIRECTORY "${third_party_dir}"
        RESULT_VARIABLE retcode
    )
    if(NOT "${retcode}" STREQUAL "0")
        message(FATAL_ERROR "Failed to checkout spdlog as submodule")
    endif(NOT "${retcode}" STREQUAL "0")
endif(NOT EXISTS ${third_party_dir}/spdlog/CMakeLists.txt)

SET(SPDLOG_BUILD_TESTING OFF CACHE BOOL "Build spdlog tests" FORCE)
add_subdirectory(${third_party_dir}/spdlog)
# make a imported target from it to keep consistency with find_package
add_library(spdlog::spdlog INTERFACE IMPORTED)
get_target_property(spdlog_inc spdlog INTERFACE_INCLUDE_DIRECTORIES)
set_target_properties(spdlog::spdlog PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${spdlog_inc}"
)
