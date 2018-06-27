function(add_submodule library_name)
    set(options "")
    set(oneValueArgs "THIRDPARTY_DIR")
    set(multiValueArgs PATCHES)
    cmake_parse_arguments(ADD_SUBMODULE
        "${options}"
        "${oneValueArgs}"
        "${multiValueArgs}"
        ${ARGN}
        )

    if(EXISTS ADD_SUBMODULE_THIRDPARTY_DIR)
        set(third_party_dir ${ADD_SUBMODULE_THIRDPARTY_DIR})
    else()
        set(third_party_dir ${CMAKE_CURRENT_LIST_DIR})
    endif()

    if(NOT EXISTS ${third_party_dir}/${library_name}/CMakeLists.txt)
        message(STATUS "   Initializing submodule")
        execute_process(COMMAND "git" "submodule" "update" "--init" "${third_party_dir}/${library_name}"
            WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
            RESULT_VARIABLE retcode
            )
        if(NOT "${retcode}" STREQUAL "0")
            message(FATAL_ERROR "Failed to checkout ${library_name} as submodule: ${retcode}")
        endif(NOT "${retcode}" STREQUAL "0")

        foreach(patch IN LISTS ADD_SUBMODULE_PATCHES)
            message(STATUS "   Applying patch ${patch}")
            get_filename_component(abs_patch ${patch} ABSOLUTE)
            execute_process(COMMAND "git" "apply" "${abs_patch}"
                WORKING_DIRECTORY "${third_party_dir}/${library_name}"
                RESULT_VARIABLE retcode
                )
            if(NOT "${retcode}" STREQUAL "0")
                message(FATAL_ERROR "Failed to intialize ${library_name} when applying ${abs_patch}: ${retcode}")
            endif(NOT "${retcode}" STREQUAL "0")
        endforeach(patch)
    endif(NOT EXISTS ${third_party_dir}/${library_name}/CMakeLists.txt)

    add_subdirectory(${third_party_dir}/${library_name})
endfunction(add_submodule)

set(third_party_dir ${CMAKE_CURRENT_LIST_DIR})

# make sure all dependencies are build in Release mode
set(CMAKE_BUILD_TYPE Release)

# Easylogging++
include(${PROJECT_SOURCE_DIR}/thirdparty/UseEasyloggingpp.cmake)

# docopt.cpp
include(${PROJECT_SOURCE_DIR}/thirdparty/UseDocopt.cmake)

# concurrentqueue
include(${third_party_dir}/UseConcurrentQueue.cmake)
