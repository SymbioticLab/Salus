set(third_party_dir ${CMAKE_CURRENT_LIST_DIR})

function(add_submodule library_name)
    set(options "")
    set(oneValueArgs "")
    set(multiValueArgs PATCHES)
    cmake_parse_arguments(ADD_SUBMODULE
        "${options}"
        "${oneValueArgs}"
        "${multiValueArgs}"
        ${ARGN}
    )

    if(NOT EXISTS ${third_party_dir}/${library_name}/CMakeLists.txt)
        message(STATUS "   Initializing submodule")
        execute_process(COMMAND "git" "submodule" "update" "--init" "${library_name}"
            WORKING_DIRECTORY "${third_party_dir}"
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

# Easylogging++
include(${PROJECT_SOURCE_DIR}/thirdparty/UseEasyloggingpp.cmake)

# docopt.cpp
include(${PROJECT_SOURCE_DIR}/thirdparty/UseDocopt.cmake)
