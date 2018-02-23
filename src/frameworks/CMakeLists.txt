function(salus_add_oplib library_name)
    set(options "")
    set(oneValueArgs "")
    set(multiValueArgs SRCS)
    cmake_parse_arguments(SALUS_ADD_OPLIB
        "${options}"
        "${oneValueArgs}"
        "${multiValueArgs}"
        ${ARGN}
    )

    list(LENGTH SALUS_ADD_OPLIB_SRCS src_count)
    if(NOT ${src_count} GREATER 0)
        message(FATAL_ERROR "salus_add_oplib() called without passing any source files. Please use the SRCS parameter.")
    endif()

    add_library(${library_name} MODULE ${SALUS_ADD_OPLIB_SRCS})
    target_link_libraries(${library_name} PRIVATE salus)
    set_target_properties(${library_name} PROPERTIES
        PREFIX ""
        # Mimic the directory structure in build tree
        LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib/salus/oplibs
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN 1
    )
    # Install
    install(TARGETS ${library_name} DESTINATION ${CMAKE_INSTALL_LIBDIR}/salus/oplibs)
endfunction(salus_add_oplib)

if (USE_TENSORFLOW)
    add_subdirectory(tensorflow)
endif(USE_TENSORFLOW)
