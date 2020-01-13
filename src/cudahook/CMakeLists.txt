set(SRC_LIST
    cudahook.cpp
    realdlsym.cpp
    kernellaunches.cpp)

add_library(cudahook SHARED ${SRC_LIST})

set_target_properties(cudahook PROPERTIES
    VISIBILITY_INLINES_HIDDEN 1
    CXX_VISIBILITY_PRESET hidden
)

include(GenerateExportHeader)
generate_export_header(cudahook)
target_include_directories(cudahook
    PUBLIC
        ${CMAKE_CURRENT_BINARY_DIR}
)

target_compile_definitions(cudahook
    PRIVATE
        _GNU_SOURCE=1
)

target_link_libraries(cudahook
    PRIVATE
        ${CMAKE_DL_LIBS}
)

install(TARGETS cudahook
    LIBRARY DESTINATION lib
)
