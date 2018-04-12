# Set compiler flags

## First some basic compiler flags
if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
    # using Clang or AppleClang

    # reasonable and standard
    add_compile_options_with_check(-Weverything)
    add_compile_options_with_check(-Werror)
    add_compile_options_with_check(-Wfatal-errors)

    add_compile_options_with_check(-Wno-c++98-compat)

    add_compile_options_with_check(-Wold-style-cast)
    # helps catch hard to track down memory errors
    add_compile_options_with_check(-Wnon-virtual-dtor)
    # warn for potential performance problem casts
    add_compile_options_with_check(-Wcast-align)
    # warn if you overload (not override) a virtual function
    add_compile_options_with_check(-Woverloaded-virtual)
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    # using GCC

    # reasonable and standard
    add_compile_options_with_check(-Wall)
    add_compile_options_with_check(-Wextra)
    add_compile_options_with_check(-pedantic)
    add_compile_options_with_check(-Werror)
    #add_compile_options_with_check(-Wfatal-errors)

    add_compile_options_with_check(-Wold-style-cast)
    # helps catch hard to track down memory errors
    add_compile_options_with_check(-Wnon-virtual-dtor)
    # warn for potential performance problem casts
    add_compile_options_with_check(-Wcast-align)
    # warn if you overload (not override) a virtual function
    add_compile_options_with_check(-Woverloaded-virtual)
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
    # using Intel C++
    message(WARNING "Using of Intel C++ is not supported, you are on your own.")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    # using Visual Studio C++

    # all reasonable warnings
    add_compile_options_with_check(/W4)
    # treat warnings as errors
    add_compile_options_with_check(/Wx)
    # enable warning on thread un-safe static member initialization
    add_compile_options_with_check(/W44640)
endif()

## Then increase the level of optimization when doing release
build_type_add_compile_option_with_check(Release -O3)
build_type_add_compile_option_with_check(Release -march=native)
# See bug: https://gcc.gnu.org/bugzilla/show_bug.cgi?id=51333
# build_type_add_compile_options_with_check(Debug -fkeep-inline-functions)
