include(CheckCXXCompilerFlag)

# A macro that determines the current time
# source: http://www.cmake.org/pipermail/cmake/2004-September/005526.html
# TODO: does not work for Windows 7
macro(current_time result)
  set(NEED_FLAG)
  if(WIN32)
    if(NOT CYGWIN)
      set(NEED_FLAG "/T")
    endif(NOT CYGWIN)
  endif(WIN32)
  exec_program(date ARGS ${NEED_FLAG} OUTPUT_VARIABLE ${result})
endmacro()

# generate unique / temporary file names
# source: http://www.cmake.org/Wiki/CMake/Language_Syntax
macro(temp_name fname)
  if(${ARGC} GREATER 1) # Have to escape ARGC to correctly compare
    set(_base ${ARGV1})
  else(${ARGC} GREATER 1)
    set(_base ".cmake-tmp")
  endif(${ARGC} GREATER 1)
  set(_counter 0)
  while(EXISTS "${_base}${_counter}")
    math(EXPR _counter "${_counter} + 1")
  endwhile(EXISTS "${_base}${_counter}")
  set(${fname} "${_base}${_counter}")
endmacro(temp_name)

# Evaluate expression
# Suggestion from the Wiki: http://cmake.org/Wiki/CMake/Language_Syntax
# Unfortunately, no built-in stuff for this: http://public.kitware.com/Bug/view.php?id=4034
macro(eval expr)
  temp_name(_fname)
  file(WRITE ${_fname} "${expr}")
  include(${_fname})
  file(REMOVE ${_fname})
endmacro(eval)

# Add a global default compile option
macro(add_compile_options_with_check flag)
  string(REPLACE "-" "_" retvar "COMPILER_SUPPORT_${flag}")
  check_cxx_compiler_flag(${flag} ${retvar})
  if(${retvar})
    add_compile_options(${flag})
  endif()
endmacro()

# Add compile option to a build type with check
macro(build_type_add_compile_option_with_check buildtype flag)
    string(REPLACE "-" "_" retvar "COMPILER_SUPPORT_${flag}")
    check_cxx_compiler_flag(${flag} ${retvar})
    if(${retvar})
        set(CMAKE_CXX_FLAGS_${buildtype} "${CMAKE_CXX_FLAGS_${buildtype}}" ${flag})
    endif()
endmacro()

# Print all properties for a target
# From https://stackoverflow.com/questions/32183975/how-to-print-all-the-properties-of-a-target-in-cmake

# Get all propreties that cmake supports
execute_process(COMMAND cmake --help-property-list OUTPUT_VARIABLE CMAKE_PROPERTY_LIST)
# Convert command output into a CMake list
STRING(REGEX REPLACE ";" "\\\\;" CMAKE_PROPERTY_LIST "${CMAKE_PROPERTY_LIST}")
STRING(REGEX REPLACE "\n" ";" CMAKE_PROPERTY_LIST "${CMAKE_PROPERTY_LIST}")

function(print_properties)
    message ("CMAKE_PROPERTY_LIST = ${CMAKE_PROPERTY_LIST}")
endfunction(print_properties)

function(print_target_properties tgt)
    if(NOT TARGET ${tgt})
      message("There is no target named '${tgt}'")
      return()
    endif()

    foreach (prop ${CMAKE_PROPERTY_LIST})
        string(REPLACE "<CONFIG>" "${CMAKE_BUILD_TYPE}" prop ${prop})
    # Fix https://stackoverflow.com/questions/32197663/how-can-i-remove-the-the-location-property-may-not-be-read-from-target-error-i
    if(prop STREQUAL "LOCATION" OR prop MATCHES "^LOCATION_" OR prop MATCHES "_LOCATION$")
        continue()
    endif()
        # message ("Checking ${prop}")
        get_property(propval TARGET ${tgt} PROPERTY ${prop} SET)
        if (propval)
            get_target_property(propval ${tgt} ${prop})
            message ("${tgt} ${prop} = ${propval}")
        endif()
    endforeach(prop)
endfunction(print_target_properties)
