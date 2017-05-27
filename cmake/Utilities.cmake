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

macro(add_compile_options_with_check flag)
  check_cxx_compiler_flag(${flag} COMPILER_SUPPORT${flag})
  if(COMPILER_SUPPORT${flag})
    add_compile_options("${flag}")
  endif()
endmacro()
