# - Run checks about whether a specific source code compiles given current compiler options
#
# Though CMake now supports Clang-style compiler features, this does
# not (at present?) check for library support features such as headers
# and implementation (e.g. <regex>, <memory> and std::shared_ptr).
#
# This is a basic function to check that a source exercising a
# particular library feature compiles (and optional runs if not
# cross-compiling)
#
# Requires cmake 3.8+ for CMP0067 to honer language standard settings
#
# Copyright 2011,2012 Rolf Eike Beer <eike@sf-mail.de>
# Copyright 2012 Andreas Weis
# Copyright 2014,2015 Ben Morgan <Ben.Morgan@warwick.ac.uk>
# Copyright 2017 Aetf <aetf@unlimitedcodeworks.xyz>
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of the copyright holders nor the
#       names of its contributors may be used to endorse or promote products
#       derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

cmake_minimum_required(VERSION 3.8.0)

if(NOT __CHECKCXXFEATURE_LOADED)
  set(__CHECKCXXFEATURE_LOADED 1)
else()
  return()
endif()

set(__CHECKCXXFEATURE_DIR "${CMAKE_CURRENT_LIST_DIR}/checks")

#-----------------------------------------------------------------------
# Check that compiler support C++ FEATURE_NAME, setting RESULT_VAR to
# TRUE if feature is present
function(check_cxx_feature FEATURE_NAME RESULT_VAR)

  if(NOT DEFINED ${RESULT_VAR})
    set(_bindir "${CMAKE_BINARY_DIR}/CheckCXXFeature/${FEATURE_NAME}")
    set(_SRCFILE_BASE ${__CHECKCXXFEATURE_DIR}/test-${FEATURE_NAME})
    set(_LOG_NAME "\"${FEATURE_NAME}\"")
    message(STATUS "Checking support for ${_LOG_NAME}")

    set(_SRCFILE "${_SRCFILE_BASE}.cpp")
    set(_SRCFILE_FAIL "${_SRCFILE_BASE}_fail.cpp")
    set(_SRCFILE_FAIL_COMPILE "${_SRCFILE_BASE}_fail_compile.cpp")

    if(NOT EXISTS "${_SRCFILE}")
      message(STATUS "Checking support for ${_LOG_NAME} not supported (no test available)")
      set(${RESULT_VAR} FALSE CACHE INTERNAL "support for ${_LOG_NAME}")
      return()
    endif()

    if(CMAKE_CROSSCOMPILING)
      try_compile(${RESULT_VAR} "${_bindir}" "${_SRCFILE}"
        COMPILE_DEFINITIONS "${COMPILE_OPTIONS}")
      if(${RESULT_VAR} AND EXISTS ${_SRCFILE_FAIL})
        try_compile(${RESULT_VAR} "${_bindir}_fail" "${_SRCFILE_FAIL}"
          COMPILE_DEFINITIONS "${COMPILE_OPTIONS}")
      endif()
    else()
      try_run(_RUN_RESULT_VAR _COMPILE_RESULT_VAR
        "${_bindir}" "${_SRCFILE}"
        COMPILE_DEFINITIONS "${COMPILE_OPTIONS}")
      if(_COMPILE_RESULT_VAR AND NOT _RUN_RESULT_VAR)
        set(${RESULT_VAR} TRUE)
      else()
        set(${RESULT_VAR} FALSE)
      endif()

      if(${RESULT_VAR} AND EXISTS ${_SRCFILE_FAIL})
        try_run(_RUN_RESULT_VAR _COMPILE_RESULT_VAR
          "${_bindir}_fail" "${_SRCFILE_FAIL}"
          COMPILE_DEFINITIONS "${COMPILE_OPTIONS}")
        if(_COMPILE_RESULT_VAR AND _RUN_RESULT_VAR)
          set(${RESULT_VAR} TRUE)
        else()
          set(${RESULT_VAR} FALSE)
        endif()
      endif()
    endif()

    if(${RESULT_VAR} AND EXISTS ${_SRCFILE_FAIL_COMPILE})
      try_compile(_TMP_RESULT "${_bindir}_fail_compile" "${_SRCFILE_FAIL_COMPILE}"
        COMPILE_DEFINITIONS "${COMPILE_OPTIONS}")
      if(_TMP_RESULT)
        set(${RESULT_VAR} FALSE)
      else()
        set(${RESULT_VAR} TRUE)
      endif()
    endif()

    if(${RESULT_VAR})
      message(STATUS "Checking support for ${_LOG_NAME}: works")
    else()
      message(STATUS "Checking support for ${_LOG_NAME}: not supported")
    endif()

    set(${RESULT_VAR} ${${RESULT_VAR}} CACHE INTERNAL "support for ${_LOG_NAME}")
  endif()
endfunction()
