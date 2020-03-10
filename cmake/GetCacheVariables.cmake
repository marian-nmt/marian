##
# This module extracts CMake cached variables into a variable.
#
# Author: snukky
#
# This module sets the following variables:
# * PROJECT_CMAKE_CACHE - to the output of "cmake -L" - an uncached list of
#     non-advanced cached variables
# * PROJECT_CMAKE_CACHE_ADVANCED - to the output of "cmake -LA" - an uncached
#     list of advanced cached variables
#

set(PROJECT_CMAKE_CACHE "")
set(PROJECT_CMAKE_CACHE_ADVANCED "")

# Get all CMake variables
get_cmake_property(_variableNames VARIABLES)
list(SORT _variableNames)
list(REMOVE_DUPLICATES _variableNames)

foreach(_variableName ${_variableNames})
  # If it is a cache variable
  get_property(_cachePropIsSet CACHE "${_variableName}" PROPERTY VALUE SET)
  if(_cachePropIsSet)
    # Get the variable's type
    get_property(_variableType CACHE ${_variableName} PROPERTY TYPE)

    # Get the variable's value
    set(_variableValue "${${_variableName}}")

    # Skip static or internal cached variables, cmake -L[A] does not print them, see
    # https://github.com/Kitware/CMake/blob/master/Source/cmakemain.cxx#L282
    if( (NOT "${_variableType}" STREQUAL "STATIC") AND
        (NOT "${_variableType}" STREQUAL "INTERNAL") AND
        (NOT "${_variableValue}" STREQUAL "") )


        set(PROJECT_CMAKE_CACHE_ADVANCED "${PROJECT_CMAKE_CACHE_ADVANCED}    \"${_variableName}=${_variableValue}\\n\"\n")

        # Get the variable's advanced flag
        get_property(_isAdvanced CACHE ${_variableName} PROPERTY ADVANCED SET)
        if(NOT _isAdvanced)
          set(PROJECT_CMAKE_CACHE "${PROJECT_CMAKE_CACHE}    \"${_variableName}=${_variableValue}\\n\"\n")
        endif()

        # Print variables for debugging
        #message(STATUS "${_variableName}=${${_variableName}}")
        #message(STATUS "  Type=${_variableType}")
        #message(STATUS "  Advanced=${_isAdvanced}")
    endif()
  endif(_cachePropIsSet)
endforeach()
