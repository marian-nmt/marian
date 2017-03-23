project(fast_align)
cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++11 -O3 -g")
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

find_package(SparseHash)
if(SPARSEHASH_FOUND)
  add_definitions(-DHAVE_SPARSEHASH)
  include_directories(${SPARSEHASH_INCLUDE_DIR})
endif(SPARSEHASH_FOUND)

find_package(OpenMP QUIET)
if (OPENMP_FOUND)
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif(OPENMP_FOUND)

add_executable(fast_align src/fast_align.cc src/ttables.cc)
add_executable(atools src/alignment_io.cc src/atools.cc)

set_target_properties(fast_align PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
set_target_properties(atools PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
