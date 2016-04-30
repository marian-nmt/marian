cmake_minimum_required(VERSION 3.5.1)
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

project(amunn CXX)
SET(CMAKE_CXX_FLAGS " -std=c++11 -g -O0 -funroll-loops -Wno-unused-result -Wno-deprecated")
LIST(APPEND CUDA_NVCC_FLAGS --default-stream per-thread; -std=c++11; -g; -O0; -arch=sm_35; -lineinfo; --use_fast_math;)
add_definitions(-DCUDA_API_PER_THREAD_DEFAULT_STREAM)
SET(CUDA_PROPAGATE_HOST_FLAGS OFF)

include_directories(${amunn_SOURCE_DIR})
find_package(CUDA REQUIRED)

find_package(Boost COMPONENTS system filesystem program_options timer)
if(Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS})
    set(EXT_LIBS ${EXT_LIBS} ${Boost_LIBRARIES})
else(Boost_FOUND)
    message(SEND_ERROR "Cannot find Boost libraries. Terminating." )
endif(Boost_FOUND)

find_package (YamlCpp)
if (YAMLCPP_FOUND)
  include_directories(${YAMLCPP_INCLUDE_DIRS})
  set(EXT_LIBS ${EXT_LIBS} ${YAMLCPP_LIBRARY})
endif (YAMLCPP_FOUND)

set(KENLM CACHE STRING "Path to compiled kenlm directory")
if (NOT EXISTS "${KENLM}/build/lib/libkenlm.a")
  message(FATAL_ERROR "Could not find ${KENLM}/build/lib/libkenlm.a")
endif()

set(EXT_LIBS ${EXT_LIBS} ${KENLM}/build/lib/libkenlm.a)
set(EXT_LIBS ${EXT_LIBS} ${KENLM}/build/lib/libkenlm_util.a)
include_directories(${KENLM})
add_definitions(-DKENLM_MAX_ORDER=6)

find_package (BZip2)
if (BZIP2_FOUND)
  include_directories(${BZIP2_INCLUDE_DIRS})
  set(EXT_LIBS ${EXT_LIBS} ${BZIP2_LIBRARIES})
endif (BZIP2_FOUND)

find_package (ZLIB)
if (ZLIB_FOUND)
  include_directories(${ZLIB_INCLUDE_DIRS})
  set(EXT_LIBS ${EXT_LIBS} ${ZLIB_LIBRARIES})
endif (ZLIB_FOUND)

find_package (LibLZMA)
if (LIBLZMA_FOUND)
  include_directories(${LIBLZMA_INCLUDE_DIRS})
  set(EXT_LIBS ${EXT_LIBS} ${LIBLZMA_LIBRARIES})
endif (LIBLZMA_FOUND)
### KenLM stuff - END ###

include_directories($amunn_SOURCE_DIR}/src)
add_subdirectory(src)
