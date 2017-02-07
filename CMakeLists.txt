cmake_minimum_required(VERSION 3.5.1)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

project(amunn C CXX)

set(CMAKE_CXX_FLAGS_RELEASE "-std=c++14 -fPIC -O3 -Ofast -m64 -march=native -funroll-loops -ffinite-math-only -Wno-unused-result -Wno-deprecated -pthread")
set(CMAKE_CXX_FLAGS_DEBUG "-std=c++14 -fPIC -g -O0 -m64  -march=native -funroll-loops -ffinite-math-only -Wno-unused-result -Wno-deprecated -pthread")
set(CMAKE_CXX_FLAGS_PROFILE "${CMAKE_CXX_FLAGS_RELEASE} -g -pg")
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS_RELEASE})

if(BUILD_STATIC)
  set(CMAKE_FIND_LIBRARY_SUFFIXES ".a")
  set(CMAKE_EXE_LINKER_FLAGS "-static")
  set(Boost_USE_STATIC_LIBS  "TRUE")
endif(BUILD_STATIC)

include_directories(${amunmt_SOURCE_DIR})

option(CUDA "Select to compile CUDA support by default if available" ON)

if(CUDA)
  find_package(CUDA)
  if(CUDA_FOUND)
    LIST(APPEND CUDA_NVCC_FLAGS --default-stream per-thread; -std=c++11; -O3; -arch=sm_35; -lineinfo; --use_fast_math;)
    # add_definitions(-DCUDA_API_PER_THREAD_DEFAULT_STREAM)
    add_definitions(-DCUDA)
    SET(CUDA_PROPAGATE_HOST_FLAGS OFF)
    message("-- Compiling with CUDA support")
  else(CUDA_FOUND)
    add_definitions(-DNO_CUDA)
    message("-- Cannot find CUDA libraries. Compiling without them." )
  endif(CUDA_FOUND)
else(CUDA)
    message("-- Forcing compilation without CUDA.")
    add_definitions(-DNO_CUDA)
endif(CUDA)

find_package(Boost COMPONENTS system filesystem program_options timer iostreams python thread)
if(Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS})
    set(EXT_LIBS ${EXT_LIBS} ${Boost_LIBRARIES})
else(Boost_FOUND)
    message(SEND_ERROR "Cannot find Boost libraries. Terminating." )
endif(Boost_FOUND)

find_package(PythonLibs 2.7)
if(PYTHONLIBS_FOUND)
  message("-- Found Python" )
  include_directories(${PYTHON_INCLUDE_DIRS})
  set(EXT_LIBS ${EXT_LIBS} ${PYTHON_LIBRARIES})
else(PYTHONLIBS_FOUND)
  message("-- Cannot find python libraries. Building without bindings." )
endif(PYTHONLIBS_FOUND)

find_package (ZLIB)
if (ZLIB_FOUND)
  include_directories(${ZLIB_INCLUDE_DIRS})
  set(EXT_LIBS ${EXT_LIBS} ${ZLIB_LIBRARIES})
endif (ZLIB_FOUND)

IF(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/.git)
  FIND_PACKAGE(Git)
  IF(GIT_FOUND)
    EXECUTE_PROCESS(
	    COMMAND ${GIT_EXECUTABLE} rev-parse --short HEAD
	    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
	    OUTPUT_VARIABLE "AMUNMT_BUILD_VERSION"
	    ERROR_QUIET
	    OUTPUT_STRIP_TRAILING_WHITESPACE)
    MESSAGE( STATUS "Git version: ${AMUNMT_BUILD_VERSION}" )
  ELSE(GIT_FOUND)
    SET(AMUNMT_BUILD_VERSION 0)
  ENDIF(GIT_FOUND)
ENDIF(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/.git)
SET(GIT_SHA1 ${AMUNMT_BUILD_VERSION})

include_directories(${amunmt_SOURCE_DIR}/src)
add_subdirectory(src)
