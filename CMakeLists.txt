cmake_minimum_required(VERSION 3.5.1)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

project(amunmt C CXX)

set(CMAKE_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR})

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

option(CPU "Select to compile with CPU support" ON)
option(CUDA "Select to compile CUDA support by default if available" ON)
option(FPGA "Select to compile with FPGA support" OFF)
option(MARIAN "Select to compile with Marian library" ON)
option(MARIAN_LIBRARY_ONLY "Automatically set when building amunmt. Don't touch this." ON)

if(CPU)
    add_definitions(-DHAS_CPU)
endif(CPU)

if(FPGA)
    add_definitions(-DHAS_FPGA)
endif(FPGA)
  
if(CUDA)
  find_package(CUDA)
  if(CUDA_FOUND)
    LIST(APPEND CUDA_NVCC_FLAGS --default-stream per-thread; -std=c++11; -O3; -arch=sm_35; -lineinfo; --use_fast_math;)
    # add_definitions(-DCUDA_API_PER_THREAD_DEFAULT_STREAM)
    add_definitions(-DCUDA)
    SET(CUDA_PROPAGATE_HOST_FLAGS OFF)
    message("-- Compiling with CUDA support")
    
    if (MARIAN)
      execute_process(COMMAND git submodule update --init --recursive --remote --no-fetch
                      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    endif(MARIAN)

  else(CUDA_FOUND)
    add_definitions(-DNO_CUDA)
    message("-- Cannot find CUDA libraries. Compiling without them." )
  endif(CUDA_FOUND)
else(CUDA)
    message("-- Forcing compilation without CUDA.")
    add_definitions(-DNO_CUDA)
endif(CUDA)

if(PYTHON_VERSION)
  STRING(REPLACE "." "" PYTHON_VERSION_NORMALIZED ${PYTHON_VERSION})
else(PYTHON_VERSION)
  set(PYTHON_VERSION "2.7")
endif(PYTHON_VERSION)

if (UNIX AND NOT APPLE)

    if(PYTHON_VERSION_NORMALIZED)
      set(BOOST_PYTHON_VERSION_NAME "python-py${PYTHON_VERSION_NORMALIZED}")
    else(PYTHON_VERSION_NORMALIZED)
      set(BOOST_PYTHON_VERSION_NAME "python")
    endif(PYTHON_VERSION_NORMALIZED)

    find_package(Boost COMPONENTS system filesystem program_options timer iostreams ${BOOST_PYTHON_VERSION_NAME} thread)
elseif (APPLE)

    # On macOS, boost-python component is named python if compiled against python2.7 or
    # python3 if compiled against python3.x
    if(NOT PYTHON_VERSION_NORMALIZED EQUAL 27)
      set(BOOST_PYTHON_VERSION_NAME "python${PYTHON_VERSION_NORMALIZED}")
    else(NOT PYTHON_VERSION_NORMALIZED EQUAL 27)
      set(BOOST_PYTHON_VERSION_NAME "python")
    endif(NOT PYTHON_VERSION_NORMALIZED EQUAL 27)

    find_package(Boost COMPONENTS system filesystem program_options timer iostreams ${BOOST_PYTHON_VERSION_NAME} thread)
endif (UNIX AND NOT APPLE)

if(Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS})
    set(EXT_LIBS ${EXT_LIBS} ${Boost_LIBRARIES})
else(Boost_FOUND)
    message(SEND_ERROR "Cannot find Boost libraries. Terminating." )
endif(Boost_FOUND)

if (FPGA)
  find_package( OpenCL REQUIRED )
  if(OpenCL_FOUND)
    set(EXT_LIBS ${EXT_LIBS} ${OpenCL_LIBRARIES})
  endif(OpenCL_FOUND)
endif(FPGA)

find_package(PythonLibs ${PYTHON_VERSION} REQUIRED)
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
