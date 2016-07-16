cmake_minimum_required(VERSION 3.5.1)
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

project(amunn CXX)
SET(CMAKE_CXX_FLAGS " -std=c++11 -fPIC -Ofast -funroll-loops -Wno-unused-result -Wno-deprecated")
#SET(CMAKE_CXX_FLAGS " -std=c++11 -fPIC -fopenmp -g -Og -DDEBUG -Bdynamic -lprofiler -ltcmalloc_and_profiler -funroll-loops -Wno-unused-result -Wno-deprecated")
# execute this before profiling: export LD_PRELOAD=/usr/lib/libtcmalloc_and_profiler.so

include_directories(${amunmt_SOURCE_DIR})

find_package(Threads REQUIRED)
if(CMAKE_THREAD_LIBS_INIT)
  SET(EXT_LIBS ${EXT_LIBS} ${CMAKE_THREAD_LIBS_INIT})
endif()

find_package(Boost COMPONENTS system filesystem program_options timer iostreams python thread)
if(Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS})
    set(EXT_LIBS ${EXT_LIBS} ${Boost_LIBRARIES})
else(Boost_FOUND)
    message(SEND_ERROR "Cannot find Boost libraries. Terminating." )
endif(Boost_FOUND)

find_package(OpenBLAS)
if(OPENBLAS_FOUND)
    set(EXT_LIBS ${EXT_LIBS} ${OPENBLAS})
else(OPENBLAS_FOUND)
    message(SEND_ERROR "Cannot find OpenBLAS libraries. Terminating." )
endif(OPENBLAS_FOUND)

find_package(PythonLibs 2.7 REQUIRED)
if(PYTHONLIBS_FOUND)
    message("-- Found Python" )
    include_directories(${PYTHON_INCLUDE_DIRS})
    set(EXT_LIBS ${EXT_LIBS} ${PYTHON_LIBRARIES})
else(PYTHONLIBS_FOUND)
    message(SEND_ERROR "Cannot find python libraries. Terminating." )
endif(PYTHONLIBS_FOUND)

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

include_directories($amunmt_SOURCE_DIR}/src)
add_subdirectory(src)
