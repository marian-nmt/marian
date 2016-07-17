cmake_minimum_required(VERSION 3.5.1)
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

project(amunn CXX)
SET(CMAKE_CXX_FLAGS_RELEASE " -std=c++14 -march=native -fPIC -Ofast -funroll-loops -ffinite-math-only -Wno-unused-result -Wno-deprecated")
SET(CMAKE_CXX_FLAGS_DEBUG " -std=c++14 -fPIC -g -Og")
SET(CMAKE_CXX_FLAGS_PROFILE " -std=c++14 -march=native -fPIC -g -lprofiler -DPROFILER_MODE -ltcmalloc_and_profiler -Ofast -funroll-loops -ffinite-math-only -Wno-unused-result -Wno-deprecated")
# execute this before profiling: export LD_PRELOAD=/usr/lib/libtcmalloc_and_profiler.so

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING
      "Choose the type of build, options are: Debug Release Profile" FORCE)
endif(NOT CMAKE_BUILD_TYPE)

if(CMAKE_BUILD_TYPE STREQUAL "Profile" OR CMAKE_BUILD_TYPE STREQUAL "profile")
find_package(GooglePerfTools REQUIRED)
if(GooglePerfTools_FOUND)
  SET(EXT_LIBS ${EXT_LIBS} ${GooglePerfTools_LIBRARIES})
endif()
endif()

include_directories(${amunmt_SOURCE_DIR})

find_package(Threads REQUIRED)
if(CMAKE_THREAD_LIBS_INIT)
  SET(EXT_LIBS ${EXT_LIBS} ${CMAKE_THREAD_LIBS_INIT})
endif()

find_package(Tcmalloc)
if(Tcmalloc_FOUND)
  set(EXT_LIBS ${EXT_LIBS} ${Tcmalloc_LIBRARIES})
endif (Tcmalloc_FOUND)

find_package(Boost COMPONENTS system filesystem program_options timer iostreams python thread)
if(Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS})
    set(EXT_LIBS ${EXT_LIBS} ${Boost_LIBRARIES})
else(Boost_FOUND)
    message(SEND_ERROR "Cannot find Boost libraries. Terminating." )
endif(Boost_FOUND)

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
