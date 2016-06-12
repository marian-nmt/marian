cmake_minimum_required(VERSION 3.5.1)
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

project(amunn CXX)
SET(CMAKE_CXX_FLAGS " -std=c++11 -fopenmp -Ofast -funroll-loops -Wno-unused-result -Wno-deprecated")

include_directories(${amunmt_SOURCE_DIR})

find_package(Boost COMPONENTS system filesystem program_options timer iostreams)
if(Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS})
    set(EXT_LIBS ${EXT_LIBS} ${Boost_LIBRARIES})
else(Boost_FOUND)
    message(SEND_ERROR "Cannot find Boost libraries. Terminating." )
endif(Boost_FOUND)

find_package(OpenBLAS)
if(OPENBLAS_FOUND)
    message(${OPENBLAS})
    set(EXT_LIBS ${EXT_LIBS} ${OPENBLAS})
else(OPENBLAS_FOUND)
    message(SEND_ERROR "Cannot find OpenBLAS libraries. Terminating." )
endif(OPENBLAS_FOUND)

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
