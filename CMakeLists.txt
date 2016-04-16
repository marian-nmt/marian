cmake_minimum_required(VERSION 3.5.1)
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

project(amunn CXX)
SET(CMAKE_CXX_FLAGS " -std=c++11 -g -O3 -funroll-loops -Wno-unused-result -Wno-deprecated")
LIST(APPEND CUDA_NVCC_FLAGS --default-stream per-thread; -std=c++11; -g; -O3; -arch=sm_35; -lineinfo; --use_fast_math;)
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

### KenLM stuff - BEGIN ###
set(EXT_LIBS ${EXT_LIBS} /home/marcinj/Badania/kenlm/build/lib/libkenlm.a)
set(EXT_LIBS ${EXT_LIBS} /home/marcinj/Badania/kenlm/build/lib/libkenlm_util.a)
include_directories(/home/marcinj/Badania/kenlm)
add_definitions(-DKENLM_MAX_ORDER=9)

#find_package (OpenMP)
#if (OPENMP_FOUND)
#    LIST(APPEND CUDA_NVCC_FLAGS  -Xcompiler -fopenmp)
#    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
#endif (OPENMP_FOUND)

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
