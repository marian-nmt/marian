cmake_minimum_required(VERSION 3.5.1)
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

project(marian CXX)
SET(CMAKE_CXX_FLAGS " -std=c++11 -g -O3 -funroll-loops -Wno-unused-result -Wno-deprecated")
LIST(APPEND CUDA_NVCC_FLAGS --default-stream per-thread; -std=c++11; -g; -O3; -arch=sm_35; -lineinfo; --use_fast_math; -Xcompiler '-fPIC')
add_definitions(-DCUDA_API_PER_THREAD_DEFAULT_STREAM)
SET(CUDA_PROPAGATE_HOST_FLAGS OFF)

include_directories(${amunn_SOURCE_DIR})
find_package(CUDA REQUIRED)

find_package(Boost COMPONENTS system timer)
if(Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS})
    set(EXT_LIBS ${EXT_LIBS} ${Boost_LIBRARIES})
else(Boost_FOUND)
    message(SEND_ERROR "Cannot find Boost libraries. Terminating." )
endif(Boost_FOUND)

include_directories(${marian_SOURCE_DIR}/src)
add_subdirectory(src)
