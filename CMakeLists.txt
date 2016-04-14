cmake_minimum_required(VERSION 3.1.0)
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

project(amunn CXX)
SET(CMAKE_CXX_FLAGS " -std=c++11 -g -O3 -funroll-loops -Wno-unused-result -Wno-deprecated")
#SET(CUDA_PROPAGATE_HOST_FLAGS OFF)
SET(CUDA_NVCC_FLAGS " -std=c++11 -g -O3 -arch=sm_35 -lineinfo --use_fast_math")
#SET(CUDA_VERBOSE_BUILD ON)

include_directories(${amunn_SOURCE_DIR})
set(EXT_LIBS)

find_package(CUDA REQUIRED)

find_package(Boost COMPONENTS system filesystem program_options timer)
if(Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS})
    set(EXT_LIBS ${EXT_LIBS} ${Boost_LIBRARIES})
else(Boost_FOUND)
    message(SEND_ERROR "Cannot find Boost libraries. Terminating." )
endif(Boost_FOUND)

include_directories($amunn_SOURCE_DIR}/src)
add_subdirectory(src)
