cmake_minimum_required(VERSION 3.5.1)
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
set(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH} /usr/local/cuda-8.0/lib64)


project(marian CXX C)

# Custom CMake options
option(COMPILE_EXAMPLES "Compile examples" OFF)
option(COMPILE_TESTS "Compile tests" OFF)
option(COMPILE_SERVER "Compile marian-server" ON)
option(USE_CUDNN "Use CUDNN library" OFF)

# Project versioning
find_package(Git QUIET)
include(GetVersionFromFile)

message(STATUS "Project name: ${PROJECT_NAME}")
message(STATUS "Project version: ${PROJECT_VERSION_STRING_FULL}")


# Set compilation flags
set(CMAKE_CXX_FLAGS_RELEASE " -std=c++11 -O3 -Ofast -m64 -march=native -funroll-loops -ffinite-math-only -fPIC -Wno-unused-result -Wno-deprecated -Wno-deprecated-gpu-targets")
set(CMAKE_CXX_FLAGS_DEBUG " -std=c++11 -g -O0 -fPIC -Wno-unused-result -Wno-deprecated -Wno-deprecated-gpu-targets")
set(CMAKE_CXX_FLAGS_PROFILE "${CMAKE_CXX_FLAGS_RELEASE} -g -pg")
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS_RELEASE})

# Find packages
find_package(CUDA "8.0" REQUIRED)
if(CUDA_FOUND)
  set(EXT_LIBS ${EXT_LIBS} ${CUDA_curand_LIBRARY} ${CUDA_cusparse_LIBRARY})
endif(CUDA_FOUND)

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  list(APPEND CUDA_NVCC_FLAGS -std=c++11; --default-stream per-thread; -O0; -g; -Xcompiler '-fPIC'; -arch=sm_30; -gencode=arch=compute_30,code=sm_30; -gencode=arch=compute_50,code=sm_50; -gencode=arch=compute_52,code=sm_52; -gencode=arch=compute_60,code=sm_60; -gencode=arch=compute_61,code=sm_61; -gencode=arch=compute_61,code=compute_61 ;)
else(CMAKE_BUILD_TYPE STREQUAL "Debug")
  list(APPEND CUDA_NVCC_FLAGS -std=c++11; --default-stream per-thread; -O3; --use_fast_math; -Xcompiler '-fPIC'; -arch=sm_30; -gencode=arch=compute_30,code=sm_30; -gencode=arch=compute_50,code=sm_50; -gencode=arch=compute_52,code=sm_52; -gencode=arch=compute_60,code=sm_60; -gencode=arch=compute_61,code=sm_61; -gencode=arch=compute_61,code=compute_61 ;)
endif(CMAKE_BUILD_TYPE STREQUAL "Debug")

list(REMOVE_DUPLICATES CUDA_NVCC_FLAGS)
set(CUDA_PROPAGATE_HOST_FLAGS OFF)

if(USE_CUDNN)
  find_package(CUDNN "7.0")
  if(CUDNN_FOUND)
    include_directories(${CUDNN_INCLUDE_DIRS})
    set(EXT_LIBS ${EXT_LIBS} ${CUDNN_LIBRARIES})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCUDNN")
    LIST(APPEND CUDA_NVCC_FLAGS -DCUDNN; )
  endif(CUDNN_FOUND)
endif(USE_CUDNN)

find_package(Tcmalloc)
if(Tcmalloc_FOUND)
  include_directories(${Tcmalloc_INCLUDE_DIR})
  set(EXT_LIBS ${EXT_LIBS} ${Tcmalloc_LIBRARIES})
else(Tcmalloc_FOUND)
  message(WARNING "Cannot find TCMalloc library. Continuing.")
endif(Tcmalloc_FOUND)

find_package(ZLIB)
if(ZLIB_FOUND)
  include_directories(${ZLIB_INCLUDE_DIRS})
  set(EXT_LIBS ${EXT_LIBS} ${ZLIB_LIBRARIES})
else(ZLIB_FOUND)
  message(SEND_ERROR "Cannot find zlib.")
endif(ZLIB_FOUND)

set(BOOST_COMPONENTS system timer iostreams filesystem chrono program_options thread)

find_package(PythonLibs 2.7)
if(PYTHONLIBS_FOUND)
  set(BOOST_COMPONENTS ${BOOST_COMPONENTS} python)
  message(STATUS "Found Python")
  include_directories(${PYTHON_INCLUDE_DIRS})
  set(EXT_LIBS ${EXT_LIBS} ${PYTHON_LIBRARIES})
else(PYTHONLIBS_FOUND)
  message(WARNING "Cannot find python libraries. Building without bindings.")
endif(PYTHONLIBS_FOUND)

if(COMPILE_SERVER)
  find_package(OpenSSL)
  if(OpenSSL_FOUND)
    message(STATUS "Found OpenSSL")
    include_directories(${OPENSSL_INCLUDE_DIR})
    set(EXT_LIBS ${EXT_LIBS} ${OPENSSL_CRYPTO_LIBRARY})
  else(OpenSSL_FOUND)
    message(WARNING "Cannot find OpenSSL library. Not compiling server.")
    set(COMPILE_SERVER "off")
  endif(OpenSSL_FOUND)
endif(COMPILE_SERVER)

find_package(Boost COMPONENTS ${BOOST_COMPONENTS})
if(Boost_FOUND)
  include_directories(${Boost_INCLUDE_DIRS})
  set(EXT_LIBS ${EXT_LIBS} ${Boost_LIBRARIES})
else(Boost_FOUND)
  message(SEND_ERROR "Cannot find Boost libraries. Terminating.")
endif(Boost_FOUND)

if(COMPILE_TESTS)
  enable_testing()
endif(COMPILE_TESTS)


# Compile source files
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/src/common/version.h.in
               ${CMAKE_CURRENT_SOURCE_DIR}/src/common/version.h @ONLY)

include_directories(${marian_SOURCE_DIR}/src)
add_subdirectory(src)


# Add a target to generate API documentation with Doxygen
find_package(Doxygen)
if(DOXYGEN_FOUND)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in
           ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)
  add_custom_target(doc
    ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Generating API documentation with Doxygen" VERBATIM
  )
endif(DOXYGEN_FOUND)
