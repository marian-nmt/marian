cmake_minimum_required(VERSION 3.5.1)
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
set(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH} /usr/local/cuda-8.0/lib64)


project(marian CXX C)

# Custom CMake options
option(COMPILE_EXAMPLES "Compile examples" OFF)
option(COMPILE_TESTS "Compile tests" OFF)
option(COMPILE_SERVER "Compile marian-server" ON)
option(COMPILE_CPU "Compile CPU version" ON)
option(COMPILE_CUDA "Compile GPU version" ON)
option(USE_STATIC_LIBS "Compile GPU version" OFF)
option(USE_CUDNN "Use CUDNN library" OFF)
option(USE_MPI "Use MPI library" OFF)

# Project versioning
find_package(Git QUIET)
include(GetVersionFromFile)

message(STATUS "Project name: ${PROJECT_NAME}")
message(STATUS "Project version: ${PROJECT_VERSION_STRING_FULL}")

# Set compilation flags
set(CMAKE_CXX_FLAGS_RELEASE " -std=c++11 -O3 -Ofast -m64 -pthread -march=native -Wl,--no-as-needed -funroll-loops -ffinite-math-only -fPIC -Wno-unused-result -Wno-deprecated -Wno-deprecated-gpu-targets")
set(CMAKE_CXX_FLAGS_DEBUG " -std=c++11 -g -O0 -pthread -fPIC -Wno-unused-result -Wno-deprecated -Wno-deprecated-gpu-targets")
set(CMAKE_CXX_FLAGS_ST "${CMAKE_CXX_FLAGS_RELEASE} -DNDEBUG")
set(CMAKE_CXX_FLAGS_PROFILE "${CMAKE_CXX_FLAGS_RELEASE}  -pg -g")
set(CMAKE_CXX_FLAGS_PROFGEN "${CMAKE_CXX_FLAGS_RELEASE} -fprofile-generate -fprofile-correction")
set(CMAKE_CXX_FLAGS_PROFUSE "${CMAKE_CXX_FLAGS_RELEASE} -fprofile-use -fprofile-correction")
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS_RELEASE})

# Find packages

set(EXT_LIBS ${EXT_LIBS} ${CMAKE_DL_LIBS})

if(COMPILE_CUDA)
find_package(CUDA "8.0" REQUIRED)
if(CUDA_FOUND)
  set(EXT_LIBS ${EXT_LIBS} ${CUDA_curand_LIBRARY} ${CUDA_cusparse_LIBRARY})

  if(USE_CUDNN)
    find_package(CUDNN "7.0")
    if(CUDNN_FOUND)
      include_directories(${CUDNN_INCLUDE_DIRS})
      set(EXT_LIBS ${EXT_LIBS} ${CUDNN_LIBRARIES})
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCUDNN")
      LIST(APPEND CUDA_NVCC_FLAGS -DCUDNN; )
    endif(CUDNN_FOUND)
  endif(USE_CUDNN)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCUDA_FOUND")
  list(APPEND CUDA_NVCC_FLAGS -DCUDA_FOUND; )
else(CUDA_FOUND)
  message(WARNING "Cannot find CUDA compiling CPU version only")
endif(CUDA_FOUND)

else(COMPILE_CUDA)
  message(WARNING "COMPILE_CUDA=off : Building only CPU version")
endif(COMPILE_CUDA)

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  list(APPEND CUDA_NVCC_FLAGS -std=c++11; --default-stream per-thread; -O0; -g; -Xcompiler '-fPIC'; -arch=sm_30; -gencode=arch=compute_30,code=sm_30; -gencode=arch=compute_50,code=sm_50; -gencode=arch=compute_52,code=sm_52; -gencode=arch=compute_60,code=sm_60; -gencode=arch=compute_61,code=sm_61; -gencode=arch=compute_61,code=compute_61 ;)
else(CMAKE_BUILD_TYPE STREQUAL "Debug")
  list(APPEND CUDA_NVCC_FLAGS -std=c++11; --default-stream per-thread; -O3; -g; --use_fast_math; -Xcompiler '-fPIC'; -arch=sm_30; -gencode=arch=compute_30,code=sm_30; -gencode=arch=compute_50,code=sm_50; -gencode=arch=compute_52,code=sm_52; -gencode=arch=compute_60,code=sm_60; -gencode=arch=compute_61,code=sm_61; -gencode=arch=compute_61,code=compute_61 ;)
endif(CMAKE_BUILD_TYPE STREQUAL "Debug")

list(REMOVE_DUPLICATES CUDA_NVCC_FLAGS)
set(CUDA_PROPAGATE_HOST_FLAGS OFF)

if(USE_STATIC_LIBS)
  set(_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES})
  if(WIN32)
    list(INSERT CMAKE_FIND_LIBRARY_SUFFIXES 0 .lib .a)
  else()
    set(CMAKE_FIND_LIBRARY_SUFFIXES .a)
  endif()
endif()

find_package(Tcmalloc)
if(Tcmalloc_FOUND)
  include_directories(${Tcmalloc_INCLUDE_DIR})
  set(EXT_LIBS ${EXT_LIBS} ${Tcmalloc_LIBRARIES})
else(Tcmalloc_FOUND)
  message(WARNING "Cannot find TCMalloc library. Continuing.")
endif(Tcmalloc_FOUND)

if(USE_MPI)
  find_package(MPI 2.0)
  if(MPI_FOUND)
    include_directories(${MPI_INCLUDE_PATH})
    set(EXT_LIBS ${EXT_LIBS} ${MPI_LIBRARIES})
    add_definitions(-DMPI_FOUND=1)
  endif(MPI_FOUND)
endif(USE_MPI)

if(COMPILE_CPU)
  find_package(MKL)
  if(MKL_FOUND)
    include_directories(${MKL_INCLUDE_DIR})
    set(EXT_LIBS ${EXT_LIBS} ${MKL_LIBRARIES})
    add_definitions(-DBLAS_FOUND=1 -DMKL_FOUND=1)
  else(MKL_FOUND)
    set(BLA_VENDOR "OpenBLAS")
    find_package(BLAS)
    if(BLAS_FOUND)
      include_directories(${BLAS_INCLUDE_DIR})
      set(EXT_LIBS ${EXT_LIBS} ${BLAS_LIBRARIES})
      add_definitions(-DBLAS_FOUND=1)
    endif(BLAS_FOUND)
  endif(MKL_FOUND)
endif(COMPILE_CPU)

set(BOOST_COMPONENTS system timer iostreams filesystem chrono program_options thread)

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.9)
  add_definitions(-DUSE_BOOST_REGEX=1)
  set(BOOST_COMPONENTS ${BOOST_COMPONENTS} regex)
  message("-- Using boost::regex")
else()
  message("-- Using std::regex")
endif()


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

find_package(ZLIB)
if(ZLIB_FOUND)
  include_directories(${ZLIB_INCLUDE_DIRS})
  set(EXT_LIBS ${EXT_LIBS} ${ZLIB_LIBRARIES})
else(ZLIB_FOUND)
  message(SEND_ERROR "Cannot find zlib.")
endif(ZLIB_FOUND)

if(USE_STATIC_LIBS)
  set(CMAKE_FIND_LIBRARY_SUFFIXES ${_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES})
endif()

if(USE_STATIC_LIBS)
  set(Boost_USE_STATIC_LIBS ON)
endif()

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

if(COMPILE_EXAMPLES)
  add_definitions(-DCOMPILE_EXAMPLES=1)
endif(COMPILE_EXAMPLES)

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
