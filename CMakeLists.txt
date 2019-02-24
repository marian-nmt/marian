cmake_minimum_required(VERSION 3.5.1)
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

if (POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW) # CMake 3.12
endif ()


project(marian CXX C)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(BUILD_ARCH native CACHE STRING "Compile for this CPU architecture.")

# Custom CMake options
option(COMPILE_CPU "Compile CPU version" ON)
option(COMPILE_CUDA "Compile GPU version" ON)
option(USE_SENTENCEPIECE "Download and compile SentencePiece" OFF)
option(USE_STATIC_LIBS "Link statically against non-system libs" OFF)
option(USE_CUDNN "Use CUDNN library" OFF)
option(USE_NCCL "Use NCCL library" ON)
option(USE_MPI "Use MPI library" OFF)
option(COMPILE_EXAMPLES "Compile examples" OFF)
option(COMPILE_TESTS "Compile tests" OFF)
option(COMPILE_SERVER "Compile marian-server" ON)

# Project versioning
find_package(Git QUIET)
include(GetVersionFromFile)

message(STATUS "Project name: ${PROJECT_NAME}")
message(STATUS "Project version: ${PROJECT_VERSION_STRING_FULL}")

execute_process(COMMAND git submodule update --init --recursive --no-fetch
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

# Set compilation flags
if(MSVC)
# These are used in src/CMakeLists.txt on a per-target basis
  list(APPEND ALL_WARNINGS /WX; /W4;)

  # Disabled bogus warnings for CPU intrincics:
  # C4310: cast truncates constant value
  # C4324: 'marian::cpu::int16::`anonymous-namespace'::ScatterPut': structure was padded due to alignment specifier
  set(DISABLE_GLOBALLY "/wd\"4310\" /wd\"4324\"")

  set(INTRINSICS "/arch:AVX")

  # Or maybe use these?
  # set(INTRINSICS "/arch:AVX2")
  # set(INTRINSICS "/arch:AVX512")

  set(CMAKE_CXX_FLAGS           "/EHsc /DWIN32 /D_WINDOWS /DUNICODE /D_UNICODE /D_CRT_NONSTDC_NO_WARNINGS /D_CRT_SECURE_NO_WARNINGS ${DISABLE_GLOBALLY}")
  set(CMAKE_CXX_FLAGS_RELEASE   "${CMAKE_CXX_FLAGS} /MT /O2 ${INTRINSICS} /Zi /MP /GL /DNDEBUG")
  set(CMAKE_CXX_FLAGS_DEBUG     "${CMAKE_CXX_FLAGS} /MTd /Od /Ob0 ${INTRINSICS} /RTC1 /Zi /D_DEBUG")

  # ignores warning LNK4049: locally defined symbol free imported - this comes from zlib
  set(CMAKE_EXE_LINKER_FLAGS    "${CMAKE_EXE_LINKER_FLAGS} /DEBUG /LTCG:incremental /INCREMENTAL:NO /NODEFAULTLIB:MSVCRT /ignore:4049")
  set(CMAKE_STATIC_LINKER_FLAGS "${CMAKE_STATIC_LINKER_FLAGS} /LTCG:incremental")

  find_library(SHLWAPI Shlwapi.lib)
  set(EXT_LIBS ${EXT_LIBS} SHLWAPI)
else()

# Detect support CPU instrinsics for the current platform. This will
# only by used with BUILD_ARCH=native. For overridden BUILD_ARCH we
# minimally use -msse4.1. This seems to work with MKL.
set(INTRINSICS "")
if(BUILD_ARCH STREQUAL "native")
  message(STATUS "Checking support for CPU intrinsics")
  include(FindSSE)
  if(SSE2_FOUND)
    message(STATUS "SSE2 support found")
    set(INTRINSICS "${INTRINSICS} -msse2")
  endif(SSE2_FOUND)
  if(SSE3_FOUND)
    message(STATUS "SSE3 support found")
    set(INTRINSICS "${INTRINSICS} -msse3")
  endif(SSE3_FOUND)
  if(SSE4_1_FOUND)
    message(STATUS "SSE4.1 support found")
    set(INTRINSICS "${INTRINSICS} -msse4.1")
  endif(SSE4_1_FOUND)
  if(AVX_FOUND)
    message(STATUS "AVX support found")
    set(INTRINSICS "${INTRINSICS} -mavx")
  endif(AVX_FOUND)
  if(AVX2_FOUND)
    message(STATUS "AVX2 support found")
    set(INTRINSICS "${INTRINSICS} -mavx2")
  endif(AVX2_FOUND)
else()
  set(INTRINSICS "-msse4.1")
endif()

set(DISABLE_GLOBALLY "-Wno-unused-result")

# These are used in src/CMakeLists.txt on a per-target basis
list(APPEND ALL_WARNINGS -Wall; -Werror; -Wno-unused-result; -Wno-deprecated; -Wno-pragmas; -Wno-unused-parameter; -Wextra; -Wno-unused-function;
                         -Wno-unused-value; -Wno-unknown-pragmas; -Wno-sign-compare; -Wno-missing-field-initializers;)

  # This warning does not exist prior to gcc 5.0
  if(CMAKE_COMPILER_IS_GNUCC AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 5.0)
    list(APPEND ALL_WARNINGS -Wsuggest-override)
  endif()

  set(CMAKE_CXX_FLAGS                 "-std=c++11 -O3 -Ofast -m64 -pthread -march=${BUILD_ARCH} ${INTRINSICS} -Wl,--no-as-needed -funroll-loops -ffinite-math-only -fPIC ${DISABLE_GLOBALLY}")
  set(CMAKE_CXX_FLAGS_RELEASE         "${CMAKE_CXX_FLAGS} -g -rdynamic")
  set(CMAKE_CXX_FLAGS_DEBUG           "-std=c++11 -g -rdynamic -O0 -pthread -Wl,--no-as-needed -fPIC -Wno-unused-result -Wno-deprecated -Wno-pragmas")
  set(CMAKE_CXX_FLAGS_SLIM            "${CMAKE_CXX_FLAGS} -DNDEBUG")
  set(CMAKE_CXX_FLAGS_RELWITHDEBINFO  "${CMAKE_CXX_FLAGS} -g -rdynamic")
  set(CMAKE_CXX_FLAGS_PROFILE         "${CMAKE_CXX_FLAGS_RELEASE} -pg -g -rdynamic")
  set(CMAKE_CXX_FLAGS_PROFGEN         "${CMAKE_CXX_FLAGS_RELEASE} -fprofile-generate -fprofile-correction")
  set(CMAKE_CXX_FLAGS_PROFUSE         "${CMAKE_CXX_FLAGS_RELEASE} -fprofile-use -fprofile-correction")
  endif()

# Downloading SentencePiece if requested and set to compile with it.
# Requires all the dependencies imposed by SentencePiece
if(USE_SENTENCEPIECE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_SENTENCEPIECE")
  LIST(APPEND CUDA_NVCC_FLAGS -DUSE_SENTENCEPIECE; )
  set(EXT_LIBS ${EXT_LIBS} sentencepiece sentencepiece_train)
endif()


# Find packages
set(EXT_LIBS ${EXT_LIBS} ${CMAKE_DL_LIBS})

if(COMPILE_CUDA)

if(USE_STATIC_LIBS)
  # link statically to stdlib libraries
  set(CMAKE_EXE_LINKER_FLAGS "-static-libgcc -static-libstdc++")

  # look for libraries that have .a suffix
  set(_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES})
  if(WIN32)
    list(INSERT CMAKE_FIND_LIBRARY_SUFFIXES 0 .lib .a)
  else()
    set(CMAKE_FIND_LIBRARY_SUFFIXES .a _static.a)
  endif()
endif()

find_package(CUDA "8.0")
if(CUDA_FOUND)
  if(USE_STATIC_LIBS)
    find_library(CUDA_culibos_LIBRARY NAMES culibos PATHS ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
    set(EXT_LIBS ${EXT_LIBS} ${CUDA_curand_LIBRARY} ${CUDA_cusparse_LIBRARY} ${CUDA_culibos_LIBRARY} ${CUDA_CUBLAS_LIBRARIES})
    message(STATUS "Found CUDA libraries: ${CUDA_curand_LIBRARY} ${CUDA_cusparse_LIBRARY} ${CUDA_culibos_LIBRARY} ${CUDA_CUBLAS_LIBRARIES}")
  else(USE_STATIC_LIBS)
    set(EXT_LIBS ${EXT_LIBS} ${CUDA_curand_LIBRARY} ${CUDA_cusparse_LIBRARY} ${CUDA_CUBLAS_LIBRARIES})
    message(STATUS "Found CUDA libraries: ${CUDA_curand_LIBRARY} ${CUDA_cusparse_LIBRARY} ${CUDA_CUBLAS_LIBRARIES}")
endif(USE_STATIC_LIBS)

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

  if(MSVC)
    list(APPEND CUDA_NVCC_FLAGS -DBOOST_PP_VARIADICS=0; )
  endif()

  if(USE_NCCL)
    add_library(nccl STATIC IMPORTED)
    set(EXT_LIBS ${EXT_LIBS} nccl)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_NCCL")
    LIST(APPEND CUDA_NVCC_FLAGS -DUSE_NCCL; )
  endif(USE_NCCL)

if(USE_STATIC_LIBS)
  set(CMAKE_FIND_LIBRARY_SUFFIXES ${_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES})
endif()

else(CUDA_FOUND)
  message(FATAL_ERROR "CUDA has not been found, set -DCOMPILE_CUDA=off to avoid this check and to compile the CPU version only")
endif(CUDA_FOUND)

else(COMPILE_CUDA)
  message(WARNING "COMPILE_CUDA=off : Building only CPU version")
endif(COMPILE_CUDA)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  list(APPEND CUDA_NVCC_FLAGS --default-stream per-thread; -O0; -g; -arch=sm_30; -gencode=arch=compute_30,code=sm_30; -gencode=arch=compute_50,code=sm_50; -gencode=arch=compute_52,code=sm_52; -gencode=arch=compute_60,code=sm_60; -gencode=arch=compute_61,code=sm_61; -gencode=arch=compute_61,code=compute_61 ;)
else(CMAKE_BUILD_TYPE STREQUAL "Debug")
  list(APPEND CUDA_NVCC_FLAGS --default-stream per-thread; -O3; -g; --use_fast_math; -arch=sm_30; -gencode=arch=compute_30,code=sm_30; -gencode=arch=compute_50,code=sm_50; -gencode=arch=compute_52,code=sm_52; -gencode=arch=compute_60,code=sm_60; -gencode=arch=compute_61,code=sm_61; -gencode=arch=compute_61,code=compute_61 ;)
endif(CMAKE_BUILD_TYPE STREQUAL "Debug")
if(NOT MSVC)
  # @TODO: add warnings here too
  list(APPEND CUDA_NVCC_FLAGS -ccbin ${CMAKE_C_COMPILER}; -std=c++11; -Xcompiler\ -fPIC; -Xcompiler\ -Wno-unused-result; -Xcompiler\ -Wno-deprecated; -Xcompiler\ -Wno-pragmas; -Xcompiler\ -Wno-unused-value; -Xcompiler\ -Werror;)
else()
  list(APPEND CUDA_NVCC_FLAGS -Xcompiler\ /FS; )
endif()

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

if(NOT WIN32)
  find_package(Tcmalloc)
  if(Tcmalloc_FOUND)
    include_directories(${Tcmalloc_INCLUDE_DIR})
    set(EXT_LIBS ${EXT_LIBS} ${Tcmalloc_LIBRARIES})
  else(Tcmalloc_FOUND)
    message(WARNING "Cannot find TCMalloc library. Continuing.")
  endif(Tcmalloc_FOUND)
endif()

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

set(BOOST_COMPONENTS timer iostreams filesystem system chrono)
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.9)
  add_definitions(-DUSE_BOOST_REGEX=1)
  set(BOOST_COMPONENTS ${BOOST_COMPONENTS} regex)
  message(STATUS "Using boost::regex")
else()
  message(STATUS "Using std::regex")
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
  set(EXT_LIBS ${EXT_LIBS} ${ZLIB_LIBRARIES}) # hack for static compilation
else(Boost_FOUND)
  message(SEND_ERROR "Cannot find Boost libraries. Terminating.")
endif(Boost_FOUND)

if(COMPILE_TESTS)
  enable_testing()
endif(COMPILE_TESTS)

if(COMPILE_EXAMPLES)
  add_definitions(-DCOMPILE_EXAMPLES=1)
endif(COMPILE_EXAMPLES)

# Generate project_version.h to reflect our version number
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/src/common/project_version.h.in
               ${CMAKE_CURRENT_SOURCE_DIR}/src/common/project_version.h @ONLY)

# Compile source files
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

