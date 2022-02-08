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
option(COMPILE_EXAMPLES "Compile examples" OFF)
option(COMPILE_SERVER "Compile marian-server" OFF)
option(COMPILE_TESTS "Compile tests" OFF)
if(APPLE)
  option(USE_APPLE_ACCELERATE "Compile with Apple Accelerate" ON)
else(APPLE)
  option(USE_APPLE_ACCELERATE "Compile with Apple Accelerate" OFF)
endif(APPLE)
option(USE_CCACHE "Use ccache compiler cache (https://ccache.dev)" OFF)
option(USE_CUDNN "Use CUDNN library" OFF)
option(USE_DOXYGEN "Build documentation with Doxygen" ON)
option(USE_FBGEMM "Use FBGEMM" OFF)
option(USE_MKL "Compile with MKL support" ON)
option(USE_MPI "Use MPI library" OFF)
option(USE_NCCL "Use NCCL library" ON)
option(USE_SENTENCEPIECE "Download and compile SentencePiece" ON)
option(USE_STATIC_LIBS "Link statically against non-system libs" OFF)
option(GENERATE_MARIAN_INSTALL_TARGETS "Generate Marian install targets (requires CMake 3.12+)" OFF)
option(DETERMINISTIC "Try to make training results as deterministic as possible (e.g. for testing)" OFF)

# fbgemm and sentencepiece are both defined with "non-local" installation targets (the source projects don't define them,
# so we define them in src\3rd_party\CMakeLists.txt), but that isn't supported until CMake 3.12. Prior to CMake 3.12,
# targets could only be install(...)ed in the same CMakeLists.txt they were defined. We currently target CMake 3.5.1
# as our minimum supported CMake version, so this option exists to provide compatibility by disabling install targets.
if(GENERATE_MARIAN_INSTALL_TARGETS AND ${CMAKE_VERSION} VERSION_LESS "3.12")
  message(WARNING "Marian install targets cannot be generated on CMake <3.12.\
    Please upgrade your CMake version or set GENERATE_MARIAN_INSTALL_TARGETS=OFF to remove this warning. Disabling installation targets.")
  set(GENERATE_MARIAN_INSTALL_TARGETS OFF CACHE BOOL "Forcing disabled installation targets due to CMake <3.12." FORCE)
endif()

if(GENERATE_MARIAN_INSTALL_TARGETS)
  include(GNUInstallDirs)                 # This defines default values for installation directories (all platforms even if named GNU)
  include(InstallRequiredSystemLibraries) # Tell CMake that the `install` target needs to install required system libraries (eg: Windows SDK)
  include(CMakePackageConfigHelpers)      # Helper to create relocatable packages

  install(EXPORT marian-targets           # Installation target
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake)
endif(GENERATE_MARIAN_INSTALL_TARGETS)

# use ccache (https://ccache.dev) for faster compilation if requested and available
if(USE_CCACHE)
  find_program(CCACHE_PROGRAM ccache)
  if(CCACHE_PROGRAM)
    message(STATUS "Will be using ccache for faster repeat compilation (use cmake -DUSE_CCACHE=off to disable).")
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CCACHE_PROGRAM}")
  else(CCACHE_PROGRAM)
    message(WARNING "Compilation with ccache requested but no ccache found.")
  endif(CCACHE_PROGRAM)
endif(USE_CCACHE)

# Project versioning
find_package(Git QUIET)
include(GetVersionFromFile)

message(STATUS "Project name: ${PROJECT_NAME}")
message(STATUS "Project version: ${PROJECT_VERSION_STRING_FULL}")

execute_process(COMMAND git submodule update --init --recursive --no-fetch
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

# Note that with CMake MSVC build, the option CMAKE_BUILD_TYPE is automatically derived from the key
# 'configurationType' in CMakeSettings.json configurations
if(NOT CMAKE_BUILD_TYPE)
  message(WARNING "CMAKE_BUILD_TYPE not set; setting to Release")
  set(CMAKE_BUILD_TYPE "Release")
endif()

###############################################################################
# Set compilation flags
if(MSVC)
# These are used in src/CMakeLists.txt on a per-target basis
  list(APPEND ALL_WARNINGS /WX; /W4;)

  # Disabled bogus warnings for CPU intrinsics and Protobuf:
  # C4100: 'identifier' : unreferenced formal parameter
  # C4310: cast truncates constant value
  # C4324: 'marian::cpu::int16::`anonymous-namespace'::ScatterPut': structure was padded due to alignment specifier
  # C4702: unreachable code; note it is also disabled globally in the VS project file
  if(USE_SENTENCEPIECE)
    set(DISABLE_GLOBALLY "/wd\"4310\" /wd\"4324\" /wd\"4702\" /wd\"4100\"")
  else()
    set(DISABLE_GLOBALLY "/wd\"4310\" /wd\"4324\" /wd\"4702\"")
  endif()

  # set(INTRINSICS "/arch:AVX")
  add_definitions(-DUSE_SSE2=1)

  # Or maybe use these?
  set(INTRINSICS "/arch:AVX2")
  # set(INTRINSICS "/arch:AVX512")
  # /bigobj is necessary for expression_operators.cpp. See https://stackoverflow.com/questions/15110580/penalty-of-the-msvs-compiler-flag-bigobj
  set(CMAKE_CXX_FLAGS           "/EHsc /DWIN32 /D_WINDOWS /DUNICODE /D_UNICODE /D_CRT_NONSTDC_NO_WARNINGS /D_CRT_SECURE_NO_WARNINGS /bigobj ${DISABLE_GLOBALLY}")
  set(CMAKE_CXX_FLAGS_RELEASE   "${CMAKE_CXX_FLAGS} /MT /O2 ${INTRINSICS} /Zi /MP /GL /DNDEBUG")
  set(CMAKE_CXX_FLAGS_DEBUG     "${CMAKE_CXX_FLAGS} /MTd /Od /Ob0 ${INTRINSICS} /RTC1 /Zi /D_DEBUG")

  # ignores warning LNK4049: locally defined symbol free imported - this comes from zlib
  set(CMAKE_EXE_LINKER_FLAGS         "${CMAKE_EXE_LINKER_FLAGS} /DEBUG /LTCG:incremental /INCREMENTAL:NO /ignore:4049")
  set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS} /NODEFAULTLIB:MSVCRT")
  set(CMAKE_EXE_LINKER_FLAGS_DEBUG   "${CMAKE_EXE_LINKER_FLAGS} /NODEFAULTLIB:MSVCRTD")
  set(CMAKE_STATIC_LINKER_FLAGS      "${CMAKE_STATIC_LINKER_FLAGS} /LTCG:incremental")

  find_library(SHLWAPI Shlwapi.lib)
  set(EXT_LIBS ${EXT_LIBS} SHLWAPI)

  if(USE_FBGEMM)
    if(NOT USE_STATIC_LIBS) # FBGEMM on Windows can be compiled only statically via CMake
      message(FATAL_ERROR "FATAL ERROR: FBGEMM must be compiled statically on Windows, \
      add -DUSE_STATIC_LIBS=on to the cmake command")
    endif()
    set(EXT_LIBS ${EXT_LIBS} fbgemm)
    add_definitions(-DUSE_FBGEMM=1 -DFBGEMM_STATIC=1)
  endif(USE_FBGEMM)
else(MSVC)

  # Check we are using at least g++ 5.0
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 5.0)
    message(FATAL_ERROR "FATAL ERROR: Compiling Marian requires at least g++ 5.0, your version is ${CMAKE_CXX_COMPILER_VERSION}")
  endif()

  # Detect support CPU instrinsics for the current platform. This will
  # only by used with BUILD_ARCH=native. For overridden BUILD_ARCH we
  # force intrinsics as set in the options.
  set(INTRINSICS "")
  list(APPEND INTRINSICS_NVCC)

  option(COMPILE_SSE2   "Compile CPU code with SSE2 support"   ON)
  option(COMPILE_SSE3   "Compile CPU code with SSE3 support"   ON)
  option(COMPILE_SSE4_1 "Compile CPU code with SSE4.1 support" ON)
  option(COMPILE_SSE4_2 "Compile CPU code with SSE4.2 support" ON)
  option(COMPILE_AVX    "Compile CPU code with AVX support"    ON)
  option(COMPILE_AVX2   "Compile CPU code with AVX2 support"   ON)
  option(COMPILE_AVX512 "Compile CPU code with AVX512 support" ON)

  if(BUILD_ARCH STREQUAL "native")
    message(STATUS "Building with -march=native and intrinsics will be chosen automatically by the compiler to match the current machine.")
    message(STATUS "Checking support for CPU intrinsics")
    include(FindSSE)
    if(SSE2_FOUND AND NOT COMPILE_SSE2)
      message(WARNING "SSE2 enabled due to -march=native and -DCOMPILE_SSE2=${COMPILE_SSE2} is ignored.")
    endif(SSE2_FOUND AND NOT COMPILE_SSE2)
    if(SSE3_FOUND AND NOT COMPILE_SSE3)
      message(WARNING "SSE3 enabled due to -march=native and -DCOMPILE_SSE3=${COMPILE_SSE3} is ignored.")
    endif(SSE3_FOUND AND NOT COMPILE_SSE3)
    if(SSE4_1_FOUND AND NOT COMPILE_SSE4_1)
      message(WARNING "SSE4.1 enabled due to -march=native and -DCOMPILE_SSE4_1=${COMPILE_SSE4_1} is ignored.")
    endif(SSE4_1_FOUND AND NOT COMPILE_SSE4_1)
    if(SSE4_2_FOUND AND NOT COMPILE_SSE4_2)
      message(WARNING "SSE4.2 enabled due to -march=native and -DCOMPILE_SSE4_2=${COMPILE_SSE4_2} is ignored.")
    endif(SSE4_2_FOUND AND NOT COMPILE_SSE4_2)
    if(AVX_FOUND AND NOT COMPILE_AVX)
      message(WARNING "AVX enabled due to -march=native and -DCOMPILE_AVX=${COMPILE_AVX} is ignored.")
    endif(AVX_FOUND AND NOT COMPILE_AVX)
    if(AVX2_FOUND AND NOT COMPILE_AVX2)
      message(WARNING "AVX2 enabled due to -march=native and -DCOMPILE_AVX2=${COMPILE_AVX2} is ignored.")
    endif(AVX2_FOUND AND NOT COMPILE_AVX2)
    if(AVX512_FOUND AND NOT COMPILE_AVX512)
      message(WARNING "AVX512 enabled due to -march=native and -DCOMPILE_AVX512=${COMPILE_AVX512} is ignored.")
    endif(AVX512_FOUND AND NOT COMPILE_AVX512)
  else()
    # force to build with the requested intrisics, requires compiler support
    message(STATUS "Building with -march=${BUILD_ARCH} and forcing intrisics as requested")
    if(COMPILE_SSE2)
      message(STATUS "SSE2 support requested")
      set(INTRINSICS "${INTRINSICS} -msse2")
      list(APPEND INTRINSICS_NVCC -Xcompiler\ -msse2)
    endif(COMPILE_SSE2)
    if(COMPILE_SSE3)
      message(STATUS "SSE3 support requested")
      set(INTRINSICS "${INTRINSICS} -msse3")
      list(APPEND INTRINSICS_NVCC -Xcompiler\ -msse3)
    endif(COMPILE_SSE3)
    if(COMPILE_SSE4_1)
      message(STATUS "SSE4.1 support requested")
      set(INTRINSICS "${INTRINSICS} -msse4.1")
      list(APPEND INTRINSICS_NVCC -Xcompiler\ -msse4.1)
    endif(COMPILE_SSE4_1)
    if(COMPILE_SSE4_2)
      message(STATUS "SSE4.2 support requested")
      set(INTRINSICS "${INTRINSICS} -msse4.2")
      list(APPEND INTRINSICS_NVCC -Xcompiler\ -msse4.2)
    endif(COMPILE_SSE4_2)
    if(COMPILE_AVX)
      message(STATUS "AVX support requested")
      set(INTRINSICS "${INTRINSICS} -mavx")
      list(APPEND INTRINSICS_NVCC -Xcompiler\ -mavx)
    endif(COMPILE_AVX)
    if(COMPILE_AVX2)
      message(STATUS "AVX2 support requested")
      set(INTRINSICS "${INTRINSICS} -mavx2")
      list(APPEND INTRINSICS_NVCC -Xcompiler\ -mavx2)
    endif(COMPILE_AVX2)
    if(COMPILE_AVX512)
      message(STATUS "AVX512 support requested")
      set(INTRINSICS "${INTRINSICS} -mavx512f")
      list(APPEND INTRINSICS_NVCC -Xcompiler\ -mavx512f)
    endif(COMPILE_AVX512)
  endif()

  if(USE_FBGEMM)
    set(EXT_LIBS ${EXT_LIBS} fbgemm dl)
    add_definitions(-DUSE_FBGEMM=1)
  endif(USE_FBGEMM)

  if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 9.0)
    # Clang-10.0.0 complains when CUDA is newer than 10.1
    set(CLANG_IGNORE_UNKNOWN_CUDA "-Wno-unknown-warning-option -Wno-unknown-cuda-version")
  endif()
  set(DISABLE_GLOBALLY "-Wno-unused-result ${CLANG_IGNORE_UNKNOWN_CUDA}")

  # These are used in src/CMakeLists.txt on a per-target basis
  list(APPEND ALL_WARNINGS -Wall; -Werror; -Wextra; -Wno-unused-result; -Wno-deprecated;
    -Wno-pragmas; -Wno-unused-parameter; -Wno-unused-function;
    -Wno-unused-value; -Wno-unknown-pragmas; -Wno-sign-compare;
    -Wno-missing-field-initializers;)

  # This warning does not exist prior to gcc 5.0
  if(CMAKE_COMPILER_IS_GNUCC AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 5.0)
    list(APPEND ALL_WARNINGS -Wsuggest-override -Wno-int-in-bool-context)
  endif()

  if(CMAKE_COMPILER_IS_GNUCC)
    # these flags are not known to clang
    set(CMAKE_GCC_FLAGS "-Wl,--no-as-needed")
    set(CMAKE_RDYNAMIC_FLAG "-rdynamic")
  endif(CMAKE_COMPILER_IS_GNUCC)

  set(CMAKE_CXX_FLAGS                 "-std=c++11 -pthread ${CMAKE_GCC_FLAGS} -fPIC ${DISABLE_GLOBALLY} -march=${BUILD_ARCH} ${INTRINSICS}")
  set(CMAKE_CXX_FLAGS_RELEASE         "-O3 -m64 -funroll-loops -g ${CMAKE_RDYNAMIC_FLAG}")
  set(CMAKE_CXX_FLAGS_DEBUG           "-O0 -g ${CMAKE_RDYNAMIC_FLAG}")
  set(CMAKE_CXX_FLAGS_SLIM            "-O3 -m64 -funroll-loops -DNDEBUG")
  set(CMAKE_CXX_FLAGS_RELWITHDEBINFO  "${CMAKE_CXX_FLAGS_RELEASE}")
  set(CMAKE_CXX_FLAGS_PROFILE         "${CMAKE_CXX_FLAGS_RELEASE} -pg")
  set(CMAKE_CXX_FLAGS_PROFGEN         "${CMAKE_CXX_FLAGS_RELEASE} -fprofile-generate -fprofile-correction")
  set(CMAKE_CXX_FLAGS_PROFUSE         "${CMAKE_CXX_FLAGS_RELEASE} -fprofile-use -fprofile-correction")

  # these need to be set separately
  set(CMAKE_C_FLAGS                 "-pthread ${CMAKE_GCC_FLAGS} -fPIC ${DISABLE_GLOBALLY} -march=${BUILD_ARCH} ${INTRINSICS}")
  set(CMAKE_C_FLAGS_RELEASE         "-O3 -m64 -funroll-loops -g ${CMAKE_RDYNAMIC_FLAG}")
  set(CMAKE_C_FLAGS_DEBUG           "-O0 -g ${CMAKE_RDYNAMIC_FLAG}")
  set(CMAKE_C_FLAGS_SLIM            "-O3 -m64 -funroll-loops -DNDEBUG")
  set(CMAKE_C_FLAGS_RELWITHDEBINFO  "${CMAKE_C_FLAGS_RELEASE}")
  set(CMAKE_C_FLAGS_PROFILE         "${CMAKE_C_FLAGS_RELEASE} -pg")
  set(CMAKE_C_FLAGS_PROFGEN         "${CMAKE_C_FLAGS_RELEASE} -fprofile-generate -fprofile-correction")
  set(CMAKE_C_FLAGS_PROFUSE         "${CMAKE_C_FLAGS_RELEASE} -fprofile-use -fprofile-correction")
endif(MSVC)

# with gcc 7.0 and above we need to mark fallthrough in switch case statements
# that can be done in comments for backcompat, but CCACHE removes comments.
# -C makes gcc keep comments.
if(USE_CCACHE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -C")
endif()

###############################################################################
# Downloading SentencePiece if requested and set to compile with it.
# Requires all the dependencies imposed by SentencePiece
if(USE_SENTENCEPIECE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_SENTENCEPIECE")
  LIST(APPEND CUDA_NVCC_FLAGS -DUSE_SENTENCEPIECE; )
  set(EXT_LIBS ${EXT_LIBS} sentencepiece sentencepiece_train)
endif()

if(USE_ONNX)
  message(STATUS "Enabling experimental ONNX support")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_ONNX")
  # TODO: likely required to find protobuf by itself, we should check/fix this. Before it would take advantage of sentencepiece doing that.
  set(EXT_LIBS ${EXT_LIBS} protobuf)
  include_directories(${Protobuf_INCLUDE_DIRS})
endif()

# Find packages
set(EXT_LIBS ${EXT_LIBS} ${CMAKE_DL_LIBS})

###############################################################################
if(COMPILE_CUDA)

if(USE_STATIC_LIBS)
  # link statically to stdlib libraries
  if(NOT MSVC)
    set(CMAKE_EXE_LINKER_FLAGS "-static-libgcc -static-libstdc++")
  endif()

  # look for libraries that have .a suffix
  set(_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES})
  if(WIN32)
    list(INSERT CMAKE_FIND_LIBRARY_SUFFIXES 0 .lib .a)
  else()
    set(CMAKE_FIND_LIBRARY_SUFFIXES .a _static.a)
  endif()
endif()

find_package(CUDA "9.0") # TODO: only enable FP16-related options for compute_70 and higher.
if(CUDA_FOUND)
  # CUDA >= 10.0 requires CMake >= 3.12.2
  if((CUDA_VERSION VERSION_EQUAL "10.0" OR CUDA_VERSION VERSION_GREATER "10.0") AND (CMAKE_VERSION VERSION_LESS "3.12.2"))
      message(WARNING "On some Unix systems CUDA 10.0+ requires CMake 3.12.2+; you use CMake ${CMAKE_VERSION}")
  endif()

  # We want to compile as many targets as possible but different CUDA versions support different targets.
  # Let's instead enable options based on what cuda version we have.
  if((CUDA_VERSION VERSION_EQUAL "9.0" OR CUDA_VERSION VERSION_GREATER "9.0") AND CUDA_VERSION VERSION_LESS "11.0")
    option(COMPILE_KEPLER  "Compile GPU version with SM35 support" OFF)
    option(COMPILE_MAXWELL "Compile GPU version with SM50 support" OFF)
    option(COMPILE_PASCAL  "Compile GPU version with SM60 support" ON)
    option(COMPILE_VOLTA   "Compile GPU version with SM70 support" ON)
  endif()
  if((CUDA_VERSION VERSION_EQUAL "10.0" OR CUDA_VERSION VERSION_GREATER "10.0") AND CUDA_VERSION VERSION_LESS "11.0")
    option(COMPILE_KEPLER  "Compile GPU version with SM35 support" OFF)
    option(COMPILE_MAXWELL "Compile GPU version with SM50 support" OFF)
    option(COMPILE_PASCAL  "Compile GPU version with SM60 support" ON)
    option(COMPILE_VOLTA   "Compile GPU version with SM70 support" ON)
    option(COMPILE_TURING  "Compile GPU version with SM75 support" ON)
  endif()
  if(CUDA_VERSION VERSION_EQUAL "11.0" OR CUDA_VERSION VERSION_GREATER "11.0")
    option(COMPILE_KEPLER  "Compile GPU version with SM35 support" OFF) # deprecated for CUDA 11
    option(COMPILE_MAXWELL "Compile GPU version with SM50 support" OFF) # deprecated for CUDA 11
    option(COMPILE_PASCAL  "Compile GPU version with SM60 support" ON)
    option(COMPILE_VOLTA   "Compile GPU version with SM70 support" ON)
    option(COMPILE_TURING  "Compile GPU version with SM75 support" ON)
    option(COMPILE_AMPERE  "Compile GPU version with SM80 support" ON)
    LIST(APPEND COMPUTE -Wno-deprecated-gpu-targets)
  endif()
  if(CUDA_VERSION VERSION_EQUAL "11.1" OR CUDA_VERSION VERSION_GREATER "11.1")
    option(COMPILE_KEPLER      "Compile GPU version with SM35 support" OFF) # deprecated for CUDA 11
    option(COMPILE_MAXWELL     "Compile GPU version with SM50 support" OFF) # deprecated for CUDA 11
    option(COMPILE_PASCAL      "Compile GPU version with SM60 support" ON)
    option(COMPILE_VOLTA       "Compile GPU version with SM70 support" ON)
    option(COMPILE_TURING      "Compile GPU version with SM75 support" ON)
    option(COMPILE_AMPERE      "Compile GPU version with SM80 support" ON)
    option(COMPILE_AMPERE_RTX  "Compile GPU version with SM86 support" ON)
    LIST(APPEND COMPUTE -Wno-deprecated-gpu-targets)
  endif()

  if(COMPILE_KEPLER)
    message(STATUS "Compiling code for Kepler GPUs")
    LIST(APPEND COMPUTE -gencode=arch=compute_35,code=sm_35;) # Tesla K40 and above
  endif(COMPILE_KEPLER)
  if(COMPILE_MAXWELL)
    message(STATUS "Compiling code for Maxwell GPUs")
    LIST(APPEND COMPUTE -gencode=arch=compute_50,code=sm_50; -gencode=arch=compute_52,code=sm_52;)     # Maxwell GPUs
  endif(COMPILE_MAXWELL)
  if(COMPILE_PASCAL)
    message(STATUS "Compiling code for Pascal GPUs")
    LIST(APPEND COMPUTE -gencode=arch=compute_60,code=sm_60; -gencode=arch=compute_61,code=sm_61;)     # Pascal GPUs
  endif(COMPILE_PASCAL)
  if(COMPILE_VOLTA)
    message(STATUS "Compiling code for Volta GPUs")
    LIST(APPEND COMPUTE -arch=sm_70; -gencode=arch=compute_70,code=sm_70; -gencode=arch=compute_70,code=compute_70) # Volta GPUs
  endif(COMPILE_VOLTA)
  if(CUDA_VERSION VERSION_EQUAL "10.0" OR CUDA_VERSION VERSION_GREATER "10.0")
    if(COMPILE_TURING)
        message(STATUS "Compiling code for Turing GPUs")
        LIST(APPEND COMPUTE -gencode=arch=compute_75,code=sm_75; -gencode=arch=compute_75,code=compute_75) # Turing GPUs
    endif(COMPILE_TURING)
  endif()
  if(CUDA_VERSION VERSION_EQUAL "11.0" OR CUDA_VERSION VERSION_GREATER "11.0")
    if(COMPILE_AMPERE)
        message(STATUS "Compiling code for Ampere GPUs")
        LIST(APPEND COMPUTE -gencode=arch=compute_80,code=sm_80; -gencode=arch=compute_80,code=compute_80) # Ampere GPUs
    endif(COMPILE_AMPERE)
  endif()
  if(CUDA_VERSION VERSION_EQUAL "11.1" OR CUDA_VERSION VERSION_GREATER "11.1")
    if(COMPILE_AMPERE_RTX)
        message(STATUS "Compiling code for Ampere RTX GPUs")
        LIST(APPEND COMPUTE -gencode=arch=compute_86,code=sm_86; -gencode=arch=compute_86,code=compute_86) # Ampere RTX GPUs
    endif(COMPILE_AMPERE_RTX)
  endif()

  if(USE_STATIC_LIBS)
    set(EXT_LIBS ${EXT_LIBS} ${CUDA_curand_LIBRARY} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_cusparse_LIBRARY})
    set(CUDA_LIBS ${CUDA_curand_LIBRARY} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_cusparse_LIBRARY})

    find_library(CUDA_culibos_LIBRARY NAMES culibos PATHS ${CUDA_TOOLKIT_ROOT_DIR}/lib64 ${CUDA_TOOLKIT_ROOT_DIR}/lib/x64)
    # The cuLIBOS library does not seem to exist in Windows CUDA toolkit installs
    if(CUDA_culibos_LIBRARY)
      set(EXT_LIBS ${EXT_LIBS} ${CUDA_culibos_LIBRARY})
      set(CUDA_LIBS ${CUDA_LIBS} ${CUDA_culibos_LIBRARY})
    elseif(NOT WIN32)
      message(FATAL_ERROR "cuLIBOS library not found")
    endif()
    # CUDA 10.1 introduces cublasLt library that is required on static build
    if ((CUDA_VERSION VERSION_EQUAL "10.1" OR CUDA_VERSION VERSION_GREATER "10.1"))
      find_library(CUDA_cublasLt_LIBRARY NAMES cublasLt PATHS ${CUDA_TOOLKIT_ROOT_DIR}/lib64 ${CUDA_TOOLKIT_ROOT_DIR}/lib/x64)
      if(NOT CUDA_cublasLt_LIBRARY)
        message(FATAL_ERROR "cuBLASLt library not found")
      endif()
      set(EXT_LIBS ${EXT_LIBS} ${CUDA_cublasLt_LIBRARY})
      set(CUDA_LIBS ${CUDA_LIBS} ${CUDA_cublasLt_LIBRARY})
    endif()
    message(STATUS "Found CUDA libraries: ${CUDA_LIBS}")
  else(USE_STATIC_LIBS)
  set(CUDA_LIBS ${CUDA_curand_LIBRARY} ${CUDA_cusparse_LIBRARY} ${CUDA_CUBLAS_LIBRARIES})
    # We actually only need cublasLt here after cuda 11. Marian will work fine without it pre cuda 11. We want to force CMake to use the cublas
    # version that ships with CUDA 11 so we force the search to occur inside of the cuda toolkit directory.
    set(CUDA_LIBS ${CUDA_curand_LIBRARY} ${CUDA_cusparse_LIBRARY} ${CUDA_CUBLAS_LIBRARIES})
    if ((CUDA_VERSION VERSION_EQUAL "11.0" OR CUDA_VERSION VERSION_GREATER "11.0"))
      find_library(CUDA_cublasLt_LIBRARY NAMES cublasLt PATHS ${CUDA_TOOLKIT_ROOT_DIR}/lib64 ${CUDA_TOOLKIT_ROOT_DIR}/lib/x64 NO_DEFAULT_PATH)
      if(NOT CUDA_cublasLt_LIBRARY)
        message(FATAL_ERROR "cuBLASLt library not found")
      endif()
      set(EXT_LIBS ${EXT_LIBS} ${CUDA_cublasLt_LIBRARY})
      set(CUDA_LIBS ${CUDA_LIBS} ${CUDA_cublasLt_LIBRARY})
    endif()
    set(EXT_LIBS ${EXT_LIBS} ${CUDA_curand_LIBRARY} ${CUDA_cusparse_LIBRARY} ${CUDA_CUBLAS_LIBRARIES})
    message(STATUS "Found CUDA libraries: ${CUDA_LIBS}")
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
  message("
Cannot find suitable CUDA libraries. Specify the path explicitly with
  -DCUDA_TOOLKIT_ROOT_DIR=/path/to/appropriate/cuda/installation
   (hint: try /usr/local/$(readlink /usr/local/cuda))
OR compile the CPU-only version of Marian with
  -DCOMPILE_CUDA=off
")
  message(FATAL_ERROR "FATAL ERROR: No suitable CUDA library found.")
endif(CUDA_FOUND)

else(COMPILE_CUDA)
  message(WARNING "COMPILE_CUDA=off : Building only CPU version")
endif(COMPILE_CUDA)

# TODO: make compatible with older CUDA versions
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  list(APPEND CUDA_NVCC_FLAGS --default-stream per-thread; -O0; -g; --use_fast_math; ${COMPUTE})
else(CMAKE_BUILD_TYPE STREQUAL "Debug")
  list(APPEND CUDA_NVCC_FLAGS --default-stream per-thread; -O3; -g; --use_fast_math; ${COMPUTE})
endif(CMAKE_BUILD_TYPE STREQUAL "Debug")
if(NOT MSVC)
  # @TODO: add warnings here too
  list(APPEND CUDA_NVCC_FLAGS -ccbin ${CMAKE_C_COMPILER}; -std=c++11; -Xcompiler\ -fPIC; -Xcompiler\ -Wno-unused-result; -Xcompiler\ -Wno-deprecated; -Xcompiler\ -Wno-pragmas; -Xcompiler\ -Wno-unused-value; -Xcompiler\ -Werror;)
  list(APPEND CUDA_NVCC_FLAGS ${INTRINSICS_NVCC})
else()
  list(APPEND CUDA_NVCC_FLAGS -Xcompiler\ /FS; -Xcompiler\ /MT$<$<CONFIG:Debug>:d>; )
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

###############################################################################
# Find Tcmalloc
if(NOT WIN32)
  find_package(Tcmalloc)
  if(Tcmalloc_FOUND)
    include_directories(${Tcmalloc_INCLUDE_DIR})
    set(EXT_LIBS ${EXT_LIBS} ${Tcmalloc_LIBRARIES})
  else(Tcmalloc_FOUND)
    message(WARNING "Cannot find TCMalloc library. Continuing.")
  endif(Tcmalloc_FOUND)
endif()

###############################################################################
# Find BLAS library
if(COMPILE_CPU)
  if(NOT GENERATE_MARIAN_INSTALL_TARGETS)
    set(EXT_LIBS ${EXT_LIBS} intgemm) # Enable intgemm when compiling CPU
    add_definitions(-DCOMPILE_CPU=1)
  endif()
  if(USE_APPLE_ACCELERATE)
    if(NOT APPLE)
      message(FATAL_ERROR "FATAL ERROR: Apple Accelerate only works on macOS.")
    endif()
    set(BLAS_VENDOR "Accelerate")
    # see https://developer.apple.com/documentation/accelerate for more info
    # you may need to install Xcode command line tools if you don't have them already (https://developer.apple.com/xcode/features/)
    include_directories("/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/System/Library/Frameworks/Accelerate.framework/Frameworks/vecLib.framework/Headers")
    set(EXT_LIBS ${EXT_LIBS} "-framework Accelerate")
    add_definitions(-DBLAS_FOUND=1)
  else(USE_APPLE_ACCELERATE)
    if(USE_MKL)
      find_package(MKL)
    endif(USE_MKL)
    if(MKL_FOUND)
      include_directories(${MKL_INCLUDE_DIR})
      set(EXT_LIBS ${EXT_LIBS} ${MKL_LIBRARIES})
      set(BLAS_FOUND TRUE)
      add_definitions(-DBLAS_FOUND=1 -DMKL_FOUND=1)
    else(MKL_FOUND)
      set(BLAS_VENDOR "OpenBLAS")
      find_package(BLAS)
      if(BLAS_FOUND)
        include(FindCBLAS)
        if(CBLAS_FOUND)
          include_directories(${BLAS_INCLUDE_DIR} ${CBLAS_INCLUDE_DIR})
          set(EXT_LIBS ${EXT_LIBS} ${BLAS_LIBRARIES} ${CBLAS_LIBRARIES})
          add_definitions(-DBLAS_FOUND=1)
        endif(CBLAS_FOUND)
      endif(BLAS_FOUND)
    endif(MKL_FOUND)
  endif(USE_APPLE_ACCELERATE)
endif(COMPILE_CPU)

###############################################################################
# Find OpenSSL
set(BOOST_COMPONENTS "")
if(COMPILE_SERVER)
  find_package(OpenSSL)
  if(OpenSSL_FOUND)
    message(STATUS "Found OpenSSL")
    include_directories(${OPENSSL_INCLUDE_DIR})
    set(EXT_LIBS ${EXT_LIBS} ${OPENSSL_CRYPTO_LIBRARY})
    if(MSVC AND USE_STATIC_LIBS)
      # "If you link with static OpenSSL libraries then you're expected to additionally link your
      # application with WS2_32.LIB, GDI32.LIB, ADVAPI32.LIB, CRYPT32.LIB and USER32.LIB"
      # See https://github.com/openssl/openssl/blob/OpenSSL_1_1_1d/NOTES.WIN#L127
      # Linking with crypt32.lib seem to be enough.
      set(EXT_LIBS ${EXT_LIBS} crypt32.lib)
    endif()
    set(BOOST_COMPONENTS ${BOOST_COMPONENTS} system)
  else(OpenSSL_FOUND)
    message(WARNING "Cannot find OpenSSL library. Not compiling server.")
    set(COMPILE_SERVER "off")
  endif(OpenSSL_FOUND)
endif(COMPILE_SERVER)

###############################################################################
# Undo static lib search and put non-static searches here:

if(USE_STATIC_LIBS)
  set(CMAKE_FIND_LIBRARY_SUFFIXES ${_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES})
endif()

if(DETERMINISTIC)
  message(WARNING "Option DETERMINISTIC=ON: Trying to make training as deterministic as possible, may result in slow-down")
  add_definitions(-DDETERMINISTIC=1)
  list(APPEND CUDA_NVCC_FLAGS -DDETERMINISTIC=1; )
else()
  add_definitions(-DDETERMINISTIC=0)
  list(APPEND CUDA_NVCC_FLAGS -DDETERMINISTIC=0; )
endif()

# Find MPI
if(USE_MPI)
  # 2.0 refers to MPI2 standard. OpenMPI is an implementation of that standard regardless of the specific OpenMPI version
  # e.g. OpenMPI 1.10 implements MPI2 and will be found correctly.
  find_package(MPI 2.0 REQUIRED)
  if(MPI_FOUND)
    include_directories(${MPI_INCLUDE_PATH})
    set(EXT_LIBS ${EXT_LIBS} ${MPI_LIBRARIES})
    if(USE_STATIC_LIBS) # alternatively this could install OpenMPI like NCCL and link against that statically with greater control
    message(WARNING "MPI implementations are notoriously difficult to link statically, linking ${MPI_LIBRARIES} dynamically despite -DUSE_STATIC_LIBS=on")
    endif(USE_STATIC_LIBS)
    add_definitions(-DMPI_FOUND=1)
  endif(MPI_FOUND)
endif(USE_MPI)


###############################################################################
# Find Boost if required
if(BOOST_COMPONENTS)
  if(USE_STATIC_LIBS)
    set(Boost_USE_STATIC_LIBS ON)
  endif()

  find_package(Boost COMPONENTS ${BOOST_COMPONENTS})
  if(Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS})
    set(EXT_LIBS ${EXT_LIBS} ${Boost_LIBRARIES})
    set(EXT_LIBS ${EXT_LIBS} ${ZLIB_LIBRARIES}) # hack for static compilation
    if(MSVC)
      add_definitions(-DBOOST_ALL_NO_LIB=1) # hack for missing date-time stub
    endif()
  else(Boost_FOUND)
    message(SEND_ERROR "Cannot find Boost libraries. Terminating.")
  endif(Boost_FOUND)
endif(BOOST_COMPONENTS)

###############################################################################
if(COMPILE_TESTS)
  enable_testing()
endif(COMPILE_TESTS)

if(COMPILE_EXAMPLES)
  add_definitions(-DCOMPILE_EXAMPLES=1)
endif(COMPILE_EXAMPLES)

# Generate project_version.h to reflect our version number
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/src/common/project_version.h.in
               ${CMAKE_CURRENT_SOURCE_DIR}/src/common/project_version.h @ONLY)

# Generate build_info.cpp with CMake cache variables
include(GetCacheVariables)

# make sure src/common/build_info.cpp has been removed
execute_process(COMMAND rm ${CMAKE_CURRENT_SOURCE_DIR}/src/common/build_info.cpp
                OUTPUT_QUIET ERROR_QUIET)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/src/common/build_info.cpp.in
               ${CMAKE_CURRENT_BINARY_DIR}/src/common/build_info.cpp @ONLY)
# to be able to check if this is a CMake-based compilation, which always adds
# build-info option, even on Windows.
add_definitions(-DBUILD_INFO_AVAILABLE=1)

# Compile source files
include_directories(${marian_SOURCE_DIR}/src)
add_subdirectory(src)

###############################################################################
if(USE_DOXYGEN)
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
endif(USE_DOXYGEN)
