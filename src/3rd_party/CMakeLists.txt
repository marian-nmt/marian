
include_directories(.)

add_subdirectory(./yaml-cpp)
add_subdirectory(./SQLiteCpp)
add_subdirectory(./pathie-cpp)
add_subdirectory(./zlib)
add_subdirectory(./faiss)
include_directories(./faiss)

if(COMPILE_CPU)
  if(NOT GENERATE_MARIAN_INSTALL_TARGETS)
    set(INTGEMM_DONT_BUILD_TESTS ON CACHE BOOL "Disable intgemm tests")
    add_subdirectory(./intgemm)
  endif()
endif(COMPILE_CPU)

if(USE_FBGEMM)
  # @TODO: find out if this is somehow harmful. This is supppressing CMake warnings for CMAKE_SUPPRESS_DEVELOPER_WARNINGS
  # meant to silence CMakeFiles of 3rd_party tools.
  if(NOT DEFINED CMAKE_SUPPRESS_DEVELOPER_WARNINGS)
    set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS 1 CACHE INTERNAL "No dev warnings")
  endif()

  if(NOT MSVC)
    # only locally disabled for the 3rd_party folder
    # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-value -Wno-unused-parameter -Wno-unused-variable -Wno-unused-function")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused")
  endif()

  # Do not compile cpuinfo executables due to a linker error, and they are not needed
  set(CPUINFO_BUILD_TOOLS OFF CACHE BOOL "Build command-line tools")
  # Do not build cpuinfo tests and benchmarks
  set(CPUINFO_BUILD_UNIT_TESTS OFF CACHE BOOL "Do not build cpuinfo unit tests")
  set(CPUINFO_BUILD_MOCK_TESTS OFF CACHE BOOL "Do not build cpuinfo mock tests")
  set(CPUINFO_BUILD_BENCHMARKS OFF CACHE BOOL "Do not build cpuinfo benchmarks")
  if(MSVC)
    # Force static compilation of cpuinfo on Windows
    SET(CPUINFO_LIBRARY_TYPE "static" CACHE STRING "Type of cpuinfo library (shared, static, or default) to build")
    SET(CPUINFO_RUNTIME_TYPE "static" CACHE STRING "Type of runtime library (shared, static, or default) to use")
  endif(MSVC)

  set(FBGEMM_BUILD_TESTS OFF CACHE BOOL "Disable fbgemm tests")
  set(FBGEMM_BUILD_BENCHMARKS OFF CACHE BOOL "Disable fbgemm benchmark")
  add_subdirectory(./fbgemm)

  # asmjit (3rd-party submodule of fbgemm) sets -Wall -Wextra near the end of
  # the compile options, invalidating any -Wno-... flags that we may have set
  # earlier. Let's remove them.
  get_property(ASMJIT_COMPILE_OPTIONS TARGET asmjit PROPERTY COMPILE_OPTIONS)
  list(REMOVE_ITEM ASMJIT_COMPILE_OPTIONS -Wall -Wextra)
  set_property(TARGET asmjit PROPERTY COMPILE_OPTIONS ${ASMJIT_COMPILE_OPTIONS})
  message("   ASMJIT COMPILE FLAGS: ${ASMJIT_COMPILE_OPTIONS}")

  if(GENERATE_MARIAN_INSTALL_TARGETS)
    install(TARGETS fbgemm asmjit cpuinfo clog
      EXPORT marian-targets
      DESTINATION fbgemm)
  endif(GENERATE_MARIAN_INSTALL_TARGETS)

endif(USE_FBGEMM)

if(USE_SENTENCEPIECE)
  if(USE_STATIC_LIBS)
    set(_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES})
    if(WIN32)
      list(INSERT CMAKE_FIND_LIBRARY_SUFFIXES 0 .lib .a)
    else()
      set(CMAKE_FIND_LIBRARY_SUFFIXES .a)
    endif()
  endif()

  # regardless of -DUSE_STATIC_LIBS setting always build sentencepiece statically
  set(SPM_ENABLE_SHARED OFF CACHE BOOL  "Builds shared libaries in addition to static libraries." FORCE)
  set(SPM_ENABLE_TCMALLOC ON CACHE BOOL "Enable TCMalloc if available.")

  if(USE_STATIC_LIBS)
    set(SPM_TCMALLOC_STATIC ON CACHE BOOL "Link static library of TCMALLOC." FORCE)
  else(USE_STATIC_LIBS)
    set(SPM_TCMALLOC_STATIC OFF CACHE BOOL "Link static library of TCMALLOC.")
  endif(USE_STATIC_LIBS)

  add_subdirectory(./sentencepiece)
  include_directories(./sentencepiece)

  set_target_properties(spm_encode spm_decode spm_train spm_normalize spm_export_vocab
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

  if (CMAKE_CXX_COMPILER_ID MATCHES "GNU" AND (CMAKE_CXX_COMPILER_VERSION VERSION_EQUAL 8.0 OR CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 8.0))
    foreach(t sentencepiece-static sentencepiece_train-static
        spm_decode spm_encode spm_export_vocab spm_normalize spm_train)
        set_property(TARGET ${t} APPEND_STRING PROPERTY COMPILE_FLAGS " -Wno-class-memaccess")
    endforeach(t)
  endif()

  if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    foreach(t sentencepiece-static sentencepiece_train-static
        spm_decode spm_encode spm_export_vocab spm_normalize spm_train)
      set_property(TARGET ${t} APPEND_STRING PROPERTY COMPILE_FLAGS " -Wno-tautological-compare -Wno-unused")
      if (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 9.0)
        set_property(TARGET ${t} APPEND_STRING PROPERTY COMPILE_FLAGS " -Wno-range-loop-construct")
      endif()
      # get_property(SENTENCEPIECE_COMPILE_FLAGS TARGET ${t} PROPERTY COMPILE_FLAGS)
      # message("-- SENTENCPIECE: compile flags for target ${t}: ${SENTENCEPIECE_COMPILE_FLAGS}")
    endforeach(t)
  endif()

  if(USE_STATIC_LIBS)
    set(CMAKE_FIND_LIBRARY_SUFFIXES ${_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES})
  endif()

  # regardless of -DUSE_STATIC_LIBS setting always build sentencepiece statically
  if(GENERATE_MARIAN_INSTALL_TARGETS)
    install(TARGETS sentencepiece-static sentencepiece_train-static
      EXPORT marian-targets
      DESTINATION sentencepiece)
  endif(GENERATE_MARIAN_INSTALL_TARGETS)
endif(USE_SENTENCEPIECE)

include_directories(./SQLiteCpp/include)
include_directories(./CLI)
include_directories(./pathie-cpp/include)

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  #set_target_properties(SQLiteCpp PROPERTIES COMPILE_FLAGS
  set_property(TARGET SQLiteCpp APPEND_STRING PROPERTY COMPILE_FLAGS
    " -Wno-parentheses-equality -Wno-unused-value")
  if (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 9.0)
    set_property(TARGET SQLiteCpp APPEND_STRING PROPERTY COMPILE_FLAGS
      " -Wno-implicit-int-float-conversion")
  endif()
  set_property(TARGET libyaml-cpp APPEND_STRING PROPERTY COMPILE_FLAGS
    " -fPIC -Wno-unused-value")
  set_property(TARGET pathie-cpp APPEND_STRING PROPERTY COMPILE_FLAGS
    " -fPIC -Wno-unused-value")
endif()

include_directories(./zlib)

include(ExternalProject)

set(INSTALLS "") # this will contain a list of 3rd part dependencies that we install locally
if(CUDA_FOUND)
  if(USE_NCCL)

    # disables compilation for sm_30 to avoid ptxas warning... that is general Kepler support. But K80s are supported for instance by sm_35

    set(GENCODE "")
    if(CUDA_VERSION VERSION_EQUAL "11.0" OR CUDA_VERSION VERSION_GREATER "11.0")
      set(GENCODE "${GENCODE} -Wno-deprecated-gpu-targets")
    endif()
    if(COMPILE_KEPLER)
      set(GENCODE "${GENCODE} -gencode=arch=compute_35,code=sm_35")
    endif(COMPILE_KEPLER)
    if(COMPILE_MAXWELL)
      set(GENCODE "${GENCODE} -gencode=arch=compute_50,code=sm_50")
    endif(COMPILE_MAXWELL)
    if(COMPILE_PASCAL)
      set(GENCODE "${GENCODE} -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_61,code=sm_61")
    endif(COMPILE_PASCAL)
    if(COMPILE_VOLTA)
      set(GENCODE "${GENCODE} -arch=sm_70 -gencode=arch=compute_70,code=sm_70")
    endif(COMPILE_VOLTA)
    if(COMPILE_TURING)
      set(GENCODE "${GENCODE} -gencode=arch=compute_75,code=sm_75; -gencode=arch=compute_75,code=compute_75")
    endif(COMPILE_TURING)
    if(COMPILE_AMPERE)
      set(GENCODE "${GENCODE} -gencode=arch=compute_80,code=sm_80; -gencode=arch=compute_80,code=compute_80")
    endif(COMPILE_AMPERE)

    # install nccl in ${CMAKE_BINARY_DIR}/local similar to /usr/local linux installation
    # Using $(MAKE) instead of $CMAKE_MAKE_PROGRAM in order to make parallelization in NCCL compilation work with make -j16. 
    # Apparently this does not get properly propagated otherwise and builts with only a single thread/process.
    ExternalProject_Add(nccl_install
      SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/nccl
      BINARY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/nccl
      CONFIGURE_COMMAND ""
      BUILD_COMMAND
        # Note: $(MAKE) here causes CMake cache generation via Ninja failing (Windows build with
        # CMake only) because Ninja complains about unescaped $ signs. This happens only if
        # COMPILE_CUDA=on and because of that, USE_NCCL=off is used in CMakeSettings.json for now.
        # @TODO: find proper escaping that works for MSVC builds.
        "\$(MAKE)" -f ${CMAKE_CURRENT_SOURCE_DIR}/nccl/Makefile src.build
        BUILDDIR=${CMAKE_BINARY_DIR}/local CUDA_HOME=${CUDA_TOOLKIT_ROOT_DIR}
        CUDA8_GENCODE=${GENCODE} CXX=${CMAKE_CXX_COMPILER} CXX_FLAGS=${NCCL_FLAGS}
      INSTALL_COMMAND "")

    set_target_properties(nccl PROPERTIES IMPORTED_LOCATION ${CMAKE_BINARY_DIR}/local/lib/libnccl_static.a)
    add_dependencies(nccl nccl_install)
    set(INSTALLS ${INSTALLS} nccl_install)

  endif(USE_NCCL)
endif(CUDA_FOUND)

# @TODO: do the same for SentencePiece, Protobuf etc.
# make clean will clean "${CMAKE_BINARY_DIR}/local"
set_directory_properties(PROPERTY ADDITIONAL_MAKE_CLEAN_FILES ${CMAKE_BINARY_DIR}/local)

add_custom_target(3rd_party_installs DEPENDS ${INSTALLS})
