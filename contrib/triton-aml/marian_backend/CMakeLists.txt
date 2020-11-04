cmake_minimum_required(VERSION 3.17)

project(tritonmarianbackend LANGUAGES C CXX)

#
# Options
#
# Must include options required for this project as well as any
# projects included in this one by FetchContent.
#
# GPU support is disabled by default because marian backend doesn't
# support GPUs.
#
option(TRITON_ENABLE_GPU "Enable GPU support in backend" OFF)
option(TRITON_ENABLE_STATS "Include statistics collections in backend" ON)

set(TRITON_COMMON_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/common repo")
set(TRITON_CORE_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/core repo")
set(TRITON_BACKEND_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/backend repo")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

#
# Dependencies
#
# FetchContent's composibility isn't very good. We must include the
# transitive closure of all repos so that we can override the tag.
#
include(FetchContent)

FetchContent_Declare(
  repo-common
  GIT_REPOSITORY https://github.com/triton-inference-server/common.git
  GIT_TAG ${TRITON_COMMON_REPO_TAG}
  GIT_SHALLOW ON
)
FetchContent_Declare(
  repo-core
  GIT_REPOSITORY https://github.com/triton-inference-server/core.git
  GIT_TAG ${TRITON_CORE_REPO_TAG}
  GIT_SHALLOW ON
)
FetchContent_Declare(
  repo-backend
  GIT_REPOSITORY https://github.com/triton-inference-server/backend.git
  GIT_TAG ${TRITON_BACKEND_REPO_TAG}
  GIT_SHALLOW ON
)
FetchContent_MakeAvailable(repo-common repo-core repo-backend)

#
# Shared library implementing the Triton Backend API
#
configure_file(src/libtriton_marian.ldscript libtriton_marian.ldscript COPYONLY)

add_library(
  triton-marian-backend SHARED
  src/marian.cc
)

add_library(
  TritonMarianBackend::triton-marian-backend ALIAS triton-marian-backend
)

target_include_directories(
  triton-marian-backend
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_compile_features(triton-marian-backend PRIVATE cxx_std_11)
target_compile_options(
  triton-marian-backend PRIVATE
  $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
    -Wall -Wextra -Wno-unused-parameter -Wno-type-limits -Werror>
)

target_link_libraries(
  triton-marian-backend
  PRIVATE
    triton-backend-utils    # from repo-backend
    triton-core-serverstub  # from repo-core
)

target_link_libraries(
  triton-marian-backend
  PRIVATE   # from marian environment
    cmarian
    cmarian_cuda
    sentencepiece
    sentencepiece_train
    fbgemm
    asmjit
    protobuf
)


set_target_properties(
  triton-marian-backend PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  OUTPUT_NAME triton_marian
  LINK_DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libtriton_marian.ldscript
  LINK_FLAGS "-Wl,--version-script libtriton_marian.ldscript"
)

#
# Install
#
include(GNUInstallDirs)
set(INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/cmake/TritonMarianBackend)

install(
  TARGETS
    triton-marian-backend
  EXPORT
    triton-marian-backend-targets
  LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/backends/marian
  ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/backends/marian
)

install(
  EXPORT
    triton-marian-backend-targets
  FILE
    TritonMarianBackendTargets.cmake
  NAMESPACE
    TritonMarianBackend::
  DESTINATION
    ${INSTALL_CONFIGDIR}
)

include(CMakePackageConfigHelpers)
configure_package_config_file(
  ${CMAKE_CURRENT_LIST_DIR}/cmake/TritonMarianBackendConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/TritonMarianBackendConfig.cmake
  INSTALL_DESTINATION ${INSTALL_CONFIGDIR}
)

install(
  FILES
  ${CMAKE_CURRENT_BINARY_DIR}/TritonMarianBackendConfig.cmake
  DESTINATION ${INSTALL_CONFIGDIR}
)

#
# Export from build tree
#
export(
  EXPORT triton-marian-backend-targets
  FILE ${CMAKE_CURRENT_BINARY_DIR}/TritonMarianBackendTargets.cmake
  NAMESPACE TritonMarianBackend::
)

export(PACKAGE TritonMarianBackend)
