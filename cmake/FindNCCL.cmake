set(NCCL_INC_PATHS
    /usr/include
    /usr/local/include
    /usr/local/cuda/include
    $ENV{NCCL_DIR}/include
    $ENV{CUDA_TOOLKIT_ROOT_DIRCUDA_ROOT}/include
)

set(NCCL_LIB_PATHS
    /lib
    /lib64
    /usr/lib
    /usr/lib64
    /usr/local/lib
    /usr/local/lib64
    /usr/local/cuda/lib64
    $ENV{NCCL_DIR}/lib64
    $ENV{CUDA_TOOLKIT_ROOT_DIR}/lib64
    /usr/local/cuda/lib
    $ENV{NCCL_DIR}/lib
    $ENV{CUDA_TOOLKIT_ROOT_DIR}/lib
)

find_path(NCCL_INCLUDE_DIR NAMES nccl.h PATHS ${NCCL_INC_PATHS})

if (USE_STATIC_LIBS)
  message(STATUS "Trying to find static NCCL library")
  find_library(NCCL_LIBRARIES NAMES libnccl_static.a PATHS ${NCCL_LIB_PATHS})
else (USE_STATIC_LIBS)
  find_library(NCCL_LIBRARIES NAMES nccl PATHS ${NCCL_LIB_PATHS})
endif (USE_STATIC_LIBS)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NCCL DEFAULT_MSG NCCL_INCLUDE_DIR NCCL_LIBRARIES)

if (NCCL_FOUND)
  message(STATUS "Found NCCL (include: ${NCCL_INCLUDE_DIR}, library: ${NCCL_LIBRARIES})")
  mark_as_advanced(NCCL_INCLUDE_DIR NCCL_LIBRARIES)
endif ()
