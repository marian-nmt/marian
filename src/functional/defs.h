#pragma once

#ifdef __CUDACC__  // Compiling with NVCC, host or device code

#include <cuda.h>
#define HOST __host__
#define DEVICE __device__
#define DEVICE_INLINE __device__ inline
#define HOST_INLINE __host__ inline
#define HOST_DEVICE __host__ __device__
#define HOST_DEVICE_INLINE __host__ __device__ inline

#else // Compiling with GCC or other host compiler

#define HOST
#define DEVICE
#define DEVICE_INLINE inline
#define HOST_INLINE inline
#define HOST_DEVICE
#define HOST_DEVICE_INLINE inline

#endif
