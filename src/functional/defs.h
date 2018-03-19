#pragma once

#include <type_traits>

#ifdef __CUDA_ARCH__

#include <cuda.h>
#define __H__ __host__
#define __D__ __device__
#define __HI__ __host__ inline
#define __DI__ __device__ inline
#define __HD__ __host__ __device__
#define __HDI__ __host__ __device__ inline

#else

#define __H__
#define __D__
#define __HI__ inline
#define __DI__ inline
#define __HD__
#define __HDI__ inline

#endif
