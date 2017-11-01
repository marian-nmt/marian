#pragma once

#include <cuda.h>

namespace marian {

namespace gpu {

#define __H__ __host__
#define __D__ __device__
#define __HI__ __host__ inline
#define __DI__ __device__ inline
#define __HD__ __host__ __device__
#define __HDI__ __host__ __device__ inline

template <typename T, size_t N>
struct Array {
  typedef T value_type;

  T data_[N];

  __HDI__ const T* data() const { return data_; }

  __HDI__ T* data() { return data_; }

  __HDI__ constexpr static size_t size() { return N; }

  __HDI__ T& operator[](size_t i) { return data_[i]; }
  __HDI__ const T& operator[](size_t i) const { return data_[i]; }

  __HDI__ T* begin() { return data_; }
  __HDI__ const T* begin() const { return data_; }

  __HDI__ T* end() { return data_ + N; }
  __HDI__ const T* end() const { return data_ + N; }

  __HDI__ void fill(T val) {
    for(int i = 0; i < N; ++i)
      data_[i] = val;
  }
};

}

}
