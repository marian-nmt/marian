#pragma once

#include <cstdint>
#include <string>

#include <cuda.h>
#include "common/shape.h"

namespace marian {

/**
 * @brief Represents the size of each dimension in a tensor.
 *
 * Note: this class currently is hard-coded to four dimensions.
 */

template <const int N>
struct ShapeN {
  int shape_[N];
  int stride_[N];
  int bstride_[N];
  const size_t filled_;

  ShapeN(const ShapeN& shape) : filled_{shape.filled_} {
    std::copy(shape.shape_, shape.shape_ + N, shape_);
    std::copy(shape.stride_, shape.stride_ + N, stride_);
    std::copy(shape.bstride_, shape.bstride_ + N, bstride_);
  }

  ShapeN(const Shape& shape) : filled_{shape.size()} {

    std::copy(shape.shape_.begin(), shape.shape_.end(), shape_ + N - filled_);
    std::copy(shape.stride_.begin(), shape.stride_.end(), stride_ + N - filled_);
    std::copy(shape.bstride_.begin(), shape.bstride_.end(), bstride_ + N - filled_);

    if(N - filled_) {
      std::fill_n(shape_, N - filled_, 1);
      std::fill_n(stride_, N - filled_, stride_[N - filled_]);
      std::fill_n(bstride_, N - filled_, 0);
    }
  }

  __device__ inline void updateStrides() {

    stride_[N - 1] = 1;
    bstride_[N - 1] = shape_[N - 1] == 1 ? 0 : stride_[N - 1];

    for(int i = N - 2; i >= 0; --i) {
      stride_[i] = stride_[i + 1] * shape_[i + 1];
      bstride_[i] = shape_[i] == 1 ? 0 : stride_[i];
    }
  }

  __device__ inline void set(int i, int dim) {
    shape_[i] = dim;
    updateStrides();
  }

  __device__ inline int dim(int i) { return shape_[i]; }

  __device__ inline int dim(int i) const {
    return const_cast<ShapeN&>(*this).dim(i);
  }

  __device__ inline int back() const { return dim(N - 1); }

  __device__ inline int operator[](int i) { return dim(i); }

  __device__ inline int operator[](int i) const { return dim(i); }

  __device__ inline int stride(int i) const { return stride_[i]; }

  __device__ inline int bstride(int i) const { return bstride_[i]; }

  __device__ static inline constexpr size_t size() { return N; }

  __device__ inline int elements() const {
    int el = 1;
    for(int i = 0; i < N; ++i)
      el *= shape_[i];
    return el;
  }

  __device__ inline int index(int* d) const {
    int i = 0;
    for(int j = 0; j < N; ++j)
      i += d[j] * stride_[j];
    return i;
  }

  __device__ inline int bindex(int* d) const {
    int i = 0;
    for(int j = 0; j < N; ++j)
      i += d[j] * bstride_[j];
    return i;
  }

  __device__ inline void dims(int i, int* d) const {
    for(int j = 0; j < N; ++j)
      d[j] = (i / stride_[j]) % shape_[j];
    }

  __device__ bool operator==(const ShapeN& other) const {
    for(int i = 0; i < N; ++i)
      if(shape_[i] != other[i])
        return false;
    return true;
  }

  __device__ bool operator!=(const ShapeN& other) const {
    return !(*this == other);
  }
};

typedef ShapeN<4> ShapeGPU;

}
