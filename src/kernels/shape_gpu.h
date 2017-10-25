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

struct ShapeN {
  int* shape_;
  int* stride_;
  int* bstride_;
  size_t size_;

  ShapeN(const ShapeN& shape) {
    size_ = shape.size();

    cudaMalloc((void**)&shape_, sizeof(int) * size_);
    cudaMemcpy(shape_, shape.shape_, sizeof(int) * size_, cudaMemcpyDefault);

    cudaMalloc((void**)&stride_, sizeof(int) * size_);
    cudaMemcpy(stride_, shape.stride_, sizeof(int) * size_, cudaMemcpyDefault);

    cudaMalloc((void**)&bstride_, sizeof(int) * size_);
    cudaMemcpy(bstride_, shape.bstride_, sizeof(int) * size_, cudaMemcpyDefault);
  }

  ShapeN(const Shape& shape) {
    size_ = shape.size();

    cudaMalloc((void**)&shape_, sizeof(int) * size_);
    cudaMemcpy(shape_, shape.shape_.data(), sizeof(int) * size_, cudaMemcpyDefault);

    cudaMalloc((void**)&stride_, sizeof(int) * size_);
    cudaMemcpy(stride_, shape.stride_.data(), sizeof(int) * size_, cudaMemcpyDefault);

    cudaMalloc((void**)&bstride_, sizeof(int) * size_);
    cudaMemcpy(bstride_, shape.bstride_.data(), sizeof(int) * size_, cudaMemcpyDefault);
  }

  ~ShapeN() {
    size_ = 0;
    cudaFree(shape_);
    cudaFree(stride_);
    cudaFree(bstride_);
  }

  __device__ inline void updateStrides() {

    stride_[size_ - 1] = 1;
    bstride_[size_ - 1] = shape_[size_ - 1] == 1 ? 0 : stride_[size_ - 1];

    for(int i = size_ - 2; i >= 0; --i) {
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

  __device__ inline int back() const { return dim(size_ - 1); }

  __device__ inline int operator[](int i) { return dim(i); }

  __device__ inline int operator[](int i) const { return dim(i); }

  __device__ inline int stride(int i) const { return stride_[i]; }

  __device__ inline int bstride(int i) const { return bstride_[i]; }

  __host__ __device__ inline size_t size() const { return size_; }

  __device__ inline int elements() const {
    int el = 1;
    for(int i = 0; i < size_; ++i)
      el *= shape_[i];
    return el;
  }

  __device__ inline int index(int* d) const {
    int i = 0;
    for(int j = 0; j < size_; ++j)
      i += d[j] * stride_[j];
    return i;
  }

  __device__ inline int bindex(int* d) const {
    int i = 0;
    for(int j = 0; j < size_; ++j)
      i += d[j] * bstride_[j];
    return i;
  }

  __device__ inline void dims(int i, int* d) const {
    for(int j = 0; j < size_; ++j)
      d[j] = (i / stride_[j]) % shape_[j];
    }

  __device__ bool operator==(const ShapeN& other) const {
    for(int i = 0; i < size_; ++i)
      if(shape_[i] != other[i])
        return false;
    return true;
  }

  __device__ bool operator!=(const ShapeN& other) const {
    return !(*this == other);
  }
};

typedef ShapeN ShapeGPU;

}
