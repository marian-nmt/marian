#pragma once

// This file is part of the Marian toolkit.
// Marian is copyright (c) 2016 Marcin Junczys-Dowmunt.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <sstream>
#include <cuda.h>
#include <cudnn.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>

#include "exception.h"
#include "definitions.h"
#include "tensors/tensor.h"

namespace marian {

class TensorGPU : public TensorBase {
  private:
    // cuDNN stuff
    cudnnTensorDescriptor_t cudnnDesc_;

  public:
    TensorGPU(float* data, Shape shape)
    : TensorBase(data, shape) {
      cudnnCreateTensorDescriptor(&cudnnDesc_);
      cudnnSetTensor4dDescriptorEx(cudnnDesc_, CUDNN_DATA_FLOAT,
                                   shape_[0], shape_[1], 1, 1,
                                   shape_[1], 1, 1, 1);
    }

    ~TensorGPU() {
      cudnnDestroyTensorDescriptor(cudnnDesc_);
    }

    __device__
    inline float& operator()(size_t i, size_t j) {
      int rows = shape_[0];
      int cols = shape_[1];
      if(rows != 1 && cols != 1)
        return data_[i * cols + j];
      if(rows != 1 && cols == 1)
        return data_[i];
      if(rows == 1 && cols != 1)
        return data_[j];
      return data_[0];
    }

    __device__
    size_t rows() {
      return shape_[0];
    }

    __device__
    size_t cols() {
      return shape_[1];
    }

    float get(size_t i) {
      return *thrust::device_pointer_cast(data_ + i );
    }

    void set(size_t i, float value) {
      *thrust::device_pointer_cast(data_ + i) = value;
    }

    void get(std::vector<float> &v) {
      v.resize(size());
      thrust::copy(thrust::device_pointer_cast(data_),
                   thrust::device_pointer_cast(data_ + size()),
                   v.begin());
    }

    void set(float value) {
      thrust::fill(thrust::device_pointer_cast(data_),
                   thrust::device_pointer_cast(data_ + size()), value);
    }

    void set(const std::vector<float> &v) {
      thrust::copy(v.begin(), v.end(),
                   thrust::device_pointer_cast(data_));
    }

    cudnnTensorDescriptor_t& cudnn() {
      return cudnnDesc_;
    }

    std::string debug() {
      std::stringstream strm;
      assert(shape_.size());
      strm << "shape=" << shape_[0] << "x" << shape_[1] << std::endl;

      // values
      size_t totSize = shape_.elements();
      std::vector<Float> values(totSize);
      get(values);

      size_t ind = 0;
      for (size_t i = 0; i < shape()[0]; ++i) {
          for (size_t j = 0; j < shape()[1]; ++j) {
              strm << values[ind] << " ";
              ++ind;
          }
          strm << std::endl;
      }
      return strm.str();
    }
};

class DeviceGPU {
  private:
    float* data_;
    size_t size_;

  public:
    DeviceGPU()
    : data_(0), size_(0) {}

    ~DeviceGPU() {
      if(data_)
        cudaFree(data_);
    }

    typedef TensorGPU tensor_type;

    void reserve(size_t size) {
      UTIL_THROW_IF2(size < size_, "New size must be larger than old size");
      float *temp;
      cudaMalloc(&temp, size * sizeof(float));

      if(data_) {
        cudaMemcpy(temp, data_, size_* sizeof(float),
                   cudaMemcpyDeviceToDevice);
        cudaFree(data_);
      }

      data_ = temp;
      size_ = size;
    }

    float* data() {
      return data_;
    }

    size_t capacity() {
      return size_;
    }
};

}
