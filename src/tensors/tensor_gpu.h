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

#include <thrust/device_vector.h>
#include <cudnn.h>

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

    float get(size_t i) {
      return *thrust::device_pointer_cast(data_ + i);
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

    cudnnTensorDescriptor_t cudnn() {
      return cudnnDesc_;
    }
};

class DeviceGPU {
  private:
    thrust::device_vector<float> data_;

  public:
    typedef TensorGPU tensor_type;

    void reserve(size_t size) {
      data_.reserve(size);
      data_.resize(size);
    }

    float* data() {
      return thrust::raw_pointer_cast(data_.data());
    }

    size_t capacity() {
      return data_.size();
    }
};

}
