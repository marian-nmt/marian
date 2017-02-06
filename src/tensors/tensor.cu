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


#include <cuda.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>

#include "tensors/tensor.h"
#include "kernels/tensor_operators.h"
#include "kernels/cuda_helpers.h"

namespace marian {

__global__ void gFill(float* d_in, int size, float val) {
  for(int bid = 0; bid < size; bid += blockDim.x * gridDim.x) {
    int index = bid + threadIdx.x + blockDim.x * blockIdx.x;
    if (index < size) {
      d_in[index] = val;
    }
  }
}

float TensorBase::get(size_t i) {
   cudaSetDevice(device_);
   float temp;
   CUDA_CHECK(cudaMemcpy(&temp, data_ + i, sizeof(float),
              cudaMemcpyDeviceToHost));
   cudaStreamSynchronize(0);
   return temp;
 }

void TensorBase::set(size_t i, float value) {
  cudaSetDevice(device_);
  CUDA_CHECK(cudaMemcpy(data_ + i, &value, sizeof(float),
             cudaMemcpyHostToDevice));
  cudaStreamSynchronize(0);
}

void TensorBase::get(std::vector<float> &v) {
  CUDA_CHECK(cudaSetDevice(device_));
  v.resize(size());
  CUDA_CHECK(cudaHostRegister(v.data(), v.size() * sizeof(float),
                              cudaHostRegisterPortable));
  CUDA_CHECK(cudaMemcpy(v.data(), data_, size() * sizeof(float),
             cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaHostUnregister(v.data()));
  cudaStreamSynchronize(0);
}

void TensorBase::set(float value) {
  cudaSetDevice(device_);
  int threads = std::min(512, (int)size());
  int blocks = (size() / threads) + (size() % threads != 0);
  gFill<<<blocks, threads>>>(data_, size(), value);
  cudaStreamSynchronize(0);
}

void TensorBase::set(const std::vector<float> &v) {
  CUDA_CHECK(cudaSetDevice(device_));
  CUDA_CHECK(cudaHostRegister(const_cast<float*>(v.data()), v.size() * sizeof(float),
                              cudaHostRegisterPortable));
  CUDA_CHECK(cudaMemcpy(data_, v.data(), v.size() * sizeof(float),
             cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaHostUnregister(const_cast<float*>(v.data())));
  cudaStreamSynchronize(0);
}

void TensorBase::copyFrom(Tensor in) {
    cudaSetDevice(device_);
    CUDA_CHECK(cudaMemcpy(data_, in->data(), in->size() * sizeof(float),
                          cudaMemcpyDefault));
    cudaStreamSynchronize(0);
}

std::string TensorBase::debug() {
  cudaSetDevice(device_);
  std::stringstream strm;
  assert(shape_.size());
  strm << "shape=" << shape_[0];
  for(int i = 1; i < shape_.size(); ++i)
     strm << "x" << shape_[i];
  strm << " size=" << shape_.elements()
     << " (" << shape_.elements() * sizeof(float) << "B)" << std::endl;

  // values
  size_t totSize = shape_.elements();
  std::vector<Float> values(totSize);
  get(values);


  strm << std::fixed << std::setprecision(8) << std::setfill(' ');
  for(size_t k = 0; k < shape()[2]; ++k) {
     strm << "[ ";
     if(shape()[0] > 10) {
        for (size_t i = 0; i < shape()[0] && i < 3; ++i) {
           if(i > 0)
             strm << std::endl << "  ";
           for (size_t j = 0; j < shape()[1] && j < 3; ++j) {
             strm << std::setw(12)
                  << values[  i * shape().stride(0)
                            + j * shape().stride(1)
                            + k * shape().stride(2) ] << " ";
           }
           if(shape()[1] > 3)
              strm << "... ";
           for (size_t j = shape()[1] - 3; j < shape()[1]; ++j) {
             strm << std::setw(12)
                  << values[  i * shape().stride(0)
                            + j * shape().stride(1)
                            + k * shape().stride(2) ] << " ";
           }
        }
        strm << std::endl << "  ...";
        for (size_t i = shape()[0] - 3; i < shape()[0]; ++i) {
           if(i > 0)
             strm << std::endl << "  ";
           for (size_t j = 0; j < shape()[1] && j < 3; ++j) {
             strm << std::setw(12)
                  << values[  i * shape().stride(0)
                            + j * shape().stride(1)
                            + k * shape().stride(2) ] << " ";
           }
           if(shape()[1] > 3)
              strm << "... ";
           for (size_t j = shape()[1] - 3; j < shape()[1]; ++j) {
             strm << std::setw(12)
                  << values[  i * shape().stride(0)
                            + j * shape().stride(1)
                            + k * shape().stride(2) ] << " ";
           }
        }
     }
     else {
        for (size_t i = 0; i < shape()[0] && i < 10; ++i) {
           if(i > 0)
             strm << std::endl << "  ";
           for (size_t j = 0; j < shape()[1] && j < 3; ++j) {
             strm << std::setw(12)
                  << values[  i * shape().stride(0)
                            + j * shape().stride(1)
                            + k * shape().stride(2) ] << " ";
           }
           if(shape()[1] > 3)
              strm << "... ";
           for (size_t j = shape()[1] - 3; j < shape()[1]; ++j) {
             strm << std::setw(12)
                  << values[  i * shape().stride(0)
                            + j * shape().stride(1)
                            + k * shape().stride(2) ] << " ";
           }
        }
     }
     strm << "]" << std::endl;
  }
  return strm.str();
}

DeviceGPU::~DeviceGPU() {
   cudaSetDevice(device_);
   if(data_)
     CUDA_CHECK(cudaFree(data_));
}

void DeviceGPU::reserve(size_t size) {
   cudaSetDevice(device_);
   
   UTIL_THROW_IF2(size < size_, "New size must be larger than old size");

   if(data_) {
     // Allocate memory by going through host memory
     float *temp = new float[size_];
     CUDA_CHECK(cudaMemcpy(temp, data_, size_* sizeof(float),
                cudaMemcpyDeviceToHost));
     CUDA_CHECK(cudaFree(data_));
     CUDA_CHECK(cudaMalloc(&data_, size * sizeof(float)));
     CUDA_CHECK(cudaMemcpy(data_, temp, size_* sizeof(float),
                cudaMemcpyHostToDevice));
     delete[] temp;
   }
   else {
      CUDA_CHECK(cudaMalloc(&data_, size * sizeof(float)));
   }

   size_ = size;
}

Tensor operator<<(Tensor t, const std::vector<float>& v) {
  t->set(v);
  return t;
}

Tensor operator>>(Tensor t, std::vector<float>& v) {
  t->get(v);
  return t;
}

}
