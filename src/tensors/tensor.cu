
#include <cuda.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>

#include "kernels/cuda_helpers.h"
#include "kernels/tensor_operators.h"
#include "tensors/tensor.h"

namespace marian {

__global__ void gFill(float *d_in, int size, float val) {
  for(int bid = 0; bid < size; bid += blockDim.x * gridDim.x) {
    int index = bid + threadIdx.x + blockDim.x * blockIdx.x;
    if(index < size) {
      d_in[index] = val;
    }
  }
}

float TensorBase::get(size_t i) {
  cudaSetDevice(device_);
  float temp;
  CUDA_CHECK(
      cudaMemcpy(&temp, data() + i, sizeof(float), cudaMemcpyDeviceToHost));
  cudaStreamSynchronize(0);
  return temp;
}

void TensorBase::set(size_t i, float value) {
  cudaSetDevice(device_);
  CUDA_CHECK(
      cudaMemcpy(data() + i, &value, sizeof(float), cudaMemcpyHostToDevice));
  cudaStreamSynchronize(0);
}

void TensorBase::get(std::vector<float> &v) {
  CUDA_CHECK(cudaSetDevice(device_));
  v.resize(size());
  CUDA_CHECK(cudaMemcpy(
      v.data(), data(), size() * sizeof(float), cudaMemcpyDeviceToHost));
  cudaStreamSynchronize(0);
}

void TensorBase::set(float value) {
  cudaSetDevice(device_);
  int threads = std::min(512, (int)size());
  int blocks = (size() / threads) + (size() % threads != 0);
  gFill<<<blocks, threads>>>(data(), size(), value);
  cudaStreamSynchronize(0);
}

void TensorBase::set(const std::vector<float> &v) {
  CUDA_CHECK(cudaSetDevice(device_));
  CUDA_CHECK(cudaMemcpy(
      data(), v.data(), v.size() * sizeof(float), cudaMemcpyHostToDevice));
  cudaStreamSynchronize(0);
}

void TensorBase::setSparse(const std::vector<size_t> &k,
                           const std::vector<float> &v) {
  cudaSetDevice(device_);
  SetSparse(data(), k, v);
  cudaStreamSynchronize(0);
}

void TensorBase::copyFrom(Tensor in) {
  cudaSetDevice(device_);
  CUDA_CHECK(cudaMemcpy(
      data(), (float*)in->data(), in->size() * sizeof(float), cudaMemcpyDefault));
  cudaStreamSynchronize(0);
}

std::string TensorBase::debug() {
  cudaSetDevice(device_);
  std::stringstream strm;
  assert(shape_.size());
  strm << shape_;
  strm << " device=" << device_;
  strm << " ptr=" << (size_t)memory_->data();
  strm << " bytes=" << memory_->size();
  strm << std::endl;

  // values
  size_t totSize = shape_.elements();
  std::vector<float> values(totSize);
  get(values);

  size_t dispCols = 5;
  strm << std::fixed << std::setprecision(8) << std::setfill(' ');
  for(size_t l = 0; l < shape()[3]; ++l) {
    for(size_t k = 0; k < shape()[2]; ++k) {
      strm << "[ ";
      if(shape()[0] > 10) {
        for(size_t i = 0; i < shape()[0] && i < dispCols; ++i) {
          if(i > 0)
            strm << std::endl << "  ";
          for(size_t j = 0; j < shape()[1] && j < dispCols; ++j) {
            strm << std::setw(12)
                 << values[i * shape().stride(0) + j * shape().stride(1)
                           + k * shape().stride(2)
                           + l * shape().stride(3)]
                 << " ";
          }
          if(shape()[1] > dispCols)
            strm << "... ";
          for(size_t j = shape()[1] - dispCols; j < shape()[1]; ++j) {
            strm << std::setw(12)
                 << values[i * shape().stride(0) + j * shape().stride(1)
                           + k * shape().stride(2)
                           + l * shape().stride(3)]
                 << " ";
          }
        }
        strm << std::endl << "  ...";
        for(size_t i = shape()[0] - dispCols; i < shape()[0]; ++i) {
          if(i > 0)
            strm << std::endl << "  ";
          for(size_t j = 0; j < shape()[1] && j < dispCols; ++j) {
            strm << std::setw(12)
                 << values[i * shape().stride(0) + j * shape().stride(1)
                           + k * shape().stride(2)
                           + l * shape().stride(3)]
                 << " ";
          }
          if(shape()[1] > dispCols)
            strm << "... ";
          for(size_t j = shape()[1] - dispCols; j < shape()[1]; ++j) {
            strm << std::setw(12)
                 << values[i * shape().stride(0) + j * shape().stride(1)
                           + k * shape().stride(2)
                           + l * shape().stride(3)]
                 << " ";
          }
        }
      } else {
        for(size_t i = 0; i < shape()[0] && i < 10; ++i) {
          if(i > 0)
            strm << std::endl << "  ";
          for(size_t j = 0; j < shape()[1] && j < dispCols; ++j) {
            strm << std::setw(12)
                 << values[i * shape().stride(0) + j * shape().stride(1)
                           + k * shape().stride(2)
                           + l * shape().stride(3)]
                 << " ";
          }
          if(shape()[1] > dispCols)
            strm << "... ";
          for(size_t j = shape()[1] - dispCols; j < shape()[1]; ++j) {
            strm << std::setw(12)
                 << values[i * shape().stride(0) + j * shape().stride(1)
                           + k * shape().stride(2)
                           + l * shape().stride(3)]
                 << " ";
          }
        }
      }
      strm << "]" << std::endl;
    }
  }
  return strm.str();
}

Tensor operator<<(Tensor t, const std::vector<float> &v) {
  t->set(v);
  return t;
}

Tensor operator>>(Tensor t, std::vector<float> &v) {
  t->get(v);
  return t;
}
}
