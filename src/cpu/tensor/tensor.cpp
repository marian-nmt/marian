/* All or part of this file was contributed by Intel under license:
 *   Copyright (C) 2017-2018 Intel Corporation
 *   SPDX-License-Identifier: MIT
 */

#include <omp.h>

#include "kernels/tensor_operators.h"
#include "tensors/tensor.h"

namespace marian {

float TensorCPU::get(size_t i) { return data()[i]; }

void TensorCPU::set(size_t i, float value) { data()[i] = value; }

void TensorCPU::get(std::vector<float> &v) {
  v.clear();
  v.reserve(size());
  std::copy(data(), data() + size(), std::back_inserter(v));
}

void TensorCPU::set(float value) {
  #pragma omp parallel
  {
    int i = omp_get_thread_num(), n = omp_get_num_threads();
    size_t span = size() / n;
    if (span*sizeof(float) < 64) {
      #pragma omp master
      std::fill(data(), data() + size(), value);
    } else {
      float* begin = data() + i*span;
      float* end = i != n-1 ? begin + span : data() + size();
      std::fill(begin, end, value);
    }
  }
}

void TensorCPU::set(const std::vector<float> &v) { std::copy(v.begin(), v.end(), data()); }

void TensorCPU::setSparse(const std::vector<size_t> &k, const std::vector<float> &v) {
  SetSparse(shared_from_this(), k, v);
}

void TensorCPU::copyFrom(Tensor in) { std::copy(in->data(), in->data() + in->size(), data()); }

// TODO: Refactor to reduce duplication with tensor.cu
Tensor TensorCPU::view(const Shape& shape, ptrdiff_t offset) {
  size_t size = shape.elements() * sizeof(float);
  ptrdiff_t offset_bytes = offset * sizeof(float);
  auto mem = New<MemoryPiece>(memory()->data() + offset_bytes, size);
  return New<TensorCPU>(mem, shape, device_);
}

// TODO: Refactor to reduce duplication with tensor.cu
std::string TensorCPU::debug() {
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
  
}
