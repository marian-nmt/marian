#pragma once

#include <memory>
#include <functional>
#include <vector>
#include <cmath>

#include <cudnn.h>
#include <cublas_v2.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>

#include "exception.h"
#include "thrust_functions.h"

namespace marian {

struct Handles {
  cudnnHandle_t cudnnHandle;
  cublasHandle_t cublasHandle;
  
  cudnnOpTensorDescriptor_t add;  
  
  Handles() {
    cudnnCreate(&cudnnHandle);
    cublasCreate(&cublasHandle);
    cudnnCreateOpTensorDescriptor(&add);
    cudnnSetOpTensorDescriptor(add, CUDNN_OP_TENSOR_ADD, CUDNN_DATA_FLOAT, CUDNN_NOT_PROPAGATE_NAN);
  }
  
  ~Handles() {
    cudnnDestroy(cudnnHandle);
    cublasDestroy(cublasHandle);
    cudnnDestroyOpTensorDescriptor(add);
  }
};

Handles handles;

typedef std::vector<int> Shape;

template<class Float>
class TensorImpl {
  private:
    Shape shape_;
    thrust::device_vector<Float> data_;
    cudnnTensorDescriptor_t desc_;
    size_t tno_;
    static size_t tensorCounter;
    
    cudnnDataType_t dataType() {
      switch(sizeof(Float)) {
        case 2: return CUDNN_DATA_HALF;
        case 8: return CUDNN_DATA_DOUBLE;
        default: return CUDNN_DATA_FLOAT;
      }
    }

  public:
    typedef Float value_type;
    
    TensorImpl(const Shape& shape, value_type value = 0)
    : shape_(shape), tno_(tensorCounter++)
    {
      // @TODO: 
      UTIL_THROW_IF2(shape_.size() != 2,
                     "For now, only 2D Tensors, will be fixed later.");
      
      UTIL_THROW_IF2(shape_.size() < 1 || shape_.size() > 4,
                     "Wrong number of dimensions: " << shape_.size());
      int size = std::accumulate(shape_.begin(), shape_.end(),
                                 1, std::multiplies<int>());
      data_.resize(size, value);
      cudnnCreateTensorDescriptor(&desc_);
      switch (shape_.size()) {
        case 1:
          cudnnSetTensor4dDescriptor(desc_, CUDNN_TENSOR_NCHW, dataType(),
                                     shape_[0], 1, 1, 1); break;
        case 2:
          cudnnSetTensor4dDescriptor(desc_, CUDNN_TENSOR_NCHW, dataType(),
                                     shape_[0], shape_[1], 1, 1); break;
        case 3:
          cudnnSetTensor4dDescriptor(desc_, CUDNN_TENSOR_NCHW, dataType(),
                                     shape_[0], shape_[1], shape_[2], 1); break;
        case 4:
          cudnnSetTensor4dDescriptor(desc_, CUDNN_TENSOR_NCHW, dataType(),
                                     shape_[0], shape_[1], shape_[2], shape_[3]); break;
      }
    }
   
    TensorImpl(const TensorImpl&) = delete;
    TensorImpl(TensorImpl&&) = delete;
         
    ~TensorImpl() {
      cudnnDestroyTensorDescriptor(desc_);
    }
   
   value_type operator[](size_t i) const {
      return data_[i];
    }
   
    auto begin() -> decltype( data_.begin() ) {
      return data_.begin();
    }
   
    auto begin() const -> decltype( data_.begin() ) {
      return data_.begin();
    }
   
    auto end() -> decltype( data_.end() ) {
      return data_.end();
    }
   
    auto end() const -> decltype( data_.end() ) {
      return data_.end();
    }
   
    const Shape& shape() const {
        return shape_;
    }
    
    size_t size() const {
      return data_.size();
    }
    
    value_type* data() {
      return thrust::raw_pointer_cast(data_.data());
    }
    
    cudnnTensorDescriptor_t desc() const {
      return desc_;
    }
    
    size_t id() const {
      return tno_;
    }
    
    void set(value_type value) {
      thrust::fill(data_.begin(), data_.end(), value);
    }
};

template <typename Type>
size_t TensorImpl<Type>::tensorCounter = 0;

class Tensor {
  private:
    std::shared_ptr<TensorImpl<float>> pimpl_;
    
  public:
    typedef TensorImpl<float>::value_type value_type;
    
    Tensor(const Shape& shape, value_type value = 0)
      : pimpl_(new TensorImpl<value_type>(shape, value)) {}
    
    // Single value with broadcasting super powers. Might be
    // worth getting rid of this performance-wise, but is saves
    // so much typing when defining operators.
    Tensor(value_type value)
      : pimpl_(new TensorImpl<value_type>({1, 1}, value)) {}
    
    Tensor() {}
    
    ~Tensor() {}
    
    value_type operator[](size_t i) const {
      return (*pimpl_)[i];
    }
    
    size_t size() const {
      return pimpl_->size();
    }
    
    value_type* data() {
      return pimpl_->data();
    }
    
    const value_type* data() const {
      return pimpl_->data();
    }
    
    auto begin() -> decltype( pimpl_->begin() ) {
      return pimpl_->begin();
    }
   
    auto begin() const -> decltype( pimpl_->begin() ) {
      return pimpl_->begin();
    }
   
    auto end() -> decltype( pimpl_->begin() ) {
      return pimpl_->begin();
    }
   
    auto end() const -> decltype( pimpl_->begin() ) {
      return pimpl_->begin();
    }
    
    const Shape& shape() const {
      return pimpl_->shape();
    }
    
    cudnnTensorDescriptor_t desc() const {
      return pimpl_->desc();
    }
    
    void set(value_type value) {
      pimpl_->set(value);
    }
    
    size_t id() const {
      return pimpl_->id();
    }
    
    operator bool() {
      return pimpl_ != nullptr;
    }
};

Tensor uniform(Tensor t, float a=-0.1, float b=0.1) {
  std::vector<float> r(t.size());
  for(int i = 0; i < r.size(); i++)
    r[i] = (float(rand() % 2000) - 1000.0)/10000.0;
  thrust::copy(r.begin(), r.end(), t.begin());
  return t;
};

using namespace thrust::placeholders;
#define MAX_THREADS 512
#define MAX_BLOCKS 65535

template <class Functor>
__global__ void gElement(Functor functor, float* out,
                         size_t rows, size_t cols) {
  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      float* rowOut = out + j * cols;
      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int i = tid + threadIdx.x;
        if(i < cols)
          rowOut[i] = functor(rowOut[i]);;
      }
    }
  }
}

template <class Functor>
__global__ void gElement(Functor functor,
                         float* out, const float* in,
                         size_t rows, size_t cols) {
  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      float* rowOut = out + j * cols;
      const float* rowIn = in + j * cols;

      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int i = tid + threadIdx.x;
        if(i < cols)
          rowOut[i] = functor(rowOut[i], rowIn[i]);;
      }
    }
  }
}

template <class Functor>
__global__ void gElement(Functor functor,
                         float* out, const float* in1, const float* in2,
                         size_t rows, size_t cols) {
  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      float* rowOut = out + j * cols;
      const float* rowIn1 = in1 + j * cols;
      const float* rowIn2 = in2 + j * cols;

      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int i = tid + threadIdx.x;
        if(i < cols)
          rowOut[i] = functor(rowOut[i], rowIn1[i], rowIn2[i]);
      }
    }
  }
}

template <class Functor>
__global__ void gElement(Functor functor,
                         float* out, const float* in1,
                         const float* in2, const float* in3,
                         size_t rows, size_t cols) {
  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      float* rowOut = out + j * cols;
      const float* rowIn1 = in1 + j * cols;
      const float* rowIn2 = in2 + j * cols;
      const float* rowIn3 = in3 + j * cols;

      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int i = tid + threadIdx.x;
        if(i < cols)
          rowOut[i] = functor(rowOut[i], rowIn1[i], rowIn2[i], rowIn3[i]);
      }
    }
  }
}

// @TODO add broadcasting

template <class Functor>
void Element(Functor functor, Tensor Out) {
  float* d_out = Out.data();
  int blocks  = std::min(MAX_BLOCKS, (int)Out.shape()[0]);
  int threads = std::min(MAX_THREADS, (int)Out.shape()[1]);
  gElement<<<blocks, threads>>>(functor, d_out,
                                Out.shape()[0], Out.shape()[1]);
  cudaStreamSynchronize(0);
}

template <class Functor>
void Element(Functor functor,
             Tensor Out, const Tensor In) {
  float* d_out = Out.data();
  const float* d_in = In.data();

  int blocks  = std::min(MAX_BLOCKS, (int)Out.shape()[0]);
  int threads = std::min(MAX_THREADS, (int)Out.shape()[1]);
  gElement<<<blocks, threads>>>(functor, d_out, d_in,
                                Out.shape()[0], Out.shape()[1]);
  cudaStreamSynchronize(0);
}

template <class Functor>
void Element(Functor functor,
             Tensor Out, const Tensor In1, const Tensor In2) {
  
  float* d_out = Out.data();
  const float* d_in1 = In1.data();
  const float* d_in2 = In2.data();
  
  int blocks  = std::min(MAX_BLOCKS, (int)Out.shape()[0]);
  int threads = std::min(MAX_THREADS, (int)Out.shape()[1]);
  gElement<<<blocks, threads>>>(functor, d_out, d_in1, d_in2,
                                Out.shape()[0], Out.shape()[1]);
  cudaStreamSynchronize(0);
}

template <class Functor>
void Element(Functor functor,
             Tensor Out, const Tensor In1,
             const Tensor In2, const Tensor In3) {
  
  float* d_out = Out.data();
  const float* d_in1 = In1.data();
  const float* d_in2 = In2.data();
  const float* d_in3 = In3.data();
  
  int blocks  = std::min(MAX_BLOCKS, (int)Out.shape()[0]);
  int threads = std::min(MAX_THREADS, (int)Out.shape()[1]);
  gElement<<<blocks, threads>>>(functor, d_out, d_in1, d_in2, d_in3,
                                Out.shape()[0], Out.shape()[1]);
  cudaStreamSynchronize(0);
}

Tensor Prod(cublasHandle_t handle, Tensor C, const Tensor A, const Tensor B,
             bool transA, bool transB, float beta) {
  float alpha = 1.0;

  size_t m = A.shape()[0];
  size_t k = A.shape()[1];
  if(transA)
    std::swap(m, k);
  
  size_t l = B.shape()[0];
  size_t n = B.shape()[1];
  if(transB)
    std::swap(l, n);
  
  size_t lda = A.shape()[1];
  size_t ldb = B.shape()[1];
  size_t ldc = B.shape()[1];
  
  if(transB)
    ldc = B.shape()[0];
  
  cublasOperation_t opA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t opB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;
  
  cublasSgemm(handle, opB, opA,
              n, m, k, &alpha, B.data(), ldb, A.data(), lda, &beta, C.data(), ldc);
  return C;
}

Tensor Prod(Tensor C, const Tensor A, const Tensor B,
             bool transA, bool transB, float beta = 0) {

  return Prod(handles.cublasHandle, C, A, B, transA, transB, beta);
}

}