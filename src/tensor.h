#pragma once

#include <cudnn.h>
#include <cublas_v2.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <numeric>
#include <sstream>

#include "definitions.h"
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

const Handles handles;

typedef std::vector<int> Shape;

inline std::string Debug(const Shape &shape)
{
	std::stringstream strm;
	strm << shape[0];
	assert(shape.size());
	for (size_t i = 1; i < shape.size(); ++i) {
		strm << "x" << shape[i];
	}
	return strm.str();
}

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

      std::cerr << "Allocating : " << shape[0] << " " << shape[1] << std::endl;

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

    void set(const std::vector<Float> &values) {
	  size_t totSize = std::accumulate(shape().begin(), shape().end(),
			  1, std::multiplies<int>());
	  std::cerr << "tensor size=" << totSize << " vector size=" << values.size() << std::endl;
	  assert(totSize == values.size());
	  thrust::copy(values.begin(), values.end(), data_.begin());
    }

    std::string Debug() const
    {
    	std::stringstream strm;
    	assert(shape_.size());
    	strm << "shape=" << marian::Debug(shape_);
    	return strm.str();
    }
};

template <typename Type>
size_t TensorImpl<Type>::tensorCounter = 0;

class Tensor {
  private:
    std::shared_ptr<TensorImpl<Float>> pimpl_;
    
  public:
    typedef TensorImpl<Float>::value_type value_type;
    
    Tensor() {}
    Tensor(Shape shape, value_type value = 0) {
      allocate(shape, value);
    }
    
    ~Tensor() {}
    
    void allocate(Shape shape, value_type value = 0) {
      if(!pimpl_)
        pimpl_.reset(new TensorImpl<Float>(shape, value));
    }
    
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

    std::string Debug() const
    {
    	return pimpl_->Debug();
    }

    void Print() const {
      for (int i = 0; i < size(); ++i) {
        std::cerr << (*this)[i] << " ";
      }
      std::cerr << std::endl;
    }

    void Load(const std::string &path);
    void Load(const std::vector<float> &values);

};

}
