#pragma once

#include "common/definitions.h"
#include "common/shape.h"
#include "common/types.h"
#include "tensors/backend.h"
#include "tensors/memory_piece.h"
#ifdef CUDA_FOUND
#include "tensors/gpu/algorithm.h"
#endif

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>

namespace marian {

namespace io {
  struct Item;
}

/**
 * Main implementation of a <a href="https://en.wikipedia.org/wiki/Tensor">tensor</a>,
 * a multi-dimensional matrix containing elements of a single data type.
 * TensorBase contains the data, data type, pointer to
 * memory region, shape, backend info and other attributes.
 */
class TensorBase {
  MemoryPiece::PtrType memory_;
  Shape shape_;
  Type type_{Type::float32};
  Ptr<Backend> backend_;

  ENABLE_INTRUSIVE_PTR(TensorBase)

protected:
  // Constructors are protected, use TensorBase::New(...)
  TensorBase(MemoryPiece::PtrType memory,
             Shape shape,
             Type type,
             Ptr<Backend> backend)
      : memory_(memory), shape_(shape), type_(type), backend_(backend) {}

  TensorBase(MemoryPiece::PtrType memory, 
             Shape shape, 
             Ptr<Backend> backend)
      : memory_(memory),
        shape_(shape),
        type_(Type::float32),
        backend_(backend) {}

  // Wraps existing memory
  template <typename T>
  TensorBase(T* rawMemory,
             size_t rawMemoryNum,
             Shape shape,
             Type type,
             Ptr<Backend> backend)
      : memory_(MemoryPiece::New((uint8_t*)rawMemory, rawMemoryNum * sizeof(T))), 
        shape_(shape), type_(type), backend_(backend) {}

public:
  // Use this whenever pointing to TensorBase
  typedef IPtr<TensorBase> PtrType;

  // Use this whenever creating a pointer to TensorBase
  template <class ...Args>
  static PtrType New(Args&& ...args) {
    return PtrType(new TensorBase(std::forward<Args>(args)...));
  }

  virtual ~TensorBase() {}

  virtual void reset(MemoryPiece::PtrType memory) { memory_ = memory; }

  virtual MemoryPiece::PtrType memory() { return memory_; }

  virtual Type type() { return type_; }

  virtual Shape& shape() { return shape_; }

  virtual float* data() { return memory_->data<float>(); }

  template <typename T>
  T* data() {
    return memory_->data<T>();
  }

  virtual size_t size() { return shape_.elements(); }

  // this version of scalar will abort if numeric types do not match
  template <typename T>
  T scalar() {
    ABORT_IF(size() != 1, "Tensor is not a scalar");
    return get<T>(0);
  }

  // this non-template version converts all numeric types to float
  virtual float scalar() {
    DISPATCH_BY_TYPE0(type_, (float)scalar);
  }

  Ptr<Backend> getBackend() { return backend_; }
  DeviceId getDeviceId() { return backend_->getDeviceId(); }

  Tensor subtensor(size_t offset, size_t size) {
    auto mem = MemoryPiece::New(memory_->data() + sizeOf(type_) * offset, sizeOf(type_) * size);
    return TensorBase::New(mem, Shape{1, (int)size}, type(), backend_);
  }

  // @TODO: review if we can eliminate GPU-specific code here, 
  // potentially by moving this to non-class members.
  template <typename T>
  T get(size_t i) {
    if(!matchType<T>(type_)) {
      DISPATCH_BY_TYPE1(type_, (T)get, i);
    } else {
      T temp = 0;
      if(backend_->getDeviceId().type == DeviceType::cpu) {
        std::copy(data<T>() + i, data<T>() + i + 1, &temp);
      }
  #ifdef CUDA_FOUND
      else {
        gpu::copy(backend_, data<T>() + i, data<T>() + i + 1, &temp);
      }
  #endif
      return temp;
    }
  }

  float get(size_t i) {
    return get<float>(i);
  }

  template <typename T>
  void get(std::vector<T>& v) {
    ABORT_IF(!matchType<T>(type_),
             "Requested type ({}) and underlying type ({}) do not match",
             request<T>(),
             type_);

    v.resize(size());
    if(backend_->getDeviceId().type == DeviceType::cpu) {
      std::copy(data<T>(), data<T>() + size(), v.data());
    }
#ifdef CUDA_FOUND
    else {
      gpu::copy(backend_, data<T>(), data<T>() + size(), v.data());
    }
#endif
  }

  void get(io::Item& item, const std::string& name);

  template <typename T>
  void set(size_t i, T value) {
    if(!matchType<T>(type_)) {
      DISPATCH_BY_TYPE2(type_, set, i, value);
    } else {
      if(backend_->getDeviceId().type == DeviceType::cpu) {
        std::copy(&value, &value + 1, data<T>() + i);
      }
#ifdef CUDA_FOUND
      else {
        gpu::copy(backend_, &value, &value + 1, data<T>() + i);
      }
#endif
    }
  }

  template <typename T>
  void set(const T* begin, const T* end) {
    ABORT_IF(end - begin != shape_.elements(),
             "Vector size ({}) and underlying shape ({}, {}) do not match",
             end - begin,
             std::string(shape_),
             memory_->size());
    matchOrAbort<T>(type_);

    if(backend_->getDeviceId().type == DeviceType::cpu) {
      std::copy(begin, end, data<T>());
    }
#ifdef CUDA_FOUND
    else {
      gpu::copy(backend_, begin, end, data<T>());
    }
#endif
  }

  template <typename T>
  void set(const std::vector<T>& v) {
    set(v.data(), v.data() + v.size());
  }

  // a binary copy with type checking
  void set(const char* begin, const char* end, Type type) {
    ABORT_IF(type_ != type,
             "Tensor type ({}) and data type ({}) do not match",
             type_,
             type);

    size_t dataSize = (end - begin) / sizeOf(type);
    ABORT_IF(size() != dataSize,
             "Tensor size ({}) and mapped size ({}) do not match",
             size(),
             dataSize);

    if(backend_->getDeviceId().type == DeviceType::cpu) {
      std::copy(begin, end, data<char>());
    }
#ifdef CUDA_FOUND
    else {
      gpu::copy(backend_, begin, end, data<char>());
    }
#endif
  }

  void set(const std::vector<char>& v, Type type) {
    set(v.data(), v.data() + v.size(), type);
  }

  void set(const io::Item& item);

  // For single values enable conversion to other numeric formats if possible
  template <typename T>
  void set(T value) {
    if(!matchType<T>(type_)) {
      DISPATCH_BY_TYPE1(type_, setAs, value);
    } else {
      if(backend_->getDeviceId().type == DeviceType::cpu) {
        std::fill(data<T>(), data<T>() + size(), value);
      }
  #ifdef CUDA_FOUND
      else {
        gpu::fill(backend_, data<T>(), data<T>() + size(), value);
      }
  #endif
    }
  }
private: // subroutine for above: helper that accepts any type and casts it to <T>
    template <typename Tas, typename Tval>
    void setAs(Tval value) { set((Tas)value); }
public:

  void setSparse(const std::vector<size_t>& k, const std::vector<float>& v) {
    ABORT_IF(!matchType<float>(type_),
             "Requested type ({}) and underlying type ({}) do not match",
             request<float>(),
             type_);

    if(backend_->getDeviceId().type == DeviceType::cpu) {
      for(size_t i = 0; i < k.size(); ++i)
        data()[k[i]] = v[i];
    }
#ifdef CUDA_FOUND
    else {
      gpu::setSparse(backend_, k, v, data());
    }
#endif
  }

  template <typename T>
  void copyFrom(Tensor in) {
    ABORT_IF(in->shape() != shape_, "Can only copy tensors with equal shapes ({} != {})", in->shape(), shape_);
    ABORT_IF(in->type()  != type_,  "Can only copy tensors with equal types ({} != {})",  in->type(),  type_);
    ABORT_IF(!matchType<T>(type_),
             "Requested type ({}) and underlying type ({}) do not match",
             request<T>(),
             type_);

    if(in->getBackend()->getDeviceId().type == DeviceType::cpu
       && backend_->getDeviceId().type == DeviceType::cpu) {
      std::copy(in->data<T>(), in->data<T>() + in->size(), data<T>());
    }
#ifdef CUDA_FOUND
    else {
      gpu::copy(backend_, in->data<T>(), in->data<T>() + in->size(), data<T>());
    }
#endif
  }

  void copyFrom(Tensor in) {
    DISPATCH_BY_TYPE1(type_, copyFrom, in);
  }

  // Swaps the contents of the current tensor with the argument tensor
  template <typename T>
  void swap(Tensor swapee) {
    ABORT_IF(swapee->shape() != shape_, "Can only swap tensors with equal shapes ({} != {})", swapee->shape(), shape_);
    ABORT_IF(swapee->type()  != type_,  "Can only swap tensors with equal types ({} != {})",  swapee->type(),  type_);
    ABORT_IF(!matchType<T>(type_),
             "Requested type ({}) and underlying type ({}) do not match",
             request<T>(),
             type_);

    // we live on CPUs; just use stdlib
    if(swapee->getBackend()->getDeviceId().type == DeviceType::cpu
       && backend_->getDeviceId().type == DeviceType::cpu) {
      std::swap_ranges(swapee->data<T>(), swapee->data<T>() + swapee->size(), data<T>());
    }
#ifdef CUDA_FOUND
    else {
      if(backend_->getDeviceId() == swapee->getBackend()->getDeviceId()) {
        // we live on the same GPU; do an element-wise swap
        gpu::swap_ranges(backend_, swapee->data<T>(), swapee->data<T>() + swapee->size(), data<T>());
      } else {
        // we live on two different GPUs or devices; go through CPU RAM
        std::vector<T> temp;
        get(temp);
        copyFrom(swapee);
        swapee->set(temp);
      }
    }
#endif
  }

  void swap(Tensor swapee) {
    DISPATCH_BY_TYPE1(type_, swap, swapee);
  }

  template <typename T>
  std::string debug(int precision = 8, int dispCols = 5);

  std::string debug(int precision = 8, int dispCols = 5) {
    DISPATCH_BY_TYPE2(type_, debug, precision, dispCols);
  }

  size_t hash();

};

typedef TensorBase::PtrType Tensor;

template <class TensorType0, class ...TensorTypeRest>
static inline void checkCommonType(TensorType0 first, TensorTypeRest ...rest) {
  std::vector<Tensor> vTensors({first, rest...});
  Type firstType = first->type();
  for(int i = 1; i < vTensors.size(); ++i) {
    ABORT_IF(vTensors[i]->type() != firstType,
             "Type of tensor {} is different from type of tensor 0 ({} != {})",
             i, vTensors[i]->type(), firstType);
  }
}

}  // namespace marian

