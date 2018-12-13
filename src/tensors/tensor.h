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

class TensorBase : public std::enable_shared_from_this<TensorBase> {
private:
  Ptr<MemoryPiece> memory_;
  Shape shape_;
  Type type_{Type::float32};
  Ptr<Backend> backend_;

public:
  TensorBase(Ptr<MemoryPiece> memory,
             Shape shape,
             Type type,
             Ptr<Backend> backend)
      : memory_(memory), shape_(shape), type_(type), backend_(backend) {}

  TensorBase(Ptr<MemoryPiece> memory, Shape shape, Ptr<Backend> backend)
      : memory_(memory),
        shape_(shape),
        type_(Type::float32),
        backend_(backend) {}

  ~TensorBase() {}

  virtual void reset(Ptr<MemoryPiece> memory) { memory_ = memory; }

  virtual Ptr<MemoryPiece> memory() { return memory_; }

  virtual Type type() { return type_; }

  virtual Shape& shape() { return shape_; }

  virtual float* data() { return memory_->data<float>(); }

  template <typename T>
  T* data() {
    return memory_->data<T>();
  }

  virtual size_t size() { return shape_.elements(); }

  template <typename T>
  T scalar() {
    ABORT_IF(!matchType<T>(type_),
             "Requested type ({}) and underlying type ({}) do not match",
             request<T>(),
             type_);

    ABORT_IF(size() != 1, "Tensor is not a scalar");
    return get<T>(0);
  }

  virtual float scalar() {
    return scalar<float>();
  }

  Ptr<Backend> getBackend() { return backend_; }
  DeviceId getDeviceId() { return backend_->getDeviceId(); }

  Tensor subtensor(size_t offset, size_t size) {
    auto mem = New<MemoryPiece>(memory_->data() + sizeOf(type_) * offset,
                                sizeOf(type_) * size);
    return New<TensorBase>(mem, Shape{1, (int)size}, backend_);
  }

  template <typename T>
  T get(size_t i) {
    ABORT_IF(!matchType<T>(type_),
             "Requested type ({}) and underlying type ({}) do not match",
             request<T>(),
             type_);

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

  float get(size_t i) {
    return get<float>(i);
  }

  template <typename T>
  void set(size_t i, T value) {
    ABORT_IF(!matchType<T>(type_),
             "Requested type ({}) and underlying type ({}) do not match",
             request<T>(),
             type_);

    if(backend_->getDeviceId().type == DeviceType::cpu) {
      std::copy(&value, &value + 1, data<T>() + i);
    }
#ifdef CUDA_FOUND
    else {
      gpu::copy(backend_, &value, &value + 1, data<T>() + i);
    }
#endif
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

  template <typename T>
  void set(const T* begin, const T* end) {
    ABORT_IF(!matchType<T>(type_),
             "Requested type ({}) and underlying type ({}) do not match",
             request<T>(),
             type_);

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

  template <typename T>
  void set(T value) {
    if(!matchType<T>(type_)) {
      switch(type_) {
        case Type::float32: set<float   >((float   )value); break;
        case Type::float64: set<double  >((double  )value); break;
        case Type::int8:    set<int8_t  >((int8_t  )value); break;
        case Type::int16:   set<int16_t >((int16_t )value); break;
        case Type::int32:   set<int32_t >((int32_t )value); break;
        case Type::int64:   set<int64_t >((int64_t )value); break;
        case Type::uint8:   set<uint8_t >((uint8_t )value); break;
        case Type::uint16:  set<uint16_t>((uint16_t)value); break;
        case Type::uint32:  set<uint32_t>((uint32_t)value); break;
        case Type::uint64:  set<uint64_t>((uint64_t)value); break;
        default:
          ABORT(
              "Requested type ({}) cannot be converted to underlying type ({})",
              request<float>(),
              type_);
      }
    }

    if(backend_->getDeviceId().type == DeviceType::cpu) {
      std::fill(data<T>(), data<T>() + size(), value);
    }
#ifdef CUDA_FOUND
    else {
      gpu::fill(backend_, data<T>(), data<T>() + size(), value);
    }
#endif
  }

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
     switch(type_) {
      case Type::int8:    copyFrom<int8_t>(in);  break;
      case Type::int16:   copyFrom<int16_t>(in); break;
      case Type::int32:   copyFrom<int32_t>(in); break;
      case Type::int64:   copyFrom<int64_t>(in); break;

      case Type::uint8:   copyFrom<uint8_t>(in);  break;
      case Type::uint16:  copyFrom<uint16_t>(in); break;
      case Type::uint32:  copyFrom<uint32_t>(in); break;
      case Type::uint64:  copyFrom<uint64_t>(in); break;

      case Type::float32: copyFrom<float>(in);  break;
      case Type::float64: copyFrom<double>(in); break;

      default: ABORT("Unknown type {}", type_);
    }
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
     switch(type_) {
      case Type::int8:    swap<int8_t>(swapee);  break;
      case Type::int16:   swap<int16_t>(swapee); break;
      case Type::int32:   swap<int32_t>(swapee); break;
      case Type::int64:   swap<int64_t>(swapee); break;

      case Type::uint8:   swap<uint8_t>(swapee);  break;
      case Type::uint16:  swap<uint16_t>(swapee); break;
      case Type::uint32:  swap<uint32_t>(swapee); break;
      case Type::uint64:  swap<uint64_t>(swapee); break;

      case Type::float32: swap<float>(swapee);  break;
      case Type::float64: swap<double>(swapee); break;

      default: ABORT("Unknown type {}", type_);
    }
  }
  
  template <typename T>
  std::string debug() {
    ABORT_IF(!matchType<T>(type_),
             "Requested type ({}) and underlying type ({}) do not match",
             request<T>(),
             type_);

    std::stringstream strm;
    assert(shape_.size());
    strm << shape_;
    strm << " type=" << type_;
    strm << " device=" << backend_->getDeviceId();
    strm << " ptr=" << (size_t)memory_->data();
    strm << " bytes=" << memory_->size();
    strm << std::endl;

    // values
    size_t totSize = shape_.elements();
    std::vector<T> values(totSize);
    get(values);

    int dispCols = 5;
    if(isFloat(type_))
      strm << std::fixed << std::setprecision(8) << std::setfill(' ');
    else
      strm << std::fixed << std::setprecision(0) << std::setfill(' ');

    for(int i = 0; i < values.size(); ++i) {
      std::vector<int> dims;
      shape().dims(i, dims);

      bool disp = true;
      for(int j = 0; j < dims.size(); ++j)
        disp = disp && (dims[j] < dispCols || dims[j] >= shape()[j] - dispCols);

      if(disp) {
        if(dims.back() == 0) {
          bool par = true;
          std::vector<std::string> p;
          for(int j = (int)dims.size() - 1; j >= 0; --j) {
            if(dims[j] != 0)
              par = false;

            p.push_back(par ? "[" : " ");
          }
          for(auto it = p.rbegin(); it != p.rend(); ++it)
            strm << *it;
          strm << " ";
        }

        strm << std::setw(12);
        if(isFloat(type_)) {
          strm << (double)values[i];
        } else if(isSignedInt(type_)) {
          strm << (int64_t)values[i];
        } else {
          strm << (uint64_t)values[i];
        }
        strm << " ";

        if(dims.back() + 1 == shape().back()) {
          for(int j = (int)dims.size() - 1; j >= 0; --j) {
            if(dims[j] + 1 != shape()[j])
              break;
            strm << "]";
          }
          strm << std::endl;
        }

        bool prev = true;
        for(int j = (int)dims.size() - 1; j >= 0; --j) {
          if(j < (int)dims.size() - 1)
            prev = prev && dims[j + 1] + 1 == shape()[j + 1];
          if(prev && dims[j] + 1 == dispCols && shape()[j] > 2 * dispCols) {
            if(j < (int)dims.size() - 1)
              for(int k = 0; k <= j; ++k)
                strm << " ";
            strm << "... ";
            if(j < (int)dims.size() - 1)
              strm << std::endl;
            break;
          }
        }
      }
    }
    strm << std::endl;
    return strm.str();
  }

  std::string debug() {
    switch(type_) {
      case Type::int8: return debug<int8_t>();
      case Type::int16: return debug<int16_t>();
      case Type::int32: return debug<int32_t>();
      case Type::int64: return debug<int64_t>();

      case Type::uint8: return debug<uint8_t>();
      case Type::uint16: return debug<uint16_t>();
      case Type::uint32: return debug<uint32_t>();
      case Type::uint64: return debug<uint64_t>();

      case Type::float32: return debug<float>();
      case Type::float64: return debug<double>();

      default: ABORT("Unknown type {}", type_);
    }
  }
};

typedef std::shared_ptr<TensorBase> Tensor;
}  // namespace marian
