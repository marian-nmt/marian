#pragma once

#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>

#include "common/definitions.h"
#include "common/shape.h"
#include "tensors/memory_piece.h"
#include "tensors/backend.h"

#include <algorithm>
#include "tensors/gpu/algorithm.h"

namespace marian {

class TensorBase : public std::enable_shared_from_this<TensorBase> {
private:
  Ptr<MemoryPiece> memory_;
  Shape shape_;
  Ptr<Backend> backend_;

public:
  TensorBase(Ptr<MemoryPiece> memory, Shape shape, Ptr<Backend> backend)
      : memory_(memory), shape_(shape), backend_(backend) {}

  ~TensorBase() {}

  virtual void reset(Ptr<MemoryPiece> memory) { memory_ = memory; }

  virtual Ptr<MemoryPiece> memory() { return memory_; }

  virtual Shape& shape() { return shape_; }

  virtual float* data() { return (float*)memory_->data(); }

  virtual size_t size() { return shape_.elements(); }

  virtual float scalar() {
    ABORT_IF(size() != 1, "Tensor is not a scalar");
    return get(0);
  }

  Ptr<Backend> getBackend() { return backend_; }
  DeviceId getDevice() { return backend_->getDevice(); }

  Tensor subtensor(int offset, int size) {
    auto mem = New<MemoryPiece>(memory_->data() + sizeof(float) * offset,
                                sizeof(float) * size);
    return New<TensorBase>(mem, Shape{1, size}, backend_);
  }

  float get(size_t i) {
    float temp;
    if(backend_->getDevice().type == DeviceType::gpu)
      gpu::copy(backend_, data() + i, data() + i + 1, &temp);
    else
      std::copy(data() + i, data() + i + 1, &temp);
    return temp;
  }

  void set(size_t i, float value) {
    if(backend_->getDevice().type == DeviceType::gpu)
      gpu::copy(backend_, &value, &value + 1, data() + i);
    else
      std::copy(&value, &value + 1, data() + i);
  }

  void get(std::vector<float> &v) {
    v.resize(size());
    if(backend_->getDevice().type == DeviceType::gpu)
      gpu::copy(backend_, data(), data() + size(), v.data());
    else
      std::copy(data(), data() + size(), v.data());
  }

  void set(const std::vector<float> &v) {
    if(backend_->getDevice().type == DeviceType::gpu)
      gpu::copy(backend_, v.data(), v.data() + v.size(), data());
    else
      std::copy(v.data(), v.data() + v.size(), data());
  }

  void set(float value) {
    if(backend_->getDevice().type == DeviceType::gpu)
      gpu::fill(backend_, data(), data() + size(), value);
    else
      std::fill(data(), data() + size(), value);
  }

  void setSparse(const std::vector<size_t> &k,
                 const std::vector<float> &v) {
    if(backend_->getDevice().type == DeviceType::gpu) {
      gpu::setSparse(backend_, k, v, data());
    } else {
      for(int i = 0; i < k.size(); ++i)
        data()[k[i]] = v[i];
    }
  }

  void copyFrom(Tensor in) {
    if(in->getBackend()->getDevice().type == DeviceType::gpu ||
       backend_->getDevice().type == DeviceType::gpu)
      gpu::copy(backend_, in->data(), in->data() + in->size(), data());
    else
      std::copy(in->data(), in->data() + in->size(), data());
  }

  std::string debug() {
    std::stringstream strm;
    assert(shape_.size());
    strm << shape_;
    strm << " device=" << backend_->getDevice();
    strm << " ptr=" << (size_t)memory_->data();
    strm << " bytes=" << memory_->size();
    strm << std::endl;

    // values
    size_t totSize = shape_.elements();
    std::vector<float> values(totSize);
    get(values);

    size_t dispCols = 5;
    strm << std::fixed << std::setprecision(8) << std::setfill(' ');

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
          for(int j = dims.size() - 1; j >= 0; --j) {
            if(dims[j] != 0)
              par = false;

            p.push_back(par ? "[" : " ");
          }
          for(auto it = p.rbegin(); it != p.rend(); ++it)
            strm << *it;
          strm << " ";
        }

        strm << std::setw(12)
             << values[i]
             << " ";

        if(dims.back() + 1 == shape().back()) {
          for(int j = dims.size() - 1; j >= 0; --j) {
            if(dims[j] + 1 != shape()[j])
              break;
            strm << "]";
          }
          strm << std::endl;
        }

        bool prev = true;
        for(int j = dims.size() - 1; j >= 0; --j) {
          if(j < dims.size() - 1)
            prev = prev && dims[j + 1] + 1 == shape()[j + 1];
          if(prev && dims[j] + 1 == dispCols && shape()[j] > 2 * dispCols) {
            if(j < dims.size() - 1)
              for(int k = 0; k <= j; ++k)
                strm << " ";
            strm << "... ";
            if(j < dims.size() - 1)
              strm << std::endl;
            break;
          }
        }
      }
    }
    strm << std::endl;
    return strm.str();
  }

};

typedef std::shared_ptr<TensorBase> Tensor;

static Tensor operator<<(Tensor t, const std::vector<float> &v) {
  t->set(v);
  return t;
}

static Tensor operator>>(Tensor t, std::vector<float> &v) {
  t->get(v);
  return t;
}

}
