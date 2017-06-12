#pragma once

#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>

#include "3rd_party/exception.h"
#include "common/definitions.h"
#include "common/shape.h"
#include "tensors/memory_piece.h"

namespace marian {

class TensorBase : public std::enable_shared_from_this<TensorBase> {
private:
  Ptr<MemoryPiece> memory_;
  Shape shape_;
  size_t device_;

public:
  TensorBase(Ptr<MemoryPiece> memory, Shape shape, size_t device)
      : memory_(memory), shape_(shape), device_(device) {}

  ~TensorBase() {}

  virtual void reset(Ptr<MemoryPiece> memory) { memory_ = memory; }

  virtual Ptr<MemoryPiece> memory() { return memory_; }

  virtual Shape& shape() { return shape_; }

  virtual float* data() { return (float*)memory_->data(); }

  virtual size_t size() { return shape_.elements(); }

  virtual float scalar() {
    UTIL_THROW_IF2(size() != 1, "Tensor is not a scalar");
    return get(0);
  }

  size_t getDevice() { return device_; }

  Tensor subtensor(int offset, int size) {
    auto mem = New<MemoryPiece>(memory_->data() + sizeof(float) * offset,
                                sizeof(float) * size);
    return Tensor(new TensorBase(mem, {1, size}, device_));
  }

  float get(size_t i);

  void set(size_t i, float value);

  void get(std::vector<float>& v);

  void set(float value);

  void set(const std::vector<float>& v);

  void setSparse(const std::vector<size_t>& k, const std::vector<float>& v);

  void copyFrom(Tensor);

  std::string debug();
};

typedef std::shared_ptr<TensorBase> Tensor;

Tensor operator<<(Tensor t, const std::vector<float>& v);

Tensor operator>>(Tensor t, std::vector<float>& v);
}
