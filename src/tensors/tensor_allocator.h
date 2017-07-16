#pragma once

#include <deque>
#include <set>

#include "common/definitions.h"
#include "tensors/tensor.h"
#include "tensors/allocator.h"
#include "tensors/device_gpu.h"

namespace marian {

class TensorAllocator {
private:
  const size_t CHUNK = 512;
  const size_t MBYTE = 1024 * 1024;
  const size_t GROW = CHUNK * MBYTE;
  const size_t ALIGN = 256;

  Ptr<Allocator<DeviceGPU>> allocator_;


public:
  TensorAllocator(size_t device)
    : allocator_(New<Allocator<DeviceGPU>>(device, 0, GROW, ALIGN))
  {}

  ~TensorAllocator() { clear(); }

  void throwAtReallocation(bool throwRealloc) {
    allocator_->throwAtReallocation(throwRealloc);
  }

  void reserve(size_t bytes = 0) {
    float mult = bytes / GROW + 1;
    LOG(memory)->info(
        "Extending reserved space to {} MB (device {})",
        mult * CHUNK,
        allocator_->getDevice());

    allocator_->reserve(mult * GROW);
  }

  void reserveExact(size_t bytes = 0) {
    size_t mbytes = bytes / MBYTE;
    LOG(memory)->info(
        "Reserving {} MB, device {}",
        mbytes,
        allocator_->getDevice());

    allocator_->reserve(bytes);
  }

  void clear() {
    allocator_->clear();
  }

  size_t capacity(Shape shape) {
    return allocator_->capacity<float>(shape.elements());
  }

  void allocate(Tensor& t, Shape shape) {
    if(!t || t->shape() != shape) {
      int size = shape.elements();
      auto mem = allocator_->alloc<float>(size);
      t = Tensor(new TensorBase(mem, shape, allocator_->getDevice()));
    }
  }

  void free(Tensor& t) {
    allocator_->free(t->memory());
  }

  Tensor asTensor() {
    auto mem = allocator_->memory();
    int size = mem->size() / sizeof(float);
    return Tensor(new TensorBase(mem, {1, size}, allocator_->getDevice()));
  }

  size_t size() { return allocator_->size() / sizeof(float); }

};

}
