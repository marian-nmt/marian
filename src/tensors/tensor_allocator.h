#pragma once

#include <deque>
#include <set>

#include "common/definitions.h"
#include "tensors/allocator.h"
#include "tensors/tensor.h"

namespace marian {

class TensorAllocator {
private:
  const size_t CHUNK = 128;
  const size_t MBYTE = 1024 * 1024;
  const size_t GROW = CHUNK * MBYTE;
  const size_t ALIGN = 256;

  Ptr<Backend> backend_;
  Ptr<Allocator> allocator_;

public:
  TensorAllocator(Ptr<Backend> backend)
      : backend_(backend),
        allocator_(New<Allocator>(backend_->getDeviceId(), 0, GROW, ALIGN)) {}

  TensorAllocator(Ptr<Backend> backend, Ptr<Device> device)
      : backend_(backend),
        allocator_(New<Allocator>(backend_->getDeviceId(), device, 0, GROW, ALIGN)) {}

  ~TensorAllocator() { clear(); }

  void throwAtReallocation(bool throwRealloc) {
    allocator_->throwAtReallocation(throwRealloc);
  }

  void reserve(size_t bytes = 0) {
    auto mult = bytes / GROW + 1;
    LOG(info,
        "[memory] Extending reserved space to {} MB (device {})",
        mult * CHUNK,
        allocator_->getDeviceId());

    allocator_->reserve(mult * GROW);
  }

  void reserveExact(const std::vector<size_t>& bytes) {
    size_t total = 0;
    for(auto part : bytes)
      total += allocator_->alignedSize(part);
    reserveExact(total);
  }  

  void reserveExact(size_t bytes = 0) {
    size_t mbytes = bytes / MBYTE;
    if(mbytes == 0) {
      LOG(info,
          "[memory] Reserving {} B, device {}",
          bytes,
          allocator_->getDeviceId());
    } else {
      LOG(info,
          "[memory] Reserving {} MB, device {}",
          mbytes,
          allocator_->getDeviceId());
    }
    allocator_->reserve(bytes);
  }

  void clear() { allocator_->clear(); }

  size_t capacity(Shape shape, Type type = Type::float32) {
    return allocator_->capacity<char>(requiredBytes(shape, type));
  }

  void allocate(/*out*/ Tensor& t, Shape shape, Type type = Type::float32) {
    if(!t || t->shape() != shape) {
      auto mem = allocator_->alloc(requiredBytes(shape, type));
      t = Tensor(TensorBase::New(mem, shape, type, backend_));
    }
  }

  void free(const Tensor& t) { allocator_->free(t->memory()); }

  Tensor asTensor(Type type = Type::float32) {
    auto mem = allocator_->memory();
    auto size = mem->size() / sizeOf(type);
    return TensorBase::New(mem, Shape({1, (int)size}), type, backend_);
  }

  size_t size(Type type = Type::float32) { return allocator_->size() / sizeOf(type); }

  Ptr<Allocator> allocator() { return allocator_; }
};
}  // namespace marian
