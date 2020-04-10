#pragma once

#include "functional/functional.h"
#include "tensors/tensor_allocator.h"
#include "tensors/tensor_operators.h"
#include "training/gradient_dropping/sparse_tensor.h"

namespace marian {

class GradientDropBase {
protected:
  Tensor residual;
  Tensor velocity;
  Tensor tmp;

  float cut_off;
  int step;

  std::vector<Ptr<TensorAllocator>> allocators;

  Tensor newTensor(int size, Ptr<Backend> backend) {
    Tensor t;
    Ptr<TensorAllocator> allocator_ = New<TensorAllocator>(backend);
    allocator_->reserveExact(size * sizeof(float));
    allocator_->allocate(t, {1, size});
    allocators.push_back(allocator_);

    return t;
  }

  virtual float find_threshold(Tensor grads, float rate) = 0;

public:
  virtual void dropGraph(Tensor t,
                         SparseTensor destination,
                         float rate = 0.99,
                         float momentum = 0.0)
      = 0;
};

namespace gpu {
class GradientDropBase : public marian::GradientDropBase {
protected:
  float find_threshold(Tensor grads, float rate) override;

public:
  void dropGraph(Tensor t,
                 SparseTensor destination,
                 float rate = 0.99,
                 float momentum = 0.0) override;
};
}  // namespace gpu

typedef Ptr<GradientDropBase> GradientDrop;

static inline GradientDrop PrepareGradientDrop(DeviceId deviceId) {
#ifdef CUDA_FOUND
  if(deviceId.type == DeviceType::gpu)
    return GradientDrop(new gpu::GradientDropBase());
  else
    ABORT("Gradient Dropping for CPU is not yet supported");
#else
  if(deviceId.type == DeviceType::gpu)
    ABORT("CUDA support not compiled into marian");
  else
    ABORT("Gradient Dropping for CPU is not yet supported");
#endif
}

}  // namespace marian
