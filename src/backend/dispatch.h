#pragma once

#include "common/definitions.h"
#include "graph/backend.h"
#include "tensors/tensor.h"

#define DISPATCH1(Function, Arg1) \
  namespace gpu { \
    void Function(Ptr<Backend>, Arg1); \
  } \
  namespace cpu { \
    void Function(Ptr<Backend>, Arg1); \
  } \
  void Function(Ptr<Backend> backend, Arg1 arg1) { \
    if(backend->getDevice().type == DeviceType::gpu) \
      gpu::Function(backend, arg1); \
    else \
      cpu::Function(backend, arg1); \
  }

#define DISPATCH2(Function, Arg1, Arg2) \
  namespace gpu { \
    void Function(Ptr<Backend>, Arg1, Arg2); \
  } \
  namespace cpu { \
    void Function(Ptr<Backend>, Arg1, Arg2); \
  } \
  static inline void Function(Ptr<Backend> backend, Arg1 arg1, Arg2 arg2) { \
    if(backend->getDevice().type == DeviceType::gpu) \
      gpu::Function(backend, arg1, arg2); \
    else \
      cpu::Function(backend, arg1, arg2); \
  }
  
namespace marian {

  DISPATCH2(Dropout, Tensor, float)
    
}