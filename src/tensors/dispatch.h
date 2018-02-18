#pragma once

#include "common/definitions.h"
#include "tensors/tensor.h"

#define DISPATCH1(Function, Arg1) \
  namespace gpu { \
    void Function(Arg1); \
  } \
  namespace cpu { \
    void Function(Arg1); \
  } \
  void Function(Arg1 arg1) { \
    if(arg1->getBackend()->getDevice().type == DeviceType::gpu) \
      gpu::Function(arg1); \
    else \
      cpu::Function(arg1); \
  }

#define DISPATCH2(Function, Arg1, Arg2) \
  namespace gpu { \
    void Function(Arg1, Arg2); \
  } \
  namespace cpu { \
    void Function(Arg1, Arg2); \
  } \
  static inline void Function(Arg1 arg1, Arg2 arg2) { \
    if(arg1->getBackend()->getDevice().type == DeviceType::gpu) \
      gpu::Function(arg1, arg2); \
    else \
      cpu::Function(arg1, arg2); \
  }

namespace marian {

  DISPATCH2(Dropout, Tensor, float)

}
