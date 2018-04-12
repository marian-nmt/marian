#pragma once

#include "common/definitions.h"
#include "tensors/allocator.h"
#include "tensors/tensor.h"

#include "tensors/dispatch.h"

#include "functional/shape.h"
#include "functional/tensor.h"
#include "functional/tmp.h"

#ifdef CUDA_FOUND
#include "tensors/gpu/add.h"
#include "tensors/gpu/element.h"
#include "tensors/gpu/prod.h"
#endif

#include "tensors/cpu/add.h"
#include "tensors/cpu/element.h"

namespace marian {

template <class Functor, class... Tensors>
void Element(Functor functor, marian::Tensor out, Tensors... tensors) {
#ifdef CUDA_FOUND
  if(out->getBackend()->getDevice().type == DeviceType::gpu)
    gpu::Element(functor, out, tensors...);
  else
#endif
    cpu::Element(functor, out, tensors...);
}

template <class Functor, class... Tensors>
void Add(Functor functor, float scale, marian::Tensor out, Tensors... tensors) {
#ifdef CUDA_FOUND
  if(out->getBackend()->getDevice().type == DeviceType::gpu)
    gpu::Add(functor, scale, out, tensors...);
  else
#endif
    cpu::Add(functor, scale, out, tensors...);
}

template <class Functor, class... Tensors>
void Add(Functor functor, marian::Tensor out, Tensors... tensors) {
  Add(functor, 1, out, tensors...);
}

template <class Functor, class... Tensors>
void Reduce(Functor functor,
            float scale,
            marian::Tensor out,
            Tensors... tensors) {
  out->set(0.f);
  Add(functor, scale, out, tensors...);
}

template <class Functor, class... Tensors>
void Reduce(Functor functor, marian::Tensor out, Tensors... tensors) {
  out->set(0.f);
  Add(functor, out, tensors...);
}

// clang-format off
  DISPATCH7(Prod, marian::Tensor, const marian::Tensor, const marian::Tensor, bool, bool, float, float)
  DISPATCH8(ProdWithBias, marian::Tensor, const marian::Tensor, const marian::Tensor, const marian::Tensor, bool, bool, float, float)
  DISPATCH7(ProdBatched, marian::Tensor, const marian::Tensor, const marian::Tensor, bool, bool, float, float)

  DISPATCH2(Dropout, marian::Tensor, float)

  DISPATCH3(Softmax, marian::Tensor, marian::Tensor, marian::Tensor)
  DISPATCH3(SoftmaxGrad, marian::Tensor, marian::Tensor, marian::Tensor)

  DISPATCH2(LogSoftmax, marian::Tensor, marian::Tensor)
  DISPATCH3(LogSoftmaxGrad, marian::Tensor, marian::Tensor, marian::Tensor)

  DISPATCH3(CrossEntropyPick, marian::Tensor, marian::Tensor, marian::Tensor)
  DISPATCH4(CrossEntropyPickBackward, marian::Tensor, marian::Tensor, marian::Tensor, marian::Tensor)

  DISPATCH3(TransposeND, marian::Tensor, marian::Tensor, const std::vector<int>&)
  DISPATCH4(Shift, marian::Tensor, marian::Tensor, marian::Shape, bool)

  DISPATCH3(Concatenate, marian::Tensor, const std::vector<marian::Tensor>&, int)
// clang-format on

#ifdef CUDA_FOUND
namespace gpu {
void Deconcatenate(std::vector<marian::Tensor>& outputs,
                   const marian::Tensor in,
                   int ax);
}
#endif

namespace cpu {
void Deconcatenate(std::vector<marian::Tensor>& outputs,
                   const marian::Tensor in,
                   int ax);
}

static inline void Deconcatenate(std::vector<marian::Tensor>& outputs,
                                 const marian::Tensor in,
                                 int ax) {
#ifdef CUDA_FOUND
  if(in->getBackend()->getDevice().type == DeviceType::gpu)
    gpu::Deconcatenate(outputs, in, ax);
  else
#endif
    cpu::Deconcatenate(outputs, in, ax);
}

// clang-format off
  DISPATCH5(LayerNormalization, marian::Tensor, marian::Tensor, marian::Tensor, marian::Tensor, float)
  DISPATCH9(LayerNormalizationGrad, marian::Tensor, marian::Tensor, marian::Tensor, marian::Tensor, marian::Tensor, marian::Tensor, marian::Tensor, marian::Tensor, float)

  DISPATCH4(HighwayForward, marian::Tensor, const marian::Tensor, const marian::Tensor, const marian::Tensor)
  DISPATCH7(HighwayBackward, marian::Tensor, marian::Tensor, marian::Tensor, const marian::Tensor, const marian::Tensor, const marian::Tensor, const marian::Tensor)

  DISPATCH3(CopyRows, marian::Tensor, const marian::Tensor, const std::vector<size_t>&)
  DISPATCH3(PasteRows, marian::Tensor, const marian::Tensor, const std::vector<size_t>&)
  DISPATCH3(CopyCols, marian::Tensor, const marian::Tensor, const std::vector<size_t>&)
  DISPATCH3(PasteCols, marian::Tensor, const marian::Tensor, const std::vector<size_t>&)

  DISPATCH5(Select, marian::Tensor, marian::Tensor, int, const std::vector<size_t>&, Ptr<Allocator>)
  DISPATCH5(Insert, marian::Tensor, marian::Tensor, int, const std::vector<size_t>&, Ptr<Allocator>)

  DISPATCH2(LSTMCellForward, marian::Tensor, std::vector<marian::Tensor>)
  DISPATCH2(LSTMOutputForward, marian::Tensor, std::vector<marian::Tensor>);
// clang-format on

#ifdef CUDA_FOUND
namespace gpu {
void LSTMCellBackward(std::vector<marian::Tensor> outputs,
                      std::vector<marian::Tensor> inputs,
                      marian::Tensor adj);
}
#endif

namespace cpu {
void LSTMCellBackward(std::vector<marian::Tensor> outputs,
                      std::vector<marian::Tensor> inputs,
                      marian::Tensor adj);
}

static inline void LSTMCellBackward(std::vector<marian::Tensor> outputs,
                                    std::vector<marian::Tensor> inputs,
                                    marian::Tensor adj) {
#ifdef CUDA_FOUND
  if(adj->getBackend()->getDevice().type == DeviceType::gpu)
    gpu::LSTMCellBackward(outputs, inputs, adj);
  else
#endif
    cpu::LSTMCellBackward(outputs, inputs, adj);
}

#ifdef CUDA_FOUND
namespace gpu {
void LSTMOutputBackward(std::vector<marian::Tensor> outputs,
                        std::vector<marian::Tensor> inputs,
                        marian::Tensor adj);
}
#endif

namespace cpu {
void LSTMOutputBackward(std::vector<marian::Tensor> outputs,
                        std::vector<marian::Tensor> inputs,
                        marian::Tensor adj);
}

static inline void LSTMOutputBackward(std::vector<marian::Tensor> outputs,
                                      std::vector<marian::Tensor> inputs,
                                      marian::Tensor adj) {
#ifdef CUDA_FOUND
  if(adj->getBackend()->getDevice().type == DeviceType::gpu)
    gpu::LSTMOutputBackward(outputs, inputs, adj);
  else
#endif
    cpu::LSTMOutputBackward(outputs, inputs, adj);
}

DISPATCH3(GRUFastForward, marian::Tensor, std::vector<marian::Tensor>, bool)

#ifdef CUDA_FOUND
namespace gpu {
void GRUFastBackward(std::vector<marian::Tensor> outputs,
                     std::vector<marian::Tensor> inputs,
                     marian::Tensor adj,
                     bool final);
}
#endif

namespace cpu {
void GRUFastBackward(std::vector<marian::Tensor> outputs,
                     std::vector<marian::Tensor> inputs,
                     marian::Tensor adj,
                     bool final);
}

static inline void GRUFastBackward(std::vector<marian::Tensor> outputs,
                                   std::vector<marian::Tensor> inputs,
                                   marian::Tensor adj,
                                   bool final = false) {
#ifdef CUDA_FOUND
  if(adj->getBackend()->getDevice().type == DeviceType::gpu)
    gpu::GRUFastBackward(outputs, inputs, adj, final);
  else
#endif
    cpu::GRUFastBackward(outputs, inputs, adj, final);
}

// clang-format off
  DISPATCH4(Att, marian::Tensor, marian::Tensor, marian::Tensor, marian::Tensor)
  DISPATCH7(AttBack, marian::Tensor, marian::Tensor, marian::Tensor, marian::Tensor, marian::Tensor, marian::Tensor, marian::Tensor)
// clang-format on

#ifdef CUDA_FOUND
namespace gpu {
float L2Norm(marian::Tensor in);
}
#endif

namespace cpu {
float L2Norm(marian::Tensor in);
}

static inline float L2Norm(marian::Tensor in) {
#ifdef CUDA_FOUND
  if(in->getBackend()->getDevice().type == DeviceType::gpu)
    return gpu::L2Norm(in);
  else
#endif
    return cpu::L2Norm(in);
}

// clang-format off
  DISPATCH5(PoolingWithMaskingForward, marian::Tensor, marian::Tensor, marian::Tensor, int, bool)
  DISPATCH6(PoolingWithMaskingBackward, marian::Tensor, marian::Tensor, marian::Tensor, marian::Tensor, int, bool)
// clang-format on
}
