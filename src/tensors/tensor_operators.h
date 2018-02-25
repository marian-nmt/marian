#pragma once

#include "common/definitions.h"
#include "tensors/tensor.h"
#include "tensors/allocator.h"

#include "tensors/dispatch.h"

#include "gpu/shape.h"
#include "gpu/tmp.h"
#include "gpu/tensor.h"

#include "tensors/gpu/element.h"
#include "tensors/gpu/add.h"
#include "tensors/gpu/prod.h"

#include "tensors/cpu/element.h"
#include "tensors/cpu/add.h"

namespace marian {

  template <class Functor, class ...Tensors>
  void Element(Functor functor, marian::Tensor out, Tensors ...tensors) {
    if(out->getBackend()->getDevice().type == DeviceType::gpu)
      gpu::Element(functor, out, tensors...);
    else
      cpu::Element(functor, out, tensors...);
  }

  template <class Functor, class ...Tensors>
  void Add(Functor functor,
           float scale,
           marian::Tensor out,
           Tensors... tensors) {
    if(out->getBackend()->getDevice().type == DeviceType::gpu)
      gpu::Add(functor, scale, out, tensors...);
    else
      cpu::Add(functor, scale, out, tensors...);
  }

  template <class Functor, class ...Tensors>
  void Add(Functor functor,
           marian::Tensor out,
           Tensors... tensors) {
    Add(functor, 1, out, tensors...);
  }

  template <class Functor, class ...Tensors>
  void Reduce(Functor functor,
              float scale,
              marian::Tensor out,
              Tensors... tensors) {
    out->set(0);
    Add(functor, scale, out, tensors...);
  }

  template <class Functor, class ...Tensors>
  void Reduce(Functor functor,
              marian::Tensor out,
              Tensors... tensors) {
    out->set(0);
    Add(functor, out, tensors...);
  }

  DISPATCH7(Prod, marian::Tensor, const marian::Tensor, const marian::Tensor, bool, bool, float, float)
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

  namespace gpu {
    void Deconcatenate(std::vector<marian::Tensor>& outputs, const marian::Tensor in, int ax);
  }

  namespace cpu {
    void Deconcatenate(std::vector<marian::Tensor>& outputs, const marian::Tensor in, int ax);
  }

  static inline void Deconcatenate(std::vector<marian::Tensor>& outputs, const marian::Tensor in, int ax) {
    if(in->getBackend()->getDevice().type == DeviceType::gpu) {
      gpu::Deconcatenate(outputs, in, ax);
    }
    else {
      cpu::Deconcatenate(outputs, in, ax);
    }
  }

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

  namespace gpu {
    void LSTMCellBackward(std::vector<marian::Tensor> outputs,
                          std::vector<marian::Tensor> inputs,
                          marian::Tensor adj);
  }

  namespace cpu {
    void LSTMCellBackward(std::vector<marian::Tensor> outputs,
                          std::vector<marian::Tensor> inputs,
                          marian::Tensor adj);
  }

  static inline void LSTMCellBackward(std::vector<marian::Tensor> outputs,
                                      std::vector<marian::Tensor> inputs,
                                      marian::Tensor adj) {
    if(adj->getBackend()->getDevice().type == DeviceType::gpu) {
      gpu::LSTMCellBackward(outputs, inputs, adj);
    }
    else {
      cpu::LSTMCellBackward(outputs, inputs, adj);
    }
  }

  namespace gpu {
    void LSTMOutputBackward(std::vector<marian::Tensor> outputs,
                            std::vector<marian::Tensor> inputs,
                            marian::Tensor adj);
  }

  namespace cpu {
    void LSTMOutputBackward(std::vector<marian::Tensor> outputs,
                            std::vector<marian::Tensor> inputs,
                            marian::Tensor adj);
  }

  static inline void LSTMOutputBackward(std::vector<marian::Tensor> outputs,
                                        std::vector<marian::Tensor> inputs,
                                        marian::Tensor adj) {
    if(adj->getBackend()->getDevice().type == DeviceType::gpu) {
      gpu::LSTMOutputBackward(outputs, inputs, adj);
    }
    else {
      cpu::LSTMOutputBackward(outputs, inputs, adj);
    }
  }

  DISPATCH3(GRUFastForward, marian::Tensor, std::vector<marian::Tensor>, bool)

  namespace gpu {
    void GRUFastBackward(std::vector<marian::Tensor> outputs,
                         std::vector<marian::Tensor> inputs,
                         marian::Tensor adj,
                         bool final);
  }

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
    if(adj->getBackend()->getDevice().type == DeviceType::gpu) {
      gpu::GRUFastBackward(outputs, inputs, adj, final);
    }
    else {
      cpu::GRUFastBackward(outputs, inputs, adj, final);
    }
  }

  DISPATCH4(Att, marian::Tensor, marian::Tensor, marian::Tensor, marian::Tensor)
  DISPATCH7(AttBack, marian::Tensor, marian::Tensor, marian::Tensor, marian::Tensor, marian::Tensor, marian::Tensor, marian::Tensor)

  namespace gpu {
    float L2Norm(marian::Tensor in);
  }

  namespace cpu {
    float L2Norm(marian::Tensor in);
  }

  static inline float L2Norm(marian::Tensor in) {
    if(in->getBackend()->getDevice().type == DeviceType::gpu) {
      return gpu::L2Norm(in);
    }
    else {
      return cpu::L2Norm(in);
    }
  }
  
  DISPATCH5(PoolingWithMaskingForward, marian::Tensor, marian::Tensor, marian::Tensor, int, bool)
  DISPATCH6(PoolingWithMaskingBackward, marian::Tensor, marian::Tensor, marian::Tensor, marian::Tensor, int, bool)
  
}
