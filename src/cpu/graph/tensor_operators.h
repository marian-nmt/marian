/* All or part of this file was contributed by Intel under license:
 *   Copyright (C) 2017-2018 Intel Corporation
 *   SPDX-License-Identifier: MIT
 */

#pragma once

#include "graph/backend.h"
#include "kernels/tensor_operators_cpu.h"

#if CUDA_FOUND
#include "kernels/tensor_operators_gpu.h"
#endif

namespace marian {

template <class Functor>
static void Add(Functor functor, Tensor out, Tensor in, float scale = 1.f) {
  if (out->residency == DEVICE_CPU) {
    cpu::Add(functor, out, in, scale);
  }
  #if CUDA_FOUND
  else {
    gpu::Add(functor, out, in, scale);
  }
  #endif
}

template <class Functor>
static void Add(Functor functor, Tensor out, Tensor in1, Tensor in2, float scale = 1.f) {
  if (out->residency == DEVICE_CPU) {
    cpu::Add(functor, out, in1, in2, scale);
  }
  #if CUDA_FOUND
  else {
    gpu::Add(functor, out, in1, in2, scale);
  }
  #endif
}

template <class Functor>
static void Add(Functor functor, Tensor out, Tensor in1, Tensor in2, Tensor in3, float scale = 1.f) {
  if (out->residency == DEVICE_CPU) {
    cpu::Add(functor, out, in1, in2, in3, scale);
  }
  #if CUDA_FOUND
  else {
    gpu::Add(functor, out, in1, in2, in3, scale);
  }
  #endif
}

template <class Functor>
static void Reduce(Functor functor, Tensor out, Tensor in, float scale = 1.f) {
  out->set(0);
  Add(functor, out, in, scale);
}

template <class Functor>
static void Reduce(Functor functor, Tensor out, Tensor in1, Tensor in2, float scale = 1.f) {
  out->set(0);
  Add(functor, out, in1, in2, scale);
}

template <class Functor>
static void Reduce(Functor functor, Tensor out, Tensor in1, Tensor in2, Tensor in3, float scale = 1.f) {
  out->set(0);
  Add(functor, out, in1, in2, in3, scale);
}

template <typename Functor, typename... Tensors>
static void Element(Functor functor, Tensor out, Tensors... in) {
  if (out->residency == DEVICE_CPU) {
    cpu::Element(functor, out, in...);
  }
  #if CUDA_FOUND
  else {
    gpu::Element(functor, out, in...);
  }
  #endif
}

template <typename Functor>
static void Pick(Functor functor, Tensor out, Tensor picks) {
  if (out->residency == DEVICE_CPU) {
    cpu::Pick(functor, out, picks);
  }
  #if CUDA_FOUND
  else {
    gpu::Pick(functor, out,  picks);
  }
  #endif
}

template <typename Functor>
static void Pick(Functor functor, Tensor out, Tensor in, Tensor picks) {
  if (out->residency == DEVICE_CPU) {
    cpu::Pick(functor, out, in, picks);
  }
  #if CUDA_FOUND
  else {
    gpu::Pick(functor, out, in, picks);
  }
  #endif
}

template <typename Functor>
static void PickReduce(Functor functor, Tensor out, Tensor in, Tensor picks) {
  if (out->residency == DEVICE_CPU) {
    cpu::PickReduce(functor, out, in, picks);
  }
  #if CUDA_FOUND
  else {
    gpu::PickReduce(functor, out, in, picks);
  }
  #endif
}

static float L2Norm(Tensor in) {
  #if CUDA_FOUND
  if (in->residency == DEVICE_CPU) {
  #endif
    return cpu::L2Norm(in);
  #if CUDA_FOUND
  } else {
    return gpu::L2Norm(in);
  }
  #endif
}

static void Softmax(Tensor out, Tensor in, Tensor mask = nullptr) {
  if (out->residency == DEVICE_CPU) {
    return cpu::Softmax(out, in, mask);
  }
  #if CUDA_FOUND
  else {
    return gpu::Softmax(out, in, mask);
  }
  #endif
}

static void LogSoftmax(Tensor out, Tensor in) {
  if (out->residency == DEVICE_CPU) {
    return cpu::LogSoftmax(out, in);
  }
  #if CUDA_FOUND
  else {
    return gpu::LogSoftmax(out, in);
  }
  #endif
}

static void SoftmaxGrad(Tensor grad, Tensor adj, Tensor val) {
  if (grad->residency == DEVICE_CPU) {
    return cpu::SoftmaxGrad(grad, adj, val);
  }
  #if CUDA_FOUND
  else {
    return gpu::SoftmaxGrad(grad, adj, val);
  }
  #endif
}

static void LogSoftmaxGrad(Tensor grad, Tensor adj, Tensor val) {
  if (grad->residency == DEVICE_CPU) {
    return cpu::LogSoftmaxGrad(grad, adj, val);
  }
  #if CUDA_FOUND
  else {
    return gpu::LogSoftmaxGrad(grad, adj, val);
  }
  #endif
}

static void CrossEntropyPick(Tensor out, Tensor in, Tensor pick) {
  if (out->residency == DEVICE_CPU) {
    return cpu::CrossEntropyPick(out, in, pick);
  }
  #if CUDA_FOUND
  else {
    return gpu::CrossEntropyPick(out, in, pick);
  }
  #endif
}

static void CrossEntropyPickBackward(Tensor out, Tensor adj, Tensor a, Tensor pick) {
  if (out->residency == DEVICE_CPU) {
    return cpu::CrossEntropyPickBackward(out, adj, a, pick);
  }
  #if CUDA_FOUND
  else {
    return gpu::CrossEntropyPickBackward(out, adj, a, pick);
  }
  #endif
}

#if BLAS_FOUND
static void Prod(Tensor C, const Tensor A, const Tensor B, bool transA, bool transB,
    float beta = 0.f) {
  cpu::Prod(C, A, B, transA, transB, beta);
}
#endif

#if CUDA_FOUND
static void Prod(cublasHandle_t handle, Tensor C, const Tensor A, const Tensor B,
    bool transA, bool transB, float beta = 0.f) {
  gpu::Prod(handle, C, A, B, transA, transB, beta);
}
#endif

#if BLAS_FOUND || CUDA_FOUND
static void Prod(Ptr<Backend> backend, Tensor C, const Tensor A, const Tensor B,
    bool transA, bool transB, float beta = 0.f) {
  if (backend->residency == DEVICE_CPU) {
    Prod(C, A, B, transA, transB, beta);
  }
  #if CUDA_FOUND
  else {
    Prod(std::static_pointer_cast<BackendGPU>(backend)->getCublasHandle(),
        C, A, B, transA, transB, beta);
  }
  #endif
}
#endif

static void CopyRows(Tensor out, const Tensor in, const std::vector<size_t>& indices) {
  if (out->residency == DEVICE_CPU) {
    return cpu::CopyRows(out, in, indices);
  }
  #if CUDA_FOUND
  else {
    return gpu::CopyRows(out, in, indices);
  }
  #endif
}

static void PasteRows(Tensor out, const Tensor in, const std::vector<size_t>& indices) {
  if (out->residency == DEVICE_CPU) {
    return cpu::PasteRows(out, in, indices);
  }
  #if CUDA_FOUND
  else {
    return gpu::PasteRows(out, in, indices);
  }
  #endif
}

static void CopyCols(Tensor out, const Tensor in, const std::vector<size_t>& indices) {
  if (out->residency == DEVICE_CPU) {
    return cpu::CopyCols(out, in, indices);
  }
  #if CUDA_FOUND
  else {
    return gpu::CopyCols(out, in, indices);
  }
  #endif
}

static void PasteCols(Tensor out, const Tensor in, const std::vector<size_t>& indices) {
  if (out->residency == DEVICE_CPU) {
    return cpu::PasteCols(out, in, indices);
  }
  #if CUDA_FOUND
  else {
    return gpu::PasteCols(out, in, indices);
  }
  #endif
}

#if MKL_FOUND
static void Transpose(Tensor out, const Tensor in) {
  cpu::Transpose(out, in);
}
#endif

#if CUDA_FOUND
static void Transpose(cublasHandle_t handle, Tensor out, const Tensor in) {
  gpu::Transpose(handle, out, in);
}
#endif

#if MKL_FOUND || CUDA_FOUND
static void Transpose(Ptr<Backend> backend, Tensor out, const Tensor in) {
  #if MKL_FOUND
  if (backend->residency == DEVICE_CPU) {
    Transpose(out, in);
  }
  #endif

  #if CUDA_FOUND
  Transpose(std::static_pointer_cast<BackendGPU>(backend)->getCublasHandle(),
      out, in);
  #endif
}
#endif

static void Concatenate(Tensor out, const std::vector<Tensor>& inputs, int ax) {
  if (out->residency == DEVICE_CPU) {
    return cpu::Concatenate(out, inputs, ax);
  }
  #if CUDA_FOUND
  else {
    return gpu::Concatenate(out, inputs, ax);
  }
  #endif
}

static void Deconcatenate(std::vector<Tensor>& outputs, const Tensor in, int ax) {
  if (outputs[0]->residency == DEVICE_CPU) {
    return cpu::Deconcatenate(outputs, in, ax);
  }
  #if CUDA_FOUND
  else {
    return gpu::Deconcatenate(outputs, in, ax);
  }
  #endif
}

static void LSTMCellForward(Tensor out, std::vector<Tensor> inputs) {
  if (out->residency == DEVICE_CPU) {
    return cpu::LSTMCellForward(out, inputs);
  }
  #if CUDA_FOUND
  else {
    return gpu::LSTMCellForward(out, inputs);
  }
  #endif
}

static void LSTMOutputForward(Tensor out, std::vector<Tensor> inputs) {
  if (out->residency == DEVICE_CPU) {
    return cpu::LSTMOutputForward(out, inputs);
  }
  #if CUDA_FOUND
  else {
    return gpu::LSTMOutputForward(out, inputs);
  }
  #endif
}

static void LSTMCellBackward(std::vector<Tensor> outputs, std::vector<Tensor> inputs, Tensor adj) {
  // Any of the outputs may be nullptr, so check input instead, even if inconsistent with others in this file
  if (inputs[0]->residency == DEVICE_CPU) {
    return cpu::LSTMCellBackward(outputs, inputs, adj);
  }
  #if CUDA_FOUND
  else {
    return gpu::LSTMCellBackward(outputs, inputs, adj);
  }
  #endif
}

static void LSTMOutputBackward(std::vector<Tensor> outputs, std::vector<Tensor> inputs, Tensor adj) {
  // Any of the outputs may be nullptr, so check input instead, even if inconsistent with others in this file
  if (inputs[0]->residency == DEVICE_CPU) {
    return cpu::LSTMOutputBackward(outputs, inputs, adj);
  }
  #if CUDA_FOUND
  else {
    return gpu::LSTMOutputBackward(outputs, inputs, adj);
  }
  #endif
}

static void GRUFastForward(Tensor out, std::vector<Tensor> inputs, bool final = false) {
  if (out->residency == DEVICE_CPU) {
    return cpu::GRUFastForward(out, inputs, final);
  }
  #if CUDA_FOUND
  else {
    return gpu::GRUFastForward(out, inputs, final);
  }
  #endif
}

static void GRUFastBackward(std::vector<Tensor> outputs, std::vector<Tensor> inputs,
    Tensor adj, bool final = false) {
  if (inputs[0]->residency == DEVICE_CPU) {
    return cpu::GRUFastBackward(outputs, inputs, adj, final);
  }
  #if CUDA_FOUND
  else {
    return gpu::GRUFastBackward(outputs, inputs, adj, final);
  }
  #endif
}

static void Att(Tensor out, Tensor va, Tensor context, Tensor state, Tensor coverage) {
  if (out->residency == DEVICE_CPU) {
    return cpu::Att(out, va, context, state, coverage);
  }
  #if CUDA_FOUND
  else {
    return gpu::Att(out, va, context, state, coverage);
  }
  #endif
}

static void AttBack(Tensor gva, Tensor gContext, Tensor gState, Tensor gCoverage, Tensor va,
    Tensor context, Tensor state, Tensor coverage, Tensor adj) {
  if (gva->residency == DEVICE_CPU) {
    return cpu::AttBack(gva, gContext, gState, gCoverage, va, context, state, coverage, adj);
  }
  #if CUDA_FOUND
  else {
    return gpu::AttBack(gva, gContext, gState, gCoverage, va, context, state, coverage, adj);
  }
  #endif
}

static void LayerNormalization(Tensor out, Tensor in, Tensor gamma, Tensor beta, float eps = 1e-9) {
  if (out->residency == DEVICE_CPU) {
    return cpu::LayerNormalization(out, in, gamma, beta, eps);
  }
  #if CUDA_FOUND
  else {
    return gpu::LayerNormalization(out, in, gamma, beta, eps);
  }
  #endif
}

static void LayerNormalizationGrad(Tensor gradX, Tensor gradGamma, Tensor gradBeta, Tensor adj,
    Tensor y, Tensor x, Tensor gamma, Tensor beta) {
  if (gradX->residency == DEVICE_CPU) {
    return cpu::LayerNormalizationGrad(gradX, gradGamma, gradBeta, adj, y, x, gamma, beta);
  }
  #if CUDA_FOUND
  else {
    return gpu::LayerNormalizationGrad(gradX, gradGamma, gradBeta, adj, y, x, gamma, beta);
  }
  #endif
}

static void Shift(Tensor out, Tensor in, Shape shift, bool invert = false) {
  if (out->residency == DEVICE_CPU) {
    return cpu::Shift(out, in, shift, invert);
  }
  #if CUDA_FOUND
  else {
    return gpu::Shift(out, in, shift, invert);
  }
  #endif
}

static void SetSparse(Tensor out, const std::vector<size_t>& indices, const std::vector<float>& values) {
  if (out->residency == DEVICE_CPU) {
    return cpu::SetSparse(out->data(), indices, values);
  }
  #if CUDA_FOUND
  else {
    return gpu::SetSparse(out->data(), indices, values);
  }
  #endif
}

}
