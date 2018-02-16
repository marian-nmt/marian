/* All or part of this file was contributed by Intel under license:
 *   Copyright (C) 2017-2018 Intel Corporation
 *   SPDX-License-Identifier: MIT
 */

#pragma once

#include <thrust/functional.h>
#include "tensors/tensor.h"

namespace marian {

namespace cpu {

using namespace thrust::placeholders;

template <typename Functor, typename... Tensors>
static void gAdd_fallback(Functor functor, float scale, const Shape& full, Tensor out_, Tensors... in_) {
  int lengths[] { in_->shape().elements()... };
  int outLength = out_->shape().elements();

  bool same = true;
  for (int n : lengths) {
    if (n != outLength) {
      same = false;
      break;
    }
  }

  float* out = out_->data();
  if (same) {
    #pragma omp parallel for simd
    for (int index = 0; index < outLength; ++index) {
      out[index] += functor(in_->data()[index]...) * scale;
    }
  } else {
    const Shape& outShape = out_->shape();
    int I = outShape[0];
    int J = outShape[1];
    int K = outShape[2];
    int L = outShape[3];
    int II = full[0] / I;
    int JJ = full[1] / J;
    int KK = full[2] / K;
    int LL = full[3] / L;

    #pragma omp parallel for collapse(4)
    for (int l = 0; l < L; ++l)
    for (int k = 0; k < K; ++k)
    for (int i = 0; i < I; ++i)
    for (int j = 0; j < J; ++j) {
      float sum = 0.f;
      #pragma omp simd collapse(4) reduction(+:sum)
      for (int ll = 0; ll < LL; ++ll)
      for (int kk = 0; kk < KK; ++kk)
      for (int ii = 0; ii < II; ++ii)
      for (int jj = 0; jj < JJ; ++jj) {
        sum += functor(in_->data()[in_->shape().bindex(i + ii, j + jj, k + kk, l + ll)]...);
      }

      int index = i*outShape.stride(0) + j*outShape.stride(1) + k*outShape.stride(2) + l*outShape.stride(3);
      out[index] += sum * scale;
    }
  }
}

template <typename Functor, typename... Tensors>
static void gAdd(Functor functor, float scale, const Shape& full, Tensor out_, Tensors... in_) {
  gAdd_fallback(functor, scale, full, out_, in_...);
}

template <template <typename T> class BinaryOperator>
using ThrustBinaryOperator =
  thrust::detail::functional::actor<
    thrust::detail::functional::composite<
      thrust::detail::functional::binary_operator<BinaryOperator>,
      thrust::detail::functional::actor<thrust::detail::functional::argument<0u>>,
      thrust::detail::functional::actor<thrust::detail::functional::argument<1u>>,
      thrust::null_type,
      thrust::null_type,
      thrust::null_type,
      thrust::null_type,
      thrust::null_type,
      thrust::null_type,
      thrust::null_type,
      thrust::null_type
    >
  >;

template <>
void gAdd<ThrustBinaryOperator<thrust::multiplies>, Tensor, Tensor>(
    ThrustBinaryOperator<thrust::multiplies>, float scale, const Shape& full, Tensor out_, Tensor in1_, Tensor in2_) {
  void gAddBinaryMultiply(float scale, const Shape& full, Tensor out_, Tensor in1_, Tensor in2_);
  gAddBinaryMultiply(scale, full, out_, in1_, in2_);
}

typedef thrust::detail::functional::actor<thrust::detail::functional::argument<0u>> ThrustIdentityFunction;

template <>
void gAdd<ThrustIdentityFunction, Tensor>(
    ThrustIdentityFunction, float scale, const Shape& full, Tensor out_, Tensor in_) {
  void gAddIdentityFunction(float scale, const Shape& full, Tensor out_, Tensor in_);
  gAddIdentityFunction(scale, full, out_, in_);
}

/* n.b. We can avoid this repetition (due to the default argument) with
 *      injudicious template metaprogramming, but this is more readable, and
 *      avoids the temptation to require C++17 (or Boost.Preprocessor).
 */
template <class Functor>
static void Add(Functor functor, Tensor out, Tensor in, float scale = 1.f) {
  auto full = out->shape();
  for (int i = 0; i < in->shape().size(); ++i) {
    full.set(i, std::max(full[i], in->shape()[i]));
  }

  gAdd(functor, scale, full, out, in);
}

template <class Functor>
static void Add(Functor functor, Tensor out, Tensor in1, Tensor in2, float scale = 1.f) {
  auto full = out->shape();
  for (int i = 0; i < SHAPE_SIZE; ++i) {
    full.set(i, std::max(full[i], in1->shape()[i]));
    full.set(i, std::max(full[i], in2->shape()[i]));
  }

  gAdd(functor, scale, full, out, in1, in2);
}

template <class Functor>
static void Add(Functor functor, Tensor out, Tensor in1, Tensor in2, Tensor in3, float scale = 1.f) {
  auto full = out->shape();
  for (int i = 0; i < SHAPE_SIZE; ++i) {
    full.set(i, std::max(full[i], in1->shape()[i]));
    full.set(i, std::max(full[i], in2->shape()[i]));
    full.set(i, std::max(full[i], in3->shape()[i]));
  }

  gAdd(functor, scale, full, out, in1, in2, in3);
}

template <typename Functor, typename... Tensors>
static void Element(Functor functor, Tensor out_, Tensors... in_) {
  Shape shapes[] = { in_->shape()... };
  const Shape& outShape = out_->shape();
  bool broadcast = false;
  for (const Shape& shape : shapes) {
    if (shape != outShape) {
      broadcast = true;
      break;
    }
  }

  float* out = out_->data();
  int length = outShape.elements();
  if (broadcast) {
    int I = outShape[0];
    int J = outShape[1];
    int K = outShape[2];
    int L = outShape[3];

    #pragma omp parallel for simd collapse(4)
    for (int l = 0; l < L; ++l)
    for (int k = 0; k < K; ++k)
    for (int i = 0; i < I; ++i)
    for (int j = 0; j < J; ++j) {
      int index = i*outShape.stride(0) + j*outShape.stride(1) + k*outShape.stride(2) + l*outShape.stride(3);
      out[index] = functor(out[index], in_->data()[in_->shape().bindex(i, j, k, l)]...);
    }
  } else {
    #pragma omp parallel for simd
    for (int index = 0; index < length; ++index) {
      out[index] = functor(out[index], in_->data()[index]...);
    }
  }
}

template <typename Functor, typename... Tensors>
static void gPick(Functor functor, Tensor out_, Tensor picks, Tensors... in_) {
  float* out = out_->data();
  const float* pick = picks->data();

  Shape& outShape = out_->shape();
  int length = outShape.elements();
  for (int index = 0; index < length; ++index) {
    int dims[4];
    outShape.dims(index, dims);
    int row = dims[0], col = dims[1];
    float picked = col == (int)pick[row];
    out[index] = functor(out[index], in_->data()[in_->shape().bindex(dims)]..., picked);
  }
}

template <typename Functor>
static void Pick(Functor functor, Tensor out, Tensor picks) {
  gPick(functor, out, picks);
}

template <typename Functor>
static void Pick(Functor functor, Tensor out, Tensor in, Tensor picks) {
  gPick(functor, out, picks, in);
}

template <typename Functor>
static void PickReduce(Functor functor, Tensor out_, Tensor in_, Tensor picks) {
  out_->set(0);

  float* out = out_->data();
  const float* in = in_->data();
  const float* pick = picks->data();

  Shape& outShape = out_->shape();
  Shape& inShape = in_->shape();
  int length = inShape.elements();
  for (int index = 0; index < length; ++index) {
    int dims[4];
    inShape.dims(index, dims);
    int row = dims[0], col = dims[1];
    float picked = col == (int)pick[row];

    int outIndex = outShape.bindex(dims);
    out[outIndex] += functor(in[index], picked);
  }
}

float L2Norm(Tensor in);

void Softmax(Tensor out, Tensor in, Tensor mask = nullptr);
void LogSoftmax(Tensor out, Tensor in);

void SoftmaxGrad(Tensor grad, Tensor adj, Tensor val);
void LogSoftmaxGrad(Tensor grad, Tensor adj, Tensor val);

void CudnnSoftmax(Tensor out, Tensor in);
void CudnnSoftmaxGrad(Tensor grad, Tensor adj, Tensor val);

void CudnnLogSoftmax(Tensor out, Tensor in);
void CudnnLogSoftmaxGrad(Tensor grad, Tensor adj, Tensor val);

void CrossEntropyPick(Tensor out, Tensor in, Tensor pick);
void CrossEntropyPickBackward(Tensor out, Tensor adj, Tensor a, Tensor pick);

void Prod(Tensor C,
          const Tensor A,
          const Tensor B,
          bool transA,
          bool transB,
          float beta = 0);

void CopyRows(Tensor out, const Tensor in, const std::vector<size_t>& indices);

void PasteRows(Tensor out, const Tensor in, const std::vector<size_t>& indices);

void CopyCols(Tensor out, const Tensor in, const std::vector<size_t>& indices);

void PasteCols(Tensor out, const Tensor in, const std::vector<size_t>& indices);

void Transpose(Tensor out, const Tensor in);

void Concatenate(Tensor out, const std::vector<Tensor>& inputs, int ax);

void Deconcatenate(std::vector<Tensor>& outputs, const Tensor in, int ax);

void LSTMCellForward(Tensor out, std::vector<Tensor> inputs);
void LSTMOutputForward(Tensor out, std::vector<Tensor> inputs);
void LSTMCellBackward(std::vector<Tensor> outputs, std::vector<Tensor> inputs, Tensor adj);
void LSTMOutputBackward(std::vector<Tensor> outputs, std::vector<Tensor> inputs, Tensor adj);

void GRUFastForward(Tensor out, std::vector<Tensor> inputs, bool final = false);

void GRUFastBackward(std::vector<Tensor> outputs,
                     std::vector<Tensor> inputs,
                     Tensor adj,
                     bool final = false);

void Att(Tensor out, Tensor va, Tensor context, Tensor state, Tensor coverage);
void AttBack(Tensor gva,
             Tensor gContext,
             Tensor gState,
             Tensor gCoverage,
             Tensor va,
             Tensor context,
             Tensor state,
             Tensor coverage,
             Tensor adj);

void LayerNormalization(
    Tensor out, Tensor in, Tensor gamma, Tensor beta, float eps = 1e-9);
void LayerNormalizationGrad(Tensor gradX,
                            Tensor gradGamma,
                            Tensor gradBeta,
                            Tensor adj,
                            Tensor y,
                            Tensor x,
                            Tensor gamma,
                            Tensor beta);

void Shift(Tensor out, Tensor in, Shape shift, bool invert = false);

void SetSparse(float*,
               const std::vector<size_t>& indices,
               const std::vector<float>& values);

}

}
