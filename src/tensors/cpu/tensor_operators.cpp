/* All or part of this file was contributed by Intel under license:
 *   Copyright (C) 2017-2018 Intel Corporation
 *   SPDX-License-Identifier: MIT
 */

#include "tensors/tensor_operators.h"
#include "tensors/cpu/backend.h"

#include "gpu/tensor.h"
#include "functional/functional.h"

namespace marian {

namespace cpu {

void ConcatCont(marian::Tensor out, const std::vector<marian::Tensor>& inputs, int axis) {
  int step = 1;
  for(int i = 0; i < axis; ++i)
    step *= out->shape()[i];

  size_t offset1 = 0;
  for(int i = 0; i < step; ++i) {
    for(auto in : inputs) {
      size_t size = in->shape().elements() / step;
      size_t offset2 = i * size;

      std::copy(in->data() + offset2,
                in->data() + offset2 + size,
                out->data() + offset1);

      offset1 += size;
    }
  }
}

inline void gInsertCols(float* out,
                 const float* in,
                 size_t rows,
                 size_t cols,
                 size_t cols_out,
                 size_t cols_in,
                 size_t offset_out,
                 size_t offset_in) {
  for(int j = 0; j < rows; ++j) {
    float* rowOut = out + j * cols_out + offset_out;
    const float* rowIn = in + j * cols_in + offset_in;
    for(int i = 0; i < cols; ++i) {
      rowOut[i] = rowIn[i];
    }
  }
}

void Concatenate1(marian::Tensor out, const std::vector<marian::Tensor>& inputs) {
  int rows = out->shape().elements() / out->shape().back();

  size_t offset = 0;
  int cols_out = out->shape().back();

  for(auto in : inputs) {
    ABORT_IF(rows != in->shape().elements() / in->shape().back(),
                   "First dimension must be equal");
    int cols_in = in->shape().back();
    cpu::gInsertCols(out->data(), in->data(), rows, cols_in, cols_out, cols_in, offset, 0);
    offset += cols_in;
  }
}

void Concatenate(marian::Tensor out, const std::vector<marian::Tensor>& inputs, int ax) {
  if(ax == out->shape().size() - 1)
    Concatenate1(out, inputs);
  else
    ConcatCont(out, inputs, ax);
}

void Deconcatenate(std::vector<marian::Tensor>& outputs, const marian::Tensor in, int ax) {
  ABORT("Not implemented!");
}

// @TODO: optimize this, currently it's quite horrible
void TransposeND(marian::Tensor out, marian::Tensor in, const std::vector<int>& vAxis) {
  gpu::Array<int, gpu::Shape::size()> permute;
  int diff = gpu::Shape::size() - vAxis.size();
  for(int i = 0; i < permute.size(); ++i)
    if(i < diff)
      permute[i] = i;
    else
      permute[i] = vAxis[i - diff] + diff;

  int length = out->shape().elements();

  constexpr size_t N = gpu::Shape::size();
  gpu::Array<int, N> oDims;
  gpu::Array<int, N> pDims;
  gpu::Tensor<float> gOut = out;
  gpu::Tensor<float> gIn = in;

  for(int index = 0; index < length; ++index) {
    gOut.shape().dims(index, oDims);
    for(int i = 0; i < N; ++i)
      pDims[permute[i]] = oDims[i];
    gOut[index] = gIn[pDims];
  }
}

void Softmax(Tensor out_, Tensor in_, Tensor mask_) {
  float* out = out_->data();
  const float* in = in_->data();
  const float* mask = mask_ ? mask_->data() : nullptr;

  int rows = out_->shape().elements() / out_->shape().back();
  int cols = out_->shape().back();

  for (int j = 0; j < rows; ++j) {
    float* so = out + j*cols;
    const float* sp = in + j*cols;
    const float* mp = mask ? mask + j*cols : nullptr;

    float max = sp[0];
    for (int i = 1; i < cols; ++i) {
      max = std::max(max, sp[i]);
    }

    float sum = 0.f;
    for (int i = 0; i < cols; ++i) {
      float ex = !mask || mp[i] ? std::exp(sp[i] - max) : 0.f;
      so[i] = ex;
      sum += ex;
    }

    for (int i = 0; i < cols; ++i) {
      so[i] /= sum;
    }
  }
}

void LogSoftmax(Tensor out_, Tensor in_) {
  float* out = out_->data();
  const float* in = in_->data();

  int rows = out_->shape().elements() / out_->shape().back();
  int cols = out_->shape().back();

  for (int j = 0; j < rows; ++j) {
    float* so = out + j * cols;
    const float* sp = in + j*cols;

    float max = sp[0];
    for (int i = 1; i < cols; ++i) {
      max = std::max(max, sp[i]);
    }

    float sum = 0.f;
    for (int i = 0; i < cols; ++i) {
      float sm = sp[i] - max;
      float ex = std::exp(sm);
      so[i] = sm;
      sum += ex;
    }

    for (int i = 0; i < cols; ++i) {
      so[i] -= std::log(sum);
    }
  }
}

void SoftmaxGrad(marian::Tensor grad, marian::Tensor adj, marian::Tensor val) {
  ABORT("Not implemented!");
}

void LogSoftmaxGrad(marian::Tensor grad, marian::Tensor adj, marian::Tensor val) {
  ABORT("Not implemented!");
}

void CopyRows(marian::Tensor out_, const marian::Tensor in_, const std::vector<size_t>& indices) {
  size_t cols = in_->shape()[1];
  size_t rows = indices.size();

  float* out = out_->data();
  const float* in = in_->data();

  #pragma omp parallel for
  for (int j = 0; j < rows; ++j) {
    size_t dst = j;
    size_t src = indices[j];

    float* rowOut = out + dst*cols;
    const float* rowIn = in + src*cols;

    std::copy(rowIn, rowIn + cols, rowOut);
  }
}

void PasteRows(marian::Tensor out,
               const marian::Tensor in,
               const std::vector<size_t>& indices) {
  ABORT("Not implemented!");
}

void CopyCols(marian::Tensor out, const marian::Tensor in, const std::vector<size_t>& indices) {
  ABORT("Not implemented!");
}

void PasteCols(marian::Tensor out,
               const marian::Tensor in,
               const std::vector<size_t>& indices) {
  ABORT("Not implemented!");
}

void Select(marian::Tensor out,
            const marian::Tensor in,
            int axis,
            const std::vector<size_t>& indices,
            Ptr<Allocator> allocator) {
  ABORT("Not implemented!");
}

void Insert(marian::Tensor out,
            const marian::Tensor in,
            int axis,
            const std::vector<size_t>& indices,
            Ptr<Allocator> allocator) {
  ABORT("Not implemented!");
}

void GRUFastForward(marian::Tensor out_, std::vector<marian::Tensor> inputs, bool final) {
  int rows = out_->shape().elements() / out_->shape().back();
  int cols = out_->shape().back();

  float* out = out_->data();

  const float* state = inputs[0]->data();
  const float* xW = inputs[1]->data();
  const float* sU = inputs[2]->data();
  const float* b = inputs[3]->data();
  const float* mask = inputs.size() > 4 ? inputs[4]->data() : nullptr;

  #pragma omp parallel for
  for (int j = 0; j < rows; ++j) {
    float m = !mask || mask[j];
    float* rowOut = out + j * cols;
    const float* rowState = state + j * cols;

    const float* xWrow = xW + j * cols * 3;
    const float* sUrow = sU + j * cols * 3;

    #pragma omp simd
    for (int i = 0; i < cols; ++i) {
      // @TODO: stable logit
      float ev1 = std::exp(-(xWrow[i] + sUrow[i] + b[i]));
      float r = 1.0f / (1.0f + ev1);

      int k = i + cols;
      // @TODO: stable logit
      float ev2 = std::exp(-(xWrow[k] + sUrow[k] + b[k]));
      float z = 1.0f / (1.0f + ev2);

      int l = i + 2 * cols;
      float h;
      if(final)
        h = std::tanh(xWrow[l] + (sUrow[l] + b[l]) * r);
      else
        h = std::tanh(xWrow[l] + sUrow[l] * r + b[l]);

      float out = (1.0f - z) * h + z * rowState[i];
      rowOut[i] = m * out + (1 - m) * rowState[i];
    }
  }
}

void GRUFastBackward(std::vector<marian::Tensor> outputs,
                     std::vector<marian::Tensor> inputs,
                     marian::Tensor adj,
                     bool final) {
  ABORT("Not implemented!");
}

void CrossEntropyPick(marian::Tensor out_, marian::Tensor in_, marian::Tensor pick_) {
  float* out = out_->data();
  Shape& outShape = out_->shape();
  const float* in = in_->data();
  Shape& inShape = in_->shape();
  float* pick = pick_->data();

  int rows = inShape.elements() / inShape.back();
  int cols = inShape.back();

  #pragma omp parallel for
  for (int j = 0; j < rows; ++j) {
    const float* sp = in + j*cols;
    float max = sp[0];
    #pragma omp simd reduction(max:max)
    for (int i = 1; i < cols; ++i) {
      max = std::max(max, sp[i]);
    }

    float sum = 0.f;
    #pragma omp simd reduction(+:sum)
    for (int i = 0; i < cols; ++i) {
      sum += std::exp(sp[i] - max);
    }

    // cross-entropy
    int i = pick[j];
    // This appears to be safe i.e. that i >= 0 && i < cols is known
    out[j] = std::log(sum) - sp[i] + max;
  }
}

void CrossEntropyPickBackward(marian::Tensor out, marian::Tensor adj, marian::Tensor a, marian::Tensor pick) {
  ABORT("Not implemented!");
}


float L2Norm(marian::Tensor in) {
  ABORT("Not implemented!");
}

void Att(marian::Tensor out_, marian::Tensor va_, marian::Tensor context_, marian::Tensor state_) {
  float* out = out_->data();
  const float* va = va_->data();
  const float* ctx = context_->data();
  const float* state = state_->data();

  int m = out_->shape().elements() / out_->shape().back();
  int k = context_->shape()[-1];
  int b = context_->shape()[-2];
  int t = context_->shape()[-3];

  int rows = m;
  int cols = k;

  #pragma omp parallel for
  for (size_t j = 0; j < rows; ++j) {
    const float* vaRow = va;
    const float* ctxRow = ctx + (j % (b * t)) * cols;
    const float* stateRow = state + ((j / (b * t)) * b + j % b) * cols;

    float sum = 0.f;
    #pragma omp simd reduction(+:sum)
    for (size_t i = 0; i < cols; ++i) {
      float z = ctxRow[i] + stateRow[i];
      sum += std::tanh(z) * vaRow[i];
    }

    out[j] = sum;
  }
}

void AttBack(marian::Tensor gVa,
             marian::Tensor gContext,
             marian::Tensor gState,
             marian::Tensor va,
             marian::Tensor context,
             marian::Tensor state,
             marian::Tensor adj) {
  ABORT("Not implemented!");
}

void LayerNormalization(marian::Tensor out_,
                        marian::Tensor in_,
                        marian::Tensor gamma_,
                        marian::Tensor beta_,
                        float eps) {
  float* out = out_->data();
  const float* in = in_->data();
  const float* alpha = gamma_->data();
  const float* beta = beta_ ? beta_->data() : nullptr;

  int rows = in_->shape().elements() / in_->shape().back();
  int cols = in_->shape().back();

  #pragma omp parallel for
  for (int j = 0; j < rows; ++j) {
    float* so = out + j*cols;
    const float* sp = in + j*cols;

    float sum = 0.f;
    #pragma omp simd reduction(+:sum)
    for (int i = 0; i < cols; ++i) {
      sum += sp[i];
    }

    float mean = sum / cols;
    float sqSum = 0.f;
    #pragma omp simd reduction(+:sqSum)
    for (int i = 0; i < cols; ++i) {
      float ex = sp[i] - mean;
      sqSum += ex*ex;
    }

    float sigma = std::sqrt(eps + sqSum / cols);

    #pragma omp simd
    for (int i = 0; i < cols; ++i) {
      float t = alpha[i] * ((sp[i] - mean) / sigma);
      if (beta != nullptr) {
        t += beta[i];
      }

      so[i] = t;
    }
  }
}

void LayerNormalizationGrad(marian::Tensor gradX,
                            marian::Tensor gradGamma,
                            marian::Tensor gradBeta,
                            marian::Tensor adj,
                            marian::Tensor y,
                            marian::Tensor x,
                            marian::Tensor gamma,
                            marian::Tensor beta,
                            float eps) {
   ABORT("Not implemented!");
}


void Shift(marian::Tensor out_, marian::Tensor in_, marian::Shape shift, bool invert) {
  int offset = 0;
  for(int i = 0; i < shift.size(); ++i)
    offset += in_->shape().stride(i) * shift[i];

  if(invert)
    offset = -offset;

  float* out = out_->data();
  const float* in = in_->data();

  int length = out_->shape().elements();
  #pragma omp parallel for
  for (int i = 0; i < length; ++i) {
    if (i - offset < 0 || i - offset >= length) {
      out[i] = 0.f;
    } else {
      out[i] = in[i - offset];
    }
  }
}

void SetSparse(float* out,
               const std::vector<size_t>& indices,
               const std::vector<float>& values) {
  ABORT("Not implemented!");
}


void LSTMCellForward(marian::Tensor out, std::vector<marian::Tensor> inputs) {
  ABORT("Not implemented!");
}

void LSTMOutputForward(marian::Tensor out, std::vector<marian::Tensor> inputs) {
  ABORT("Not implemented!");
}

void LSTMCellBackward(std::vector<marian::Tensor> outputs,
                      std::vector<marian::Tensor> inputs,
                      marian::Tensor adj) {
  ABORT("Not implemented!");
}

void LSTMOutputBackward(std::vector<marian::Tensor> outputs,
                        std::vector<marian::Tensor> inputs,
                        marian::Tensor adj) {
  ABORT("Not implemented!");
}

void HighwayForward(marian::Tensor out,
                    const marian::Tensor in1,
                    const marian::Tensor in2,
                    const marian::Tensor t) {
  ABORT("Not implemented!");
}

void HighwayBackward(marian::Tensor out1,
                     marian::Tensor out2,
                     marian::Tensor outt,
                     const marian::Tensor in1,
                     const marian::Tensor in2,
                     const marian::Tensor t,
                     const marian::Tensor adj) {
  ABORT("Not implemented!");
}

void PoolingWithMaskingForward(marian::Tensor out,
                               marian::Tensor in,
                               marian::Tensor mask,
                               int width,
                               bool isEven) {
  ABORT("Not implemented!");
}

void PoolingWithMaskingBackward(marian::Tensor adj,
                                marian::Tensor adjIn,
                                marian::Tensor in,
                                marian::Tensor mask,
                                int width,
                                bool isEven) {
  ABORT("Not implemented!");
}

}
}  // namespace marian
