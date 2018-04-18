/* All or part of this file was contributed by Intel under license:
 *   Copyright (C) 2017-2018 Intel Corporation
 *   SPDX-License-Identifier: MIT
 */

#include "tensors/tensor_operators.h"
#include "tensors/cpu/backend.h"

#include "functional/functional.h"
#include "functional/tensor.h"

namespace marian {

namespace cpu {

inline float stableLogit(float x) {
  if(x >= 0) {
    float z = expf(-x);
    return 1.0 / (1.0 + z);
  } else {
    float z = expf(x);
    return z / (1.0 + z);
  }
}

void ConcatCont(Tensor out, const std::vector<Tensor>& inputs, int axis) {
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

void Concatenate1(Tensor out, const std::vector<Tensor>& inputs) {
  int rows = out->shape().elements() / out->shape().back();

  size_t offset = 0;
  int cols_out = out->shape().back();

  for(auto in : inputs) {
    ABORT_IF(rows != in->shape().elements() / in->shape().back(),
             "First dimension must be equal");
    int cols_in = in->shape().back();
    cpu::gInsertCols(
        out->data(), in->data(), rows, cols_in, cols_out, cols_in, offset, 0);
    offset += cols_in;
  }
}

void Concatenate(Tensor out, const std::vector<Tensor>& inputs, int ax) {
  if(ax == out->shape().size() - 1)
    Concatenate1(out, inputs);
  else
    ConcatCont(out, inputs, ax);
}

void Split1(std::vector<Tensor>& outputs, const Tensor in) {
  size_t offset = 0;
  int rows = in->shape().elements() / in->shape().back();
  int cols_in = in->shape().back();
  for(auto out : outputs) {
    ABORT_IF(rows != out->shape().elements() / out->shape().back(),
             "First dimension must be equal");
    int cols_out = out->shape().back();
    cpu::gInsertCols(
        out->data(), in->data(), rows, cols_out, cols_out, cols_in, 0, offset);
    offset += cols_out;
  }
}

void SplitCont(std::vector<Tensor>& outputs, const Tensor in, int axis) {
  int step = 1;
  for(int i = 0; i < axis; ++i)
    step *= in->shape()[i];

  size_t offset1 = 0;
  for(int i = 0; i < step; ++i) {
    for(auto out : outputs) {
      size_t size = out->shape().elements() / step;
      size_t offset2 = i * size;

      std::copy(in->data() + offset1,
                in->data() + offset1 + size,
                out->data() + offset2);

      offset1 += size;
    }
  }
}

void Deconcatenate(std::vector<Tensor>& outputs, const Tensor in, int ax) {
  if(ax == in->shape().size() - 1)
    Split1(outputs, in);
  else
    SplitCont(outputs, in, ax);
}

void Transpose0213(Tensor out, Tensor in) {
  int cols = in->shape()[-1];
  int rows = in->shape().elements() / in->shape()[-1];

  int r1 = in->shape()[-2];
  int r2 = in->shape()[-3];
  int rest = rows / (r1 * r2);

  for(int k = 0; k < rest; ++k) {
    int shift = k * r1 * r2;
    for(int j = 0; j < r1 * r2; ++j) {
      int src = j + shift;
      int dst = j / r1 + (j % r1) * r2 + shift;

      const float* inRow = in->data() + src * cols ;
      float* outRow = out->data() + dst * cols;

      std::copy(inRow, inRow + cols, outRow);
    }
  }
}

inline void transpose4x4_SSE(const float *A, float *B, const int lda, const int ldb) {
  __m128 row1 = _mm_load_ps(&A[0 * lda]);
  __m128 row2 = _mm_load_ps(&A[1 * lda]);
  __m128 row3 = _mm_load_ps(&A[2 * lda]);
  __m128 row4 = _mm_load_ps(&A[3 * lda]);
  _MM_TRANSPOSE4_PS(row1, row2, row3, row4);
  _mm_store_ps(&B[0 * ldb], row1);
  _mm_store_ps(&B[1 * ldb], row2);
  _mm_store_ps(&B[2 * ldb], row3);
  _mm_store_ps(&B[3 * ldb], row4);
}

// from https://stackoverflow.com/questions/16737298/what-is-the-fastest-way-to-transpose-a-matrix-in-c
#define ROUND_UP(x, s) (((x)+((s)-1)) & -(s))

void Transpose10(Tensor out, const Tensor in) {
  const float* A = in->data();
  float* B = out->data();

  const int n = in->shape().elements() / in->shape()[-1];
  const int m = in->shape()[-1];

  const int block_size = 16;
  int lda = ROUND_UP(m, block_size);
  int ldb = ROUND_UP(n, block_size);

  for(int i = 0; i < n; i += block_size) {
    for(int j = 0; j < m; j += block_size) {
      int max_i2 = i + block_size < n ? i + block_size : n;
      int max_j2 = j + block_size < m ? j + block_size : m;
      for(int i2 = i; i2 < max_i2; i2 += 4) {
        for(int j2 = j; j2 < max_j2; j2 += 4) {
          transpose4x4_SSE(&A[i2 * lda + j2], &B[j2 * ldb + i2], lda, ldb);
        }
      }
    }
  }
}

// @TODO: optimize this, currently it's quite horrible
void TransposeGeneric(Tensor out, Tensor in, const std::vector<int>& vAxis) {
  functional::Array<int, functional::Shape::size()> permute;
  int diff = functional::Shape::size() - vAxis.size();
  for(int i = 0; i < permute.size(); ++i)
    if(i < diff)
      permute[i] = i;
    else
      permute[i] = vAxis[i - diff] + diff;

  int length = out->shape().elements();

  constexpr size_t N = functional::Shape::size();
  functional::Array<int, N> oDims;
  functional::Array<int, N> pDims;
  functional::Tensor<float> gOut = out;
  functional::Tensor<float> gIn = in;

  for(int index = 0; index < length; ++index) {
    gOut.shape().dims(index, oDims);
    for(int i = 0; i < N; ++i)
      pDims[permute[i]] = oDims[i];
    gOut[index] = gIn[pDims];
  }
}

void TransposeND(Tensor out, Tensor in, const std::vector<int>& vAxis) {
  if(vAxis == std::vector<int>({0, 2, 1, 3}))
    Transpose0213(out, in);
  else if(vAxis == std::vector<int>({1, 0}) 
          && in->shape()[-1] % 16 == 0 
          && in->shape()[-2] % 16 == 0)
    Transpose10(out, in);
  else
    TransposeGeneric(out, in, vAxis);
}

void Softmax(Tensor out_, Tensor in_, Tensor mask_) {
  float* out = out_->data();
  const float* in = in_->data();
  const float* mask = mask_ ? mask_->data() : nullptr;

  int rows = out_->shape().elements() / out_->shape().back();
  int cols = out_->shape().back();

  for(int j = 0; j < rows; ++j) {
    float* so = out + j * cols;
    const float* sp = in + j * cols;
    const float* mp = mask ? mask + j * cols : nullptr;

    float max = sp[0];
    for(int i = 1; i < cols; ++i) {
      max = std::max(max, sp[i]);
    }

    float sum = 0.f;
    for(int i = 0; i < cols; ++i) {
      float ex = !mask || mp[i] ? expf(sp[i] - max) : 0.f;
      so[i] = ex;
      sum += ex;
    }

    for(int i = 0; i < cols; ++i) {
      so[i] /= sum;
    }
  }
}

void LogSoftmax(Tensor out_, Tensor in_) {
  float* out = out_->data();
  const float* in = in_->data();

  int rows = out_->shape().elements() / out_->shape().back();
  int cols = out_->shape().back();

  for(int j = 0; j < rows; ++j) {
    float* so = out + j * cols;
    const float* sp = in + j * cols;

    float max = sp[0];
    for(int i = 1; i < cols; ++i) {
      max = std::max(max, sp[i]);
    }

    float sum = 0.f;
    for(int i = 0; i < cols; ++i) {
      float sm = sp[i] - max;
      float ex = expf(sm);
      so[i] = sm;
      sum += ex;
    }

    for(int i = 0; i < cols; ++i) {
      so[i] -= logf(sum);
    }
  }
}

void SoftmaxGrad(Tensor grad_, Tensor adj_, Tensor val_) {
  int rows = grad_->shape().elements() / grad_->shape()[-1];
  int cols = grad_->shape()[-1];

  float* grad = grad_->data();
  const float* adj = adj_->data();
  const float* val = val_->data();

  for(size_t j = 0; j < rows; ++j) {
    float* gradRow = grad + j * cols;
    const float* adjRow = adj + j * cols;
    const float* valRow = val + j * cols;

    float sum = 0.f;
    for(size_t i = 0; i < cols; ++i) {
      sum += valRow[i] * adjRow[i];
    }

    for(size_t i = 0; i < cols; ++i) {
      gradRow[i] += valRow[i] * (adjRow[i] - sum);
    }
  }
}

void LogSoftmaxGrad(Tensor grad_, Tensor adj_, Tensor val_) {
  int rows = grad_->shape().elements() / grad_->shape()[-1];
  int cols = grad_->shape()[-1];

  float* grad = grad_->data();
  const float* adj = adj_->data();
  const float* val = val_->data();

  for(int j = 0; j < rows; ++j) {
    float* gradRow = grad + j * cols;
    const float* adjRow = adj + j * cols;
    const float* valRow = val + j * cols;

    float sum = 0.f;
    for(int i = 0; i < cols; ++i) {
      sum += adjRow[i];
    }

    for(int i = 0; i < cols; ++i) {
      gradRow[i] += adjRow[i] - sum * expf(valRow[i]);
    }
  }
}

void CopyRows(Tensor out_,
              const Tensor in_,
              const std::vector<size_t>& indices) {
  size_t cols = in_->shape()[1];
  size_t rows = indices.size();

  float* out = out_->data();
  const float* in = in_->data();

#pragma omp parallel for
  for(int j = 0; j < rows; ++j) {
    size_t dst = j;
    size_t src = indices[j];

    float* rowOut = out + dst * cols;
    const float* rowIn = in + src * cols;

    std::copy(rowIn, rowIn + cols, rowOut);
  }
}

void PasteRows(Tensor out_,
               const Tensor in_,
               const std::vector<size_t>& indices) {
  size_t cols = in_->shape()[-1];
  size_t rows = indices.size();

  float* out = out_->data();
  const float* in = in_->data();

  for(int j = 0; j < rows; ++j) {
    size_t dst = indices[j];  // not a permutation - may alias, unlike PasteCols
    size_t src = j;

    float* rowOut = out + dst * cols;
    const float* rowIn = in + src * cols;

    for(int i = 0; i < cols; ++i) {
      rowOut[i] += rowIn[i];
    }
  }
}

void CopyCols(Tensor out_,
              const Tensor in_,
              const std::vector<size_t>& indices) {
  size_t rows = in_->shape().elements() / in_->shape()[-1];
  size_t colsIn = in_->shape()[-1];
  size_t colsOut = indices.size();

  float* out = out_->data();
  const float* in = in_->data();

#pragma omp parallel for
  for(int j = 0; j < rows; ++j) {
    const float* rowIn = in + j * colsIn;
    float* rowOut = out + j * colsOut;

    for(int i = 0; i < colsOut; ++i) {
      rowOut[i] = rowIn[indices[i]];
    }
  }
}

void PasteCols(Tensor out_,
               const Tensor in_,
               const std::vector<size_t>& indices) {
  size_t rows = out_->shape().elements() / out_->shape()[-1];
  size_t colsOut = out_->shape()[-1];
  size_t colsIn = indices.size();

  float* out = out_->data();
  const float* in = in_->data();

  /* n.b. Unlike PasteRows, currently appears safe to assume indices[i] is a
   *      permutation i.e. no racy aliases, and no need to sum vs. just assign.
   */
  for(int j = 0; j < rows; ++j) {
    const float* rowIn = in + j * colsIn;
    float* rowOut = out + j * colsOut;

    // @TODO: should this be a sum?
    for(int i = 0; i < colsIn; ++i) {
      rowOut[indices[i]] = rowIn[i];
    }
  }
}

void Select(Tensor out,
            const Tensor in,
            int axis,
            const std::vector<size_t>& indices,
            Ptr<Allocator> allocator) {
  ABORT("Not implemented!");
}

void Insert(Tensor out,
            const Tensor in,
            int axis,
            const std::vector<size_t>& indices,
            Ptr<Allocator> allocator) {
  ABORT("Not implemented!");
}

void GRUFastForward(Tensor out_, std::vector<Tensor> inputs, bool final) {
  int rows = out_->shape().elements() / out_->shape().back();
  int cols = out_->shape().back();

  float* out = out_->data();

  const float* state = inputs[0]->data();
  const float* xW = inputs[1]->data();
  const float* sU = inputs[2]->data();
  const float* b = inputs[3]->data();
  const float* mask = inputs.size() > 4 ? inputs[4]->data() : nullptr;

#pragma omp parallel for
  for(int j = 0; j < rows; ++j) {
    float m = !mask || mask[j];
    float* rowOut = out + j * cols;
    const float* rowState = state + j * cols;

    const float* xWrow = xW + j * cols * 3;
    const float* sUrow = sU + j * cols * 3;

#pragma omp simd
    for(int i = 0; i < cols; ++i) {
      // @TODO: stable logit
      float r = stableLogit(xWrow[i] + sUrow[i] + b[i]);

      int k = i + cols;

      float z = stableLogit(xWrow[k] + sUrow[k] + b[k]);

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

void GRUFastBackward(std::vector<Tensor> outputs,
                     std::vector<Tensor> inputs,
                     Tensor adj_,
                     bool final) {
  int rows = adj_->shape().elements() / adj_->shape().back();
  int cols = adj_->shape().back();

  float* outState = outputs[0] ? outputs[0]->data() : nullptr;
  float* outXW = outputs[1] ? outputs[1]->data() : nullptr;
  float* outSU = outputs[2] ? outputs[2]->data() : nullptr;
  float* outB = outputs[3] ? outputs[3]->data() : nullptr;

  const float* state = inputs[0]->data();
  const float* xW = inputs[1]->data();
  const float* sU = inputs[2]->data();
  const float* b = inputs[3]->data();
  const float* mask = inputs.size() > 4 ? inputs[4]->data() : 0;
  const float* adj = adj_->data();

#pragma omp parallel
  for(int j = 0; j < rows; ++j) {
    float m = !mask || mask[j];

    float* rowOutState = outState + j * cols;
    float* rowOutXW = outXW + j * cols * 3;
    float* rowOutSU = outSU + j * cols * 3;

    const float* rowState = state + j * cols;
    const float* rowXW = xW + j * cols * 3;
    const float* rowSU = sU + j * cols * 3;
    const float* rowAdj = adj + j * cols;

#pragma omp for simd nowait
    for(int i = 0; i < cols; ++i) {
      int k = i + cols;
      int l = i + 2 * cols;

      float r = stableLogit(rowXW[i] + rowSU[i] + b[i]);
      float z = stableLogit(rowXW[k] + rowSU[k] + b[k]);

      float h;
      if(final)
        h = std::tanh(rowXW[l] + (rowSU[l] + b[l]) * r);
      else
        h = std::tanh(rowXW[l] + rowSU[l] * r + b[l]);

      float adj = rowAdj[i];

      float t = (1 - z) * (1 - h * h);

      // df/ds
      if(outState)
        rowOutState[i] += (m * z - m + 1) * adj;

      // df/d(xW_r) ...
      float dfdxW_r = m * r * (1 - r) * t * adj;
      if(final)
        dfdxW_r *= rowSU[l] + b[l];
      else
        dfdxW_r *= rowSU[l];
      if(outXW)
        rowOutXW[i] += dfdxW_r;
      if(outSU)
        rowOutSU[i] += dfdxW_r;
      if(outB)
        outB[i] += dfdxW_r;

      // df/d(xW_z) ...
      float dfdxW_z = m * (1 - z) * z * (rowState[i] - h) * adj;
      if(outXW)
        rowOutXW[k] += dfdxW_z;
      if(outSU)
        rowOutSU[k] += dfdxW_z;
      if(outB)
        outB[k] += dfdxW_z;

      // df/d(xW_x) ...
      float dfdxW_x = m * t * adj;
      if(outXW)
        rowOutXW[l] += dfdxW_x;
      if(outSU)
        rowOutSU[l] += dfdxW_x * r;
      if(outB)
        if(final)
          outB[l] += dfdxW_x * r;
        else
          outB[l] += dfdxW_x;
    }
  }
}

void CrossEntropyPick(Tensor out_, Tensor in_, Tensor pick_) {
  float* out = out_->data();
  Shape& outShape = out_->shape();
  const float* in = in_->data();
  Shape& inShape = in_->shape();
  float* pick = pick_->data();

  int rows = inShape.elements() / inShape.back();
  int cols = inShape.back();

#pragma omp parallel for
  for(int j = 0; j < rows; ++j) {
    const float* sp = in + j * cols;
    float max = sp[0];
#pragma omp simd reduction(max : max)
    for(int i = 1; i < cols; ++i) {
      max = std::max(max, sp[i]);
    }

    float sum = 0.f;
#pragma omp simd reduction(+ : sum)
    for(int i = 0; i < cols; ++i) {
      sum += std::exp(sp[i] - max);
    }

    // cross-entropy
    int i = pick[j];
    // This appears to be safe i.e. that i >= 0 && i < cols is known
    out[j] = std::log(sum) - sp[i] + max;
  }
}

void CrossEntropyPickBackward(Tensor out_,
                              Tensor adj_,
                              Tensor a,
                              Tensor pick_) {
  float* out = out_->data();
  Shape& outShape = out_->shape();
  const float* adj = adj_->data();
  const float* in = a->data();
  const float* pick = pick_->data();

  int rows = outShape.elements() / outShape.back();
  int cols = outShape.back();

#pragma omp parallel for
  for(int j = 0; j < rows; ++j) {
    const float* sp = in + j * cols;
    float* so = out + j * cols;

    float max = sp[0];
    for(int i = 1; i < cols; ++i) {
      max = std::max(max, sp[i]);
    }

    float sum = 0.f;
    for(int i = 0; i < cols; ++i) {
      sum += std::exp(sp[i] - max);
    }

    // cross-entropy
    for(int i = 0; i < cols; ++i) {
      float sub = (float)(i == (int)pick[j]);
      so[i] += adj[j] * (std::exp(sp[i] - max) / sum - sub);
    }
  }
}

float L2Norm(Tensor in) {
  float sum = 0.f;
  size_t size = in->size();
  const float* data = in->data();
#pragma omp parallel for simd reduction(+ : sum)
  for(size_t i = 0; i < size; ++i) {
    sum += data[i] * data[i];
  }
  return std::sqrt(sum);
}

void Att(Tensor out_, Tensor va_, Tensor context_, Tensor state_) {
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
  for(size_t j = 0; j < rows; ++j) {
    const float* vaRow = va;
    const float* ctxRow = ctx + (j % (b * t)) * cols;
    const float* stateRow = state + ((j / (b * t)) * b + j % b) * cols;

    float sum = 0.f;
#pragma omp simd reduction(+ : sum)
    for(size_t i = 0; i < cols; ++i) {
      float z = ctxRow[i] + stateRow[i];
      sum += std::tanh(z) * vaRow[i];
    }

    out[j] = sum;
  }
}

void AttBack(Tensor gVa_,
             Tensor gContext_,
             Tensor gState_,
             Tensor va_,
             Tensor context_,
             Tensor state_,
             Tensor adj_) {
  float* gVa = gVa_->data();
  float* gContext = gContext_->data();
  float* gState = gState_->data();

  const float* va = va_->data();
  const float* context = context_->data();
  const float* state = state_->data();
  const float* adj = adj_->data();

  size_t m = adj_->shape().elements() / adj_->shape()[-1];
  size_t k = context_->shape()[-1];
  size_t n = context_->shape()[-2];

#pragma omp parallel for reduction(+ : gState[ : n* k], gVa[ : k])
  for(size_t j = 0; j < m; ++j) {
    float* gcRow = gContext + j * k;
    float* gsRow = gState + (j % n) * k;

    const float* cRow = context + j * k;
    const float* sRow = state + (j % n) * k;

    float adj_j = adj[j];

#pragma omp simd
    for(size_t i = 0; i < k; ++i) {
      float z = cRow[i] + sRow[i];

      float t = std::tanh(z);
      float r = va[i] * (1.f - t * t);

      float r_adj_j = r * adj_j;
      gcRow[i] += r_adj_j;
      gsRow[i] += r_adj_j;

      gVa[i] += t * adj_j;
    }
  }
}

void LayerNormalization(Tensor out_,
                        Tensor in_,
                        Tensor gamma_,
                        Tensor beta_,
                        float eps) {
  float* out = out_->data();
  const float* in = in_->data();
  const float* alpha = gamma_->data();
  const float* beta = beta_ ? beta_->data() : nullptr;

  int rows = in_->shape().elements() / in_->shape().back();
  int cols = in_->shape().back();

#pragma omp parallel for
  for(int j = 0; j < rows; ++j) {
    float* so = out + j * cols;
    const float* sp = in + j * cols;

    float sum = 0.f;
#pragma omp simd reduction(+ : sum)
    for(int i = 0; i < cols; ++i) {
      sum += sp[i];
    }

    float mean = sum / cols;
    float sqSum = 0.f;
#pragma omp simd reduction(+ : sqSum)
    for(int i = 0; i < cols; ++i) {
      float ex = sp[i] - mean;
      sqSum += ex * ex;
    }

    float sigma = std::sqrt(eps + sqSum / cols);

#pragma omp simd
    for(int i = 0; i < cols; ++i) {
      float t = alpha[i] * ((sp[i] - mean) / sigma);
      if(beta != nullptr) {
        t += beta[i];
      }

      so[i] = t;
    }
  }
}

void LayerNormalizationGrad(Tensor gradX_,
                            Tensor gradGamma_,
                            Tensor gradBeta_,
                            Tensor adj_,
                            Tensor y_,
                            Tensor x_,
                            Tensor gamma_,
                            Tensor beta_,
                            float eps) {
  float* gradX = gradX_->data();
  float* gradGamma = gradGamma_->data();
  float* gradBeta = gradBeta_ ? gradBeta_->data() : nullptr;
  float* adj = adj_->data();
  float* y = y_->data();
  float* x = x_->data();
  float* gamma = gamma_->data();
  float* beta = beta_ ? beta_->data() : nullptr;

  size_t rows = y_->shape().elements() / y_->shape()[-1];
  size_t cols = y_->shape()[-1];

  if(beta) {
#pragma omp parallel for reduction(+ : gradGamma[ : cols], gradBeta[ : cols])
    for(size_t j = 0; j < rows; ++j) {
      const float* xRow = x + j * cols;
      const float* yRow = y + j * cols;
      const float* adjRow = adj + j * cols;
      float* gradXRow = gradX + j * cols;

      float sum_x = 0.f;
      float sum_adj = 0.f;
      float sum_adj_x = 0.f;
      float sum_sqr = 0.f;

#pragma omp simd reduction(+ : sum_x, sum_adj_x, sum_adj)
      for(size_t i = 0; i < cols; ++i) {
        sum_x += xRow[i];
        sum_adj_x += adjRow[i] * (yRow[i] - (beta ? beta[i] : 0.f)) / gamma[i];
        sum_adj += adjRow[i];
      }

      float mean = sum_x / cols;
#pragma omp simd reduction(+ : sum_sqr)
      for(size_t i = 0; i < cols; ++i) {
        float ex = xRow[i] - mean;
        sum_sqr += ex * ex;
      }

      float sigma = std::sqrt(eps + sum_sqr / cols);
#pragma omp simd
      for(size_t i = 0; i < cols; ++i) {
        float grad_x = 0.f;
        float x_hat = (yRow[i] - beta[i]) / gamma[i];
        grad_x += cols * adjRow[i];
        grad_x -= sum_adj;
        grad_x -= sum_adj_x * x_hat;
        grad_x /= cols * sigma;

        gradXRow[i] += gamma[i] * grad_x;
        gradGamma[i] += adjRow[i] * x_hat;
        gradBeta[i] += adjRow[i];
      }
    }
  } else {
#pragma omp parallel for reduction(+ : gradGamma[ : cols])
    for(size_t j = 0; j < rows; ++j) {
      const float* xRow = x + j * cols;
      const float* yRow = y + j * cols;
      const float* adjRow = adj + j * cols;
      float* gradXRow = gradX + j * cols;

      float sum_x = 0.f;
      float sum_adj = 0.f;
      float sum_adj_x = 0.f;
      float sum_sqr = 0.f;

#pragma omp simd reduction(+ : sum_x, sum_adj_x, sum_adj)
      for(size_t i = 0; i < cols; ++i) {
        sum_x += xRow[i];
        sum_adj_x += adjRow[i] * (yRow[i] - (beta ? beta[i] : 0.f)) / gamma[i];
        sum_adj += adjRow[i];
      }

      float mean = sum_x / cols;
#pragma omp simd reduction(+ : sum_sqr)
      for(size_t i = 0; i < cols; ++i) {
        float ex = xRow[i] - mean;
        sum_sqr += ex * ex;
      }

      float sigma = std::sqrt(eps + sum_sqr / cols);
#pragma omp simd
      for(size_t i = 0; i < cols; ++i) {
        float grad_x = 0.f;
        float x_hat = yRow[i] / gamma[i];
        grad_x += cols * adjRow[i];
        grad_x -= sum_adj;
        grad_x -= sum_adj_x * x_hat;
        grad_x /= cols * sigma;

        gradXRow[i] += gamma[i] * grad_x;
        gradGamma[i] += adjRow[i] * x_hat;
      }
    }
  }
}

void Shift(Tensor out_, Tensor in_, marian::Shape shift, bool invert) {
  int offset = 0;
  for(int i = 0; i < shift.size(); ++i)
    offset += in_->shape().stride(i) * shift[i];

  if(invert)
    offset = -offset;

  float* out = out_->data();
  const float* in = in_->data();

  int length = out_->shape().elements();
#pragma omp parallel for
  for(int i = 0; i < length; ++i) {
    if(i - offset < 0 || i - offset >= length) {
      out[i] = 0.f;
    } else {
      out[i] = in[i - offset];
    }
  }
}

void SetSparse(float* out,
               const std::vector<size_t>& indices,
               const std::vector<float>& values) {
  int length = indices.size();
  for(int index = 0; index < length; ++index) {
    out[indices[index]] = values[index];
  }
}

void LSTMCellForward(Tensor out_, std::vector<Tensor> inputs) {
  int rows = out_->shape().elements() / out_->shape()[-1];
  int cols = out_->shape()[-1];

  float* out = out_->data();
  const float* cell = inputs[0]->data();
  const float* xW = inputs[1]->data();
  const float* sU = inputs[2]->data();
  const float* b = inputs[3]->data();
  const float* mask = inputs.size() > 4 ? inputs[4]->data() : nullptr;

  for(int j = 0; j < rows; ++j) {
    float m = !mask || mask[j];

    float* rowOut = out + j * cols;
    const float* rowCell = cell + j * cols;

    const float* xWrow = xW + j * cols * 4;
    const float* sUrow = sU + j * cols * 4;

    for(int i = 0; i < cols; ++i) {
      float gf = stableLogit(xWrow[i] + sUrow[i] + b[i]);

      int k = i + cols;
      float gi = stableLogit(xWrow[k] + sUrow[k] + b[k]);

      int l = i + 2 * cols;
      float gc = std::tanh(xWrow[l] + sUrow[l] + b[l]);

      float cout = gf * rowCell[i] + gi * gc;
      rowOut[i] = m * cout + (1 - m) * rowCell[i];
    }
  }
}

void LSTMOutputForward(Tensor out_, std::vector<Tensor> inputs) {
  int rows = out_->shape().elements() / out_->shape()[-1];
  int cols = out_->shape()[-1];

  float* out = out_->data();
  const float* cell = inputs[0]->data();
  const float* xW = inputs[1]->data();
  const float* sU = inputs[2]->data();
  const float* b = inputs[3]->data();

  for(int j = 0; j < rows; ++j) {
    float* rowOut = out + j * cols;
    const float* rowCell = cell + j * cols;

    const float* xWrow = xW + j * cols * 4;
    const float* sUrow = sU + j * cols * 4;

    for(int i = 0; i < cols; ++i) {
      int k = i + 3 * cols;
      float go = stableLogit(xWrow[k] + sUrow[k] + b[k]);

      rowOut[i] = go * std::tanh(rowCell[i]);
    }
  }
}

void LSTMCellBackward(std::vector<Tensor> outputs,
                      std::vector<Tensor> inputs,
                      Tensor adj_) {
  int rows = adj_->shape().elements() / adj_->shape()[-1];
  int cols = adj_->shape()[-1];

  float* outCell = outputs[0] ? outputs[0]->data() : nullptr;
  float* outXW = outputs[1] ? outputs[1]->data() : nullptr;
  float* outSU = outputs[2] ? outputs[2]->data() : nullptr;
  float* outB = outputs[3] ? outputs[3]->data() : nullptr;

  const float* cell = inputs[0]->data();
  const float* xW = inputs[1]->data();
  const float* sU = inputs[2]->data();
  const float* b = inputs[3]->data();

  const float* mask = inputs.size() > 4 ? inputs[4]->data() : nullptr;
  const float* adj = adj_->data();

  for(int j = 0; j < rows; ++j) {
    float m = !mask || mask[j];

    float* rowOutCell = outCell + j * cols;
    float* rowOutXW = outXW + j * cols * 4;
    float* rowOutSU = outSU + j * cols * 4;

    const float* rowCell = cell + j * cols;
    const float* xWrow = xW + j * cols * 4;
    const float* sUrow = sU + j * cols * 4;

    const float* rowAdj = adj + j * cols;

    for(int i = 0; i < cols; ++i) {
      float gf = stableLogit(xWrow[i] + sUrow[i] + b[i]);

      int k = i + cols;
      float gi = stableLogit(xWrow[k] + sUrow[k] + b[k]);

      int l = i + 2 * cols;
      float gc = std::tanh(xWrow[l] + sUrow[l] + b[l]);

      float adj = rowAdj[i];

      // dc/dx_{t-1}
      if(outCell) {
        rowOutCell[i] += (m * gf - m + 1) * adj;
      }

      // dc/d(b_f) = dc/d(xW_f) ...
      float dcdxf = m * rowCell[i] * gf * (1 - gf) * adj;
      if(outXW) {
        rowOutXW[i] += dcdxf;
      }
      if(outSU) {
        rowOutSU[i] += dcdxf;
      }
      if(outB) {
        outB[i] += dcdxf;
      }

      // dc/d(b_i) ...
      float dcdb_i = m * gc * gi * (1 - gi) * adj;
      if(outXW) {
        rowOutXW[k] += dcdb_i;
      }
      if(outSU) {
        rowOutSU[k] += dcdb_i;
      }
      if(outB) {
        outB[k] += dcdb_i;
      }

      // dc/d(b_c) ...
      float dcdxc = m * gi * (1 - gc * gc) * adj;
      if(outXW) {
        rowOutXW[l] += dcdxc;
      }
      if(outSU) {
        rowOutSU[l] += dcdxc;
      }
      if(outB) {
        outB[l] += dcdxc;
      }
    }
  }
}

void LSTMOutputBackward(std::vector<Tensor> outputs,
                        std::vector<Tensor> inputs,
                        Tensor adj_) {
  int rows = adj_->shape().elements() / adj_->shape()[-1];
  int cols = adj_->shape()[-1];

  float* outCell = outputs[0] ? outputs[0]->data() : nullptr;
  float* outXW = outputs[1] ? outputs[1]->data() : nullptr;
  float* outSU = outputs[2] ? outputs[2]->data() : nullptr;
  float* outB = outputs[3] ? outputs[3]->data() : nullptr;

  const float* cell = inputs[0]->data();
  const float* xW = inputs[1]->data();
  const float* sU = inputs[2]->data();
  const float* b = inputs[3]->data();

  const float* adj = adj_->data();

  for(int j = 0; j < rows; ++j) {
    float* rowOutCell = outCell + j * cols;
    float* rowOutXW = outXW + j * cols * 4;
    float* rowOutSU = outSU + j * cols * 4;

    const float* rowCell = cell + j * cols;
    const float* xWrow = xW + j * cols * 4;
    const float* sUrow = sU + j * cols * 4;

    const float* rowAdj = adj + j * cols;

    for(int i = 0; i < cols; ++i) {
      int k = i + 3 * cols;
      float go = stableLogit(xWrow[k] + sUrow[k] + b[k]);

      float t = std::tanh(rowCell[i]);

      float adj = rowAdj[i];

      // dc/dc_{t-1}
      if(outCell) {
        rowOutCell[i] += go * (1 - t * t) * adj;
      }

      // dc/d(b_o) = dc/d(xW_f) ...
      float dcdxo = t * go * (1 - go) * adj;
      if(outXW) {
        rowOutXW[k] += dcdxo;
      }
      if(outSU) {
        rowOutSU[k] += dcdxo;
      }
      if(outB) {
        outB[k] += dcdxo;
      }
    }
  }
}

void HighwayForward(Tensor out,
                    const Tensor in1,
                    const Tensor in2,
                    const Tensor t) {
  ABORT("Not implemented!");
}

void HighwayBackward(Tensor out1,
                     Tensor out2,
                     Tensor outt,
                     const Tensor in1,
                     const Tensor in2,
                     const Tensor t,
                     const Tensor adj) {
  ABORT("Not implemented!");
}

void PoolingWithMaskingForward(Tensor out,
                               Tensor in,
                               Tensor mask,
                               int width,
                               bool isEven) {
  ABORT("Not implemented!");
}

void PoolingWithMaskingBackward(Tensor adj,
                                Tensor adjIn,
                                Tensor in,
                                Tensor mask,
                                int width,
                                bool isEven) {
  ABORT("Not implemented!");
}
}
}  // namespace marian
