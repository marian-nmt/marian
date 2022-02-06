/* All or part of this file was contributed by Intel under license:
 *   Copyright (C) 2017-2018 Intel Corporation
 *   SPDX-License-Identifier: MIT
 */

#include "tensors/tensor_operators.h"
#include "tensors/cpu/backend.h"
#include "tensors/allocator.h"

#include "functional/approx.h"
#include "functional/functional.h"
#include "functional/tensor.h"
#include "functional/operators.h"

#if MKL_FOUND
#include <mkl.h>
#endif

namespace marian {

namespace cpu {

void IsNaN(const Tensor /*in*/, Ptr<Allocator> /*allocator*/, bool& /*isNaN*/, bool& /*isInf*/) {
  ABORT("Not implemented");
}

bool SanitizeGradient(marian::Tensor /*in*/, Ptr<Allocator> /*allocator*/, bool /*pruneNaN*/, bool /*clipInf*/) {
  ABORT("Not implemented");
}

template <bool add, typename To, typename From>
void CopyCastTo(To* out, const From* in, int length) {
  for(int i = 0; i < length; ++i)
#ifdef _MSC_VER
#pragma warning (push)
#pragma warning (disable: 4244)  // 'argument': conversion from 'const From' to 'float', possible loss of data
#endif
    if(add)
      out[i] += (To)in[i];
    else
      out[i]  = (To)in[i];
#ifdef _MSC_VER
#pragma warning (pop)
#endif
}

// Casting has been factored into two functions "CopyCastFrom" and
// "CopyCastTo". This only serves the purpuse to automatically create
// the full Carthesian product of possible type cast via template magic.
// Extending CopyCast and CopyCastFrom with a new branch in the "if" clause
// adds all possible variants.
template <bool add, typename T>
void CopyCastFrom(Tensor out, const T* in, int length) {
  if(out->type() == Type::float32) {
    CopyCastTo<add>(out->data<float>(), in, length);
  } else if(out->type() == Type::float16) {
    CopyCastTo<add>(out->data<float16>(), in, length);
  } else {
    ABORT("CopyCastTo to type {} not implemented", out->type());
  }
}

// currently useless on the CPU until more types are added
void CopyCast(Tensor out, const Tensor in) {
  if(in->type() == Type::float32) {
    CopyCastFrom</*add=*/false>(out, in->data<float>(), (int)in->size());
  } else if(in->type() == Type::float16) {
    CopyCastFrom</*add=*/false>(out, in->data<float16>(), (int)in->size());
  } else if(in->type() == Type::uint32) {
    CopyCastFrom</*add=*/false>(out, in->data<uint32_t>(), (int)in->size());
  } else {
    ABORT("CopyCastFrom from type {} not implemented", in->type());
  }
}

// currently useless on the CPU until more types are added
void AddCast(Tensor out, const Tensor in) {
  if(in->type() == Type::float32) {
    CopyCastFrom</*add=*/true>(out, in->data<float>(), (int)in->size());
  } else if(in->type() == Type::float16) {
    CopyCastFrom</*add=*/true>(out, in->data<float16>(), (int)in->size());
  } else if(in->type() == Type::uint32) {
    CopyCastFrom</*add=*/true>(out, in->data<uint32_t>(), (int)in->size());
  } else {
    ABORT("CopyCastFrom from type {} not implemented", in->type());
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

template <bool add>
inline void gInsertCols(float* out,
                        const float* in,
                        size_t rows,
                        size_t cols,
                        size_t cols_out,
                        size_t cols_in,
                        size_t offset_out,
                        size_t offset_in) {
  for(size_t j = 0; j < rows; ++j) {
    float* rowOut = out + j * cols_out + offset_out;
    const float* rowIn = in + j * cols_in + offset_in;
    for(size_t i = 0; i < cols; ++i) {
      if(add) // this was solved earlier via beta * rowOut[i] with beta in {0,1} but 0 * nan in uninitialized tensors will result in nan.
        rowOut[i] += rowIn[i];
      else
        rowOut[i]  = rowIn[i];
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
    cpu::gInsertCols<false>(out->data(),
                            in->data(),
                            rows,
                            cols_in,
                            cols_out,
                            cols_in,
                            offset,
                            0);
    offset += cols_in;
  }
}

void Concatenate(Tensor out, const std::vector<Tensor>& inputs, int ax) {
   if(ax == (int)out->shape().size() - 1)
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

    // set last parameter to 1 to enable += instead of =
    // @TODO: do this in a more principled ways accross all/most kernels
    cpu::gInsertCols<true>(out->data(),
                           in->data(),
                           rows,
                           cols_out,
                           cols_out,
                           cols_in,
                           0,
                           offset);
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

      // BUG: This overwrites gradients!
      // std::copy(in->data() + offset1,
      //          in->data() + offset1 + size,
      //          out->data() + offset2);

      // Fixes gradient problem, @TODO: check performance
      std::transform(in->data() + offset1,
                     in->data() + offset1 + size,
                     out->data() + offset2,
                     out->data() + offset2,
                     [](float a, float b) { return a + b; });

      offset1 += size;
    }
  }
}

void Deconcatenate(std::vector<Tensor>& outputs, const Tensor in, int ax) {
  if(ax == (int)in->shape().size() - 1)
    Split1(outputs, in);
  else
    SplitCont(outputs, in, ax);
}

template <bool add>
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

      const float* inRow = in->data() + src * cols;
      float* outRow = out->data() + dst * cols;

      if(!add) {
        // mostly for fast forward computation
        std::copy(inRow, inRow + cols, outRow);
      } else {
        for(int i = 0; i < cols; ++i) {
          outRow[i] += inRow[i];
        }
      }
    }
  }
}

// This function is called only when MKL is available.
#if MKL_FOUND
// Given a 4D array, transpose (swap) the initial 3 dimensions while keeping the last dimension.
// e.g. 1234 --> 2134, 1234 --> 3214 (4 is always kept).
// This is an optimized version for swapping first 3 dimensions
// assuming the last dimension is large enough to get benefits from vectorized copy.
//
// @param out output tensor
// @param in input tensor
// @param vAxis target (transposed) axes of each given axes
template <bool add>
void TransposeFirst3In4(Tensor out, Tensor in, const std::vector<int>& vAxis) {
  ABORT_IF(vAxis.size() != 4, "This function handles only 4D arrays.");
  int innermost = in->shape()[-1];

  int l1 = in->shape()[vAxis[0]];
  int l2 = in->shape()[vAxis[1]];
  int l3 = in->shape()[vAxis[2]];

  // find the mapping between the transposed output dimensional indices (oi, oj, ok)
  // and original input dimensional indices (i, j, k)
#pragma omp parallel for
  for(int k = 0; k < l1; ++k) {
    int shift = k * l2 * l3;
    for(int j = 0; j < l2; ++j) {
      for(int i = 0; i < l3; ++i) {
        int oi, oj, ok;
        if(vAxis[0] == 0) {
          if(vAxis[1] == 1) {
            oi = i; oj = j; ok = k;
          } else {
            oi = j; oj = i; ok = k;
          }
        } else if(vAxis[0] == 1) {
          if(vAxis[1] == 0) {
            oi = i; oj = k; ok = j;
          } else {
            oi = j; oj = k; ok = i;
          }
        } else {
          if(vAxis[1] == 0) {
            oi = k; oj = i; ok = j;
          } else {
            oi = k; oj = j; ok = i;
          }
        }
        int src = ok * in->shape()[1] * in->shape()[2] + oj * in->shape()[2] + oi;
        int dst = l3 * j + shift + i;

        const float* inRow = in->data() + src * innermost;
        float* outRow = out->data() + dst * innermost;

        if(!add) {
          mkl_somatcopy('R', 'N', 1, innermost, 1.0f, inRow, innermost, outRow, innermost);
        } else {
          for(int ii = 0; ii < innermost; ++ii) {
            outRow[ii] += inRow[ii];
          }
        }
      }
    }
  }
}
#endif  // MKL_FOUND

inline void transpose4x4_SSE(const float* A,
                             float* B,
                             const int lda,
                             const int ldb) {
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

// from
// https://stackoverflow.com/questions/16737298/what-is-the-fastest-way-to-transpose-a-matrix-in-c
#define ROUND_UP(x, s) (((x) + ((s)-1)) & -(s))

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
template <bool add>
void TransposeGeneric(Tensor out, Tensor in, const std::vector<int>& vAxis) {
  functional::Array<int, functional::Shape::size()> permute;
  int diff = int(functional::Shape::size() - vAxis.size());
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
    for(size_t i = 0; i < N; ++i)
      pDims[permute[i]] = oDims[i];

    // @TODO: where does this change come from?
    int inIndex = gIn.shape().index(pDims);

    // @TODO: use internal conversion instead of raw indices
    if(add)
      gOut.data()[index] += gIn.data()[inIndex];
    else
      gOut.data()[index] = gIn.data()[inIndex];
  }
}

void TransposeND(Tensor out, Tensor in, const std::vector<int>& vAxis) {
  if(vAxis == std::vector<int>({0, 2, 1, 3}))
    Transpose0213<false>(out, in);
#if MKL_FOUND
  else if(vAxis.size() == 4 && vAxis[3] == 3)
    TransposeFirst3In4<false>(out, in, vAxis);
#endif  // MKL_FOUND
  else if(vAxis == std::vector<int>({1, 0}) && in->shape()[-1] % 16 == 0
          && in->shape()[-2] % 16 == 0)
    Transpose10(out, in);
  else
    TransposeGeneric<false>(out, in, vAxis);
}

void TransposeNDGrad(Tensor out, Tensor in, const std::vector<int>& vAxis) {
  if(vAxis == std::vector<int>({0, 2, 1, 3}))
    Transpose0213<true>(out, in);
  else
    TransposeGeneric<true>(out, in, vAxis);
}

template <typename ElementType>
void Softmax(Tensor out, Tensor in) {
  using namespace functional;
  functional::Tensor<ElementType> fout = out;
  const functional::Tensor<ElementType> fin = in;

  ElementType* pOut = fout.data();
  const ElementType* pIn = fin.data();

  int rows = fout.shape().elements() / fout.shape().back();
  int cols = fout.shape().back();

  for(int j = 0; j < rows; ++j) {
    ElementType* so = pOut + j * cols;
    const ElementType* sp = pIn + j * cols;

    ElementType max = sp[0];
    for(int i = 1; i < cols; ++i) {
      max = Ops<ElementType>::max(max, sp[i]);
    }

    // if ElementType is a complex type, e.g. float32x8, find the max of these 8 values
    typename Ops<ElementType>::Single maxs = Ops<ElementType>::maxReduce(max);

    ElementType sum = 0.f;
    for(int i = 0; i < cols; ++i) {
      ElementType ex = Ops<ElementType>::exp(Ops<ElementType>::sub(sp[i], maxs));
      sum = Ops<ElementType>::add(sum, ex);
      so[i] = ex;
    }

    // if ElementType is a complex type, e.g. float32x8, sum these 8 values
    typename Ops<ElementType>::Single sums = Ops<ElementType>::sumReduce(sum);

    for(int i = 0; i < cols; ++i) {
      so[i] = Ops<ElementType>::div(so[i], sums);
    }
  }
}


void Softmax(Tensor out, Tensor in) {
  matchOrAbort<float>(out->type());
  matchOrAbort<float>(in->type());

#ifdef __AVX__
  if(out->shape()[-1] % 8 == 0) {
    Softmax<float32x8>(out, in);
    return;
  }
#endif
  if(out->shape()[-1] % 4 == 0) {
    Softmax<float32x4>(out, in);
  } else {
    Softmax<float>(out, in);
  }
}


template <typename ElementType>
void LogSoftmax(Tensor out, Tensor in) {

  using namespace functional;
  functional::Tensor<ElementType> fout = out;
  const functional::Tensor<ElementType> fin = in;

  ElementType* pOut = fout.data();
  const ElementType* pIn = fin.data();

  int rows = fout.shape().elements() / fout.shape().back();
  int cols = fout.shape().back();

  for(int j = 0; j < rows; ++j) {
    ElementType* so = pOut + j * cols;
    const ElementType* sp = pIn + j * cols;

    ElementType max = sp[0];
    for(int i = 1; i < cols; ++i) {
      max = Ops<ElementType>::max(max, sp[i]);
    }
    typename Ops<ElementType>::Single maxs = Ops<ElementType>::maxReduce(max); // global maximum

    ElementType sum = 0.f;
    for(int i = 0; i < cols; ++i) {
      ElementType sm = Ops<ElementType>::sub(sp[i], maxs);
      sum = Ops<ElementType>::add(sum, Ops<ElementType>::exp(sm));
      so[i] = sm;
    }
    typename Ops<ElementType>::Single sums = Ops<ElementType>::sumReduce(sum); // global sum

    ElementType logSum = Ops<ElementType>::log(sums); // broadcasts Single to ElementType
    for(int i = 0; i < cols; ++i) {
      so[i] = Ops<ElementType>::sub(so[i], logSum);
    }
  }
}

void LogSoftmax(Tensor out, Tensor in) {
  matchOrAbort<float>(out->type());
  matchOrAbort<float>(in->type());

#ifdef __AVX__
  if(out->shape()[-1] % 8 == 0) {
    LogSoftmax<float32x8>(out, in);
    return;
  }
#endif
  if(out->shape()[-1] % 4 == 0) {
    LogSoftmax<float32x4>(out, in);
  } else {
    LogSoftmax<float>(out, in);
  }
}

// @TODO: Remove remaining underscores in CPU kernels
void SoftmaxGrad(Tensor grad_, Tensor adj_, Tensor val_) {
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
      sum += valRow[i] * adjRow[i];
    }

    for(int i = 0; i < cols; ++i) {
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
              const Tensor indices) {

  matchOrAbort<IndexType>(indices->type());

  size_t cols = in_->shape()[-1];
  size_t rows = indices->size();

  // note: may also be applied to IndexType; works by luck. Fix with fp16
  float* out = out_->data();
  const float* in = in_->data();

#pragma omp parallel for
  for(size_t j = 0; j < rows; ++j) {
    size_t dst = j;

    // @TODO: consider moving type checking to this function
    // instead of matchOrAbort above
    size_t src = (size_t)indices->data<IndexType>()[j];

    float* rowOut = out + dst * cols;
    const float* rowIn = in + src * cols;

    std::copy(rowIn, rowIn + cols, rowOut);
  }
}

void PasteRows(Tensor out_,
               const Tensor in_,
               const Tensor indices) {

  matchOrAbort<IndexType>(indices->type());

  size_t cols = in_->shape()[-1];
  size_t rows = indices->size();

  float* out = out_->data();
  const float* in = in_->data();

  for(size_t j = 0; j < rows; ++j) {
    size_t dst = indices->data<IndexType>()[j];  // not a permutation - may alias, unlike PasteCols
    size_t src = j;

    float* rowOut = out + dst * cols;
    const float* rowIn = in + src * cols;

    for(size_t i = 0; i < cols; ++i) {
      rowOut[i] += rowIn[i];
    }
  }
}

void CopyCols(Tensor out_,
              const Tensor in_,
              const Tensor indices) {

  matchOrAbort<IndexType>(indices->type());

  size_t rows = in_->shape().elements() / in_->shape()[-1];
  size_t colsIn = in_->shape()[-1];
  size_t colsOut = indices->size();

  float* out = out_->data();
  const float* in = in_->data();

#pragma omp parallel for
  for(size_t j = 0; j < rows; ++j) {
    const float* rowIn = in + j * colsIn;
    float* rowOut = out + j * colsOut;

    for(size_t i = 0; i < colsOut; ++i) {
      rowOut[i] = rowIn[indices->data<IndexType>()[i]];
    }
  }
}

void PasteCols(Tensor out_,
               const Tensor in_,
               const Tensor indices) {

  matchOrAbort<IndexType>(indices->type());

  size_t rows = out_->shape().elements() / out_->shape()[-1];
  size_t colsOut = out_->shape()[-1];
  size_t colsIn = indices->size();

  float* out = out_->data();
  const float* in = in_->data();

  /* n.b. Unlike PasteRows, currently appears safe to assume indices[i] is a
   *      permutation i.e. no racy aliases, and no need to sum vs. just assign.
   */
  for(size_t j = 0; j < rows; ++j) {
    const float* rowIn = in + j * colsIn;
    float* rowOut = out + j * colsOut;

    for(size_t i = 0; i < colsIn; ++i) {
      rowOut[indices->data<IndexType>()[i]] += rowIn[i];
    }
  }
}

#if 0 // this version seems to actually be buggy, but also not used in decoding?
// Optimized version of Select for axis=2
// @TODO: make this generally fast without this special version
void SelectAxis2(Tensor out,
             const Tensor in,
             const Tensor indices) {

  std::cerr << indices->debug() << std::endl;

  matchOrAbort<IndexType>(indices->type());

  functional::Shape outShape = out->shape();
  functional::Shape inShape = in->shape();

  auto idxData = indices->data<IndexType>();
  auto odata = out->data();
  const auto idata = in->data();

  int size = outShape[3];

  for(int k = 0; k < outShape[0]; ++k) {
    for(int j = 0; j < outShape[1]; ++j) {
      int outOffset = k * j * outShape[2] * size + j * outShape[2] * size;
      int inOffset = k * j * inShape[2] * size + j * inShape[2] * size;
      for(int i = 0; i < outShape[2]; ++i) {
        auto idx = idxData[i];
        int outIndex = outOffset +   i * size;
        int inIndex  = inOffset  + idx * size;
        std::copy(idata + inIndex, idata + inIndex + size, odata + outIndex);
      }
    }
  }
}
#endif

void Select(Tensor out,
            const Tensor in,
            const Tensor indices,
            int axis) {

  matchOrAbort<IndexType>(indices->type());

  // @TODO: make this efficient
  functional::Shape outShape = out->shape();
  functional::Shape inShape  = in->shape();
  functional::Shape idxShape = indices->shape();
  int length = outShape.elements();

  functional::Array<int, functional::Shape::size()> dims;
  int axisCPU = (int)(axis + functional::Shape::size() - out->shape().size());

#if 0 // buggy but not really used?
  if(axisCPU == 2 && outShape == idxShape) // specialization for axis==2 when there is no broadcasting, @TODO to be removed once we have a faster implementation below
    return SelectAxis2(out, in, indices);
#endif

  for(int index = 0; index < length; ++index) {
    outShape.dims(index, dims);                                // compute dimension-based indices from global index;
    int idxIndex = idxShape.bindex(dims);                      // return global index for indices based on dimension-specific indices from out, take broadcasting into account;
    dims[axisCPU] = (int)indices->data<IndexType>()[idxIndex]; // substitute index of out-tensor with corresponding axis-local position from in-tensor;
    int inIndex = inShape.index(dims);                         // compute global index from dimension-specific indices, no broadcasting as out and in match in all dimensions apart from axis
    out->data()[index] = in->data()[inIndex];                  // assign corresponding values.
  }
}

template <bool add>
void Insert(Tensor out,
            const Tensor in,
            const Tensor indices,
            int axis) {

  matchOrAbort<IndexType>(indices->type());

  // @TODO: make this efficient
  functional::Shape outShape = out->shape();
  functional::Shape inShape  = in->shape();
  functional::Shape idxShape = indices->shape();

  int length = inShape.elements();
  functional::Array<int, functional::Shape::size()> dims;
  int axisCPU = (int)(axis + functional::Shape::size() - out->shape().size());

  for(int index = 0; index < length; ++index) {
    inShape.dims(index, dims);
    int idxIndex = idxShape.bindex(dims); // broadcast index into indices tensor
    dims[axisCPU] = (int)indices->data<IndexType>()[idxIndex];
    int outIndex = outShape.index(dims);
    if(add)
      out->data()[outIndex] += in->data()[index];
    else
      out->data()[outIndex] = in->data()[index];
  }
}

template void Insert<true>(Tensor out, const Tensor in, const Tensor indices, int axis);
template void Insert<false>(Tensor out, const Tensor in, const Tensor indices, int axis);

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
      float r = functional::Ops<float>::sigmoid(xWrow[i] + sUrow[i] + b[i]);

      int k = i + cols;

      float z = functional::Ops<float>::sigmoid(xWrow[k] + sUrow[k] + b[k]);

      int l = i + 2 * cols;
      float h;
      if(final)
        h = std::tanh(xWrow[l] + (sUrow[l] + b[l]) * r);
      else
        h = std::tanh(xWrow[l] + sUrow[l] * r + b[l]);

      float o = (1.0f - z) * h + z * rowState[i];
      rowOut[i] = m * o + (1 - m) * rowState[i];
    }
  }
}

void GRUFastBackward(Ptr<Allocator> /*allocator*/,
                     std::vector<Tensor> outputs,
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

      float r = functional::Ops<float>::sigmoid(rowXW[i] + rowSU[i] + b[i]);
      float z = functional::Ops<float>::sigmoid(rowXW[k] + rowSU[k] + b[k]);

      float h;
      if(final)
        h = std::tanh(rowXW[l] + (rowSU[l] + b[l]) * r);
      else
        h = std::tanh(rowXW[l] + rowSU[l] * r + b[l]);

      float a = rowAdj[i];

      float t = (1 - z) * (1 - h * h);

      // df/ds
      if(outState)
        rowOutState[i] += (m * z - m + 1) * a;

      // df/d(xW_r) ...
      float dfdxW_r = m * r * (1 - r) * t * a;
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
      float dfdxW_z = m * (1 - z) * z * (rowState[i] - h) * a;
      if(outXW)
        rowOutXW[k] += dfdxW_z;
      if(outSU)
        rowOutSU[k] += dfdxW_z;
      if(outB)
        outB[k] += dfdxW_z;

      // df/d(xW_x) ...
      float dfdxW_x = m * t * a;
      if(outXW)
        rowOutXW[l] += dfdxW_x;
      if(outSU)
        rowOutSU[l] += dfdxW_x * r;
      if(outB) {
        if(final)
          outB[l] += dfdxW_x * r;
        else
          outB[l] += dfdxW_x;
      }
    }
  }
}

void CrossEntropyPick(Tensor out, Tensor in, Tensor labelIndices, float labelSmoothingAlpha = 0.f) {
  matchOrAbort<IndexType>(labelIndices->type());

  // Shape& outShape = out_->shape();
  Shape& inShape = in->shape();

  int rows = inShape.elements() / inShape.back();
  int cols = inShape.back();

  #pragma omp parallel for
  for(int j = 0; j < rows; ++j) {
    const float* sp = in->data() + j * cols;
    float max = sp[0];
    #pragma omp simd reduction(max : max)
    for(int i = 1; i < cols; ++i) {
      max = std::max(max, sp[i]);
    }

    float sumexp = 0.f;
    #pragma omp simd reduction(+ : sumexp)
    for(int i = 0; i < cols; ++i) {
      sumexp += std::exp(sp[i] - max);
    }

    float mean = 0.f;
    #pragma omp simd reduction(+ : mean)
    for(int i = 0; i < cols; ++i) {
      mean += sp[i] - max;
    }
    mean /= (float)cols;

    // Groundtruth label index
    IndexType i = labelIndices->data<IndexType>()[j];
    // This appears to be safe i.e. that i >= 0 && i < cols is known
    float logsumexp = std::log(sumexp);
    float ce = logsumexp - sp[i] + max; // -log(p_i) = - logsoftmax(x_i - max) = - (x_i - max) - log(sum_j exp(x_j - max))
    float ls = logsumexp - mean; 
    out->data()[j] = (1.f - labelSmoothingAlpha) * ce + labelSmoothingAlpha * ls;
  }
}

void CrossEntropyPickBackward(Tensor out,
                              Tensor adj,
                              Tensor in,
                              Tensor labelIndices,
                              float labelSmoothingAlpha = 0.f) {

  matchOrAbort<IndexType>(labelIndices->type());
  Shape& outShape = out->shape();

  int rows = outShape.elements() / outShape.back();
  int cols = outShape.back();

#pragma omp parallel for
  for(int j = 0; j < rows; ++j) {
    const float* sp = in->data() + j * cols;
    float* so = out->data() + j * cols;

    float max = sp[0];
    for(int i = 1; i < cols; ++i) {
      max = std::max(max, sp[i]);
    }

    float sumexp = 0.f;
    for(int i = 0; i < cols; ++i) {
      sumexp += std::exp(sp[i] - max);
    }

    // cross-entropy
    for(int i = 0; i < cols; ++i) {
      float sub = (float)(i == (int)labelIndices->data<IndexType>()[j]); // delta, true if label index and column index match
      float dce = std::exp(sp[i] - max) / sumexp - sub 
                + labelSmoothingAlpha * (sub - 1.f / (float)cols);
      so[i] += adj->data()[j] * dce;
    }
  }
}

float L2Norm(Tensor in, Ptr<Allocator> /*not used*/) {
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
  for(int j = 0; j < rows; ++j) {
    const float* vaRow = va;
    const float* ctxRow = ctx + (j % (b * t)) * cols;
    const float* stateRow = state + ((j / (b * t)) * b + j % b) * cols;

    float sum = 0.f;
    #pragma omp simd reduction(+ : sum)
    for(int i = 0; i < cols; ++i) {
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

  #pragma omp parallel for reduction(+ : gState[:n * k], gVa[:k])
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

MARIAN_FFAST_MATH_BEGIN
template <int alphaStride, int betaStride, bool hasBeta>
void LayerNormalizationImpl(float* out,
                            const float* in,
                            const float* alpha,
                            const float* beta,
                            float eps,
                            int rows,
                            int cols) {
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

    float sigma = std::sqrt(sqSum / cols + eps);

    #pragma omp simd
    for(int i = 0; i < cols; ++i) {
      float t = alpha[alphaStride * i] * ((sp[i] - mean) / sigma);
      if(hasBeta)
        t += beta[betaStride * i];

      so[i] = t;
    }
  }
}
MARIAN_FFAST_MATH_END

template <int alphaStride>
inline void LayerNormalizationDispatchBeta(float* out,
                                           const float* in,
                                           const float* alpha,
                                           Tensor beta,
                                           float eps,
                                           int rows,
                                           int cols) {
  if (beta) {
    if (beta->shape().back() > 1) {
      LayerNormalizationImpl<alphaStride, 1, true>(out, in, alpha, beta->data(), eps, rows, cols);
    } else {
      LayerNormalizationImpl<alphaStride, 0, true>(out, in, alpha, beta->data(), eps, rows, cols);
    }
  } else {
    LayerNormalizationImpl<alphaStride, 0, false>(out, in, alpha, nullptr, eps, rows, cols);
  }
}

void LayerNormalization(Tensor out_,
                        Tensor in_,
                        Tensor gamma_,
                        Tensor beta,
                        float eps) {
  float* out = out_->data();
  const float* in = in_->data();
  const float* alpha = gamma_->data();
  const int alphaStride = gamma_->shape().back() > 1;  // broadcasting for alpha and beta

  int rows = in_->shape().elements() / in_->shape().back();
  int cols = in_->shape().back();
  if (alphaStride == 0) {
    LayerNormalizationDispatchBeta<0>(out, in, alpha, beta, eps, rows, cols);
  } else {
    LayerNormalizationDispatchBeta<1>(out, in, alpha, beta, eps, rows, cols);
  }
}

MARIAN_FFAST_MATH_BEGIN
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
  // @TODO: The CPU implementation supports scalar gamma and beta. This is a left-over,
  //        we should enable that in the GPU version as well.
  const int gammaStride = gamma_->shape().back() > 1;  // broadcasting for alpha and beta. 0 means it's a scalar
  const int betaStride = beta_ && beta_->shape().back() > 1;

  size_t rows = y_->shape().elements() / y_->shape()[-1];
  size_t cols = y_->shape()[-1];

  if(beta) {
    #pragma omp parallel for reduction(+ : gradGamma[:cols], gradBeta[:cols])
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
        sum_adj_x += adjRow[i] * (yRow[i] - (beta ? beta[betaStride * i] : 0.f)) / gamma[gammaStride * i];
        sum_adj += adjRow[i];
      }

      float mean = sum_x / cols;
      #pragma omp simd reduction(+ : sum_sqr)
      for(size_t i = 0; i < cols; ++i) {
        float ex = xRow[i] - mean;
        sum_sqr += ex * ex;
      }

      float sigma = std::sqrt(sum_sqr / cols + eps);
      #pragma omp simd
      for(size_t i = 0; i < cols; ++i) {
        float grad_x = 0.f;
        float x_hat = (yRow[i] - beta[betaStride * i]) / gamma[gammaStride * i];
        grad_x += cols * adjRow[i];
        grad_x -= sum_adj;
        grad_x -= sum_adj_x * x_hat;
        grad_x /= cols * sigma;

        gradXRow[i] += gamma[gammaStride * i] * grad_x;
        gradGamma[gammaStride * i] += adjRow[i] * x_hat;
        gradBeta[betaStride * i] += adjRow[i];
      }
    }
  } else { // @TODO: this code duplication is really ugly, but required for omp to work correctly?
    #pragma omp parallel for reduction(+ : gradGamma[:cols])
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
        sum_adj_x += adjRow[i] * yRow[i] / gamma[gammaStride * i];
        sum_adj += adjRow[i];
      }

      float mean = sum_x / cols;
      #pragma omp simd reduction(+ : sum_sqr)
      for(size_t i = 0; i < cols; ++i) {
        float ex = xRow[i] - mean;
        sum_sqr += ex * ex;
      }

      float sigma = std::sqrt(sum_sqr / cols + eps);
      #pragma omp simd
      for(size_t i = 0; i < cols; ++i) {
        float grad_x = 0.f;
        float x_hat = yRow[i] / gamma[gammaStride * i];
        grad_x += cols * adjRow[i];
        grad_x -= sum_adj;
        grad_x -= sum_adj_x * x_hat;
        grad_x /= cols * sigma;

        gradXRow[i] += gamma[gammaStride * i] * grad_x;
        gradGamma[gammaStride * i] += adjRow[i] * x_hat;
      }
    }
  }
}
MARIAN_FFAST_MATH_END

MARIAN_FFAST_MATH_BEGIN
template <int alphaStride, int betaStride, bool hasBeta>
void RMSNormalizationImpl(float* out,
                          const float* in,
                          const float* alpha,
                          const float* beta,
                          float eps,
                          int rows,
                          int cols) {
  #pragma omp parallel for
  for(int j = 0; j < rows; ++j) {
    float* so = out + j * cols;
    const float* sp = in + j * cols;

    float sqSum = 0.f;
    #pragma omp simd reduction(+ : sqSum)
    for(int i = 0; i < cols; ++i) {
      sqSum += sp[i] * sp[i];
    }

    float rms = std::sqrt(sqSum / cols + eps);

    #pragma omp simd
    for(int i = 0; i < cols; ++i) {
      float t = alpha[alphaStride * i] * (sp[i] / rms);
      if(hasBeta)
        t += beta[betaStride * i];

      so[i] = t;
    }
  }
}
MARIAN_FFAST_MATH_END

template <int alphaStride>
inline void RMSNormalizationDispatchBeta(float* out,
                                           const float* in,
                                           const float* alpha,
                                           Tensor beta,
                                           float eps,
                                           int rows,
                                           int cols) {
  if (beta) {
    if (beta->shape().back() > 1) {
      RMSNormalizationImpl<alphaStride, 1, true>(out, in, alpha, beta->data(), eps, rows, cols);
    } else {
      RMSNormalizationImpl<alphaStride, 0, true>(out, in, alpha, beta->data(), eps, rows, cols);
    }
  } else {
    RMSNormalizationImpl<alphaStride, 0, false>(out, in, alpha, nullptr, eps, rows, cols);
  }
}

void RMSNormalization(Tensor out,
                      Tensor in,
                      Tensor gamma,
                      Tensor beta,
                      float eps) {
  const float* alpha = gamma->data();
  const int alphaStride = gamma->shape().back() > 1;  // broadcasting for alpha and beta

  int rows = in->shape().elements() / in->shape().back();
  int cols = in->shape().back();
  if (alphaStride == 0) {
    RMSNormalizationDispatchBeta<0>(out->data(), in->data(), alpha, beta, eps, rows, cols);
  } else {
    RMSNormalizationDispatchBeta<1>(out->data(), in->data(), alpha, beta, eps, rows, cols);
  }
}

MARIAN_FFAST_MATH_BEGIN
void RMSNormalizationGrad(Tensor gradX_,
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
  float* x = x_->data();
  float* y = y_->data();
  float* gamma = gamma_->data();
  float* beta = beta_ ? beta_->data() : nullptr;
  // @TODO: The CPU implementation supports scalar gamma and beta. This is a left-over,
  //        we should enable that in the GPU version as well.
  const int gammaStride = gamma_->shape().back() > 1;  // broadcasting for alpha and beta. 0 means it's a scalar
  const int betaStride = beta_ && beta_->shape().back() > 1;

  size_t rows = y_->shape().elements() / y_->shape()[-1];
  size_t cols = y_->shape()[-1];

  if(beta) {
    #pragma omp parallel for reduction(+ : gradGamma[:cols], gradBeta[:cols])
    for(size_t j = 0; j < rows; ++j) {
      const float* xRow = x + j * cols;
      const float* yRow = y + j * cols;
      const float* adjRow = adj + j * cols;
      float* gradXRow = gradX + j * cols;

      float sum_adj_r = 0.f;
      float sum_sqr = 0.f;

      #pragma omp simd reduction(+ : sum_adj_r, sum_sqr)
      for(size_t i = 0; i < cols; ++i) {
        sum_adj_r += adjRow[i] * (yRow[i] - beta[betaStride * i]) / gamma[gammaStride * i];
        sum_sqr   += xRow[i] * xRow[i];
      }

      float rms = std::sqrt(sum_sqr / cols + eps);
      #pragma omp simd
      for(size_t i = 0; i < cols; ++i) {
        float rmsNorm  = (yRow[i] - beta[betaStride * i]) / gamma[gammaStride * i];
        float gradNorm = cols * adjRow[i] - rmsNorm * sum_adj_r;
        gradNorm      /= cols * rms; 

        gradXRow[i]                += gamma[gammaStride * i] * gradNorm;
        gradGamma[gammaStride * i] += adjRow[i] * rmsNorm;
        gradBeta[betaStride * i]   += adjRow[i];
      }
    }
  } else {
    #pragma omp parallel for reduction(+ : gradGamma[:cols])
    for(size_t j = 0; j < rows; ++j) {
      const float* xRow = x + j * cols;
      const float* yRow = y + j * cols;
      const float* adjRow = adj + j * cols;
      float* gradXRow = gradX + j * cols;

      float sum_adj_r = 0.f;
      float sum_sqr = 0.f;

      #pragma omp simd reduction(+ : sum_adj_r, sum_sqr)
      for(size_t i = 0; i < cols; ++i) {
        sum_adj_r += yRow[i] / gamma[gammaStride * i];
        sum_sqr += xRow[i] * xRow[i];
      }

      float rms = std::sqrt(sum_sqr / cols + eps);
      #pragma omp simd
      for(size_t i = 0; i < cols; ++i) {
        float rmsNorm  = yRow[i] / gamma[gammaStride * i];
        float gradNorm = cols * adjRow[i] - rmsNorm * sum_adj_r;
        gradNorm      /= cols * rms; 

        gradXRow[i]                += gamma[gammaStride * i] * gradNorm;
        gradGamma[gammaStride * i] += adjRow[i] * rmsNorm;
      }
    }
  }
}
MARIAN_FFAST_MATH_END

void Shift(Tensor out_,
           Tensor in_,
           marian::Shape shift,
           float padValue,
           bool invert) {
  int offset = 0; // out[i + offset] = in[i]; shift>0 inserts values at front, shifts back, pushes out
  for(int i = 0; i < shift.size(); ++i)
    offset += in_->shape().stride(i) * shift[i];

  if(invert)
    offset = -offset;

  float* out = out_->data();
  const float* in = in_->data();

  int length = out_->shape().elements();
#pragma omp parallel for
  for(int i = 0; i < length; ++i) {
    // BUGBUG: This logic is only correct for the outermost axis.
    if(i - offset < 0 || i - offset >= length) {
      out[i] = padValue;
    } else {
      out[i] = in[i - offset];
    }
  }
}

void ShiftGrad(Tensor out_, Tensor in_, marian::Shape shift, bool invert) {
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
    if(i - offset >= 0 && i - offset < length) {
      out[i] += in[i - offset];
    }
  }
}

void SetSparse(float* out,
               const std::vector<size_t>& indices,
               const std::vector<float>& values) {
  int length = (int)indices.size();
  for(int index = 0; index < length; ++index) {
    out[indices[index]] = values[index];
  }
}

// should be implemented via slicing and elementwise
template <typename FType>
void LSTMCellForwardTyped(Tensor out_, const std::vector<Tensor>& inputs) {
  int rows = out_->shape().elements() / out_->shape()[-1];

  int fVecSize = sizeof(FType) / sizeof(float);
  int cols = out_->shape()[-1] / fVecSize;

  FType* out = out_->data<FType>();
  const FType* cell = inputs[0]->data<FType>();
  const FType* xW = inputs[1]->data<FType>();
  const FType* sU = inputs[2]->data<FType>();
  const FType* b = inputs[3]->data<FType>();
  const float* mask = inputs.size() > 4 ? inputs[4]->data() : nullptr;

  using fop = functional::Ops<FType>;

  for(int j = 0; j < rows; ++j) {
    float m = !mask || mask[j];

    FType* rowOut = out + j * cols;
    const FType* rowCell = cell + j * cols;

    const FType* xWrow = xW + j * cols * 4;
    const FType* sUrow = sU + j * cols * 4;

    for(int i = 0; i < cols; ++i) {
      FType gf   = fop::sigmoid(fop::add(fop::add(xWrow[i], sUrow[i]), b[i]));

      int k = i + cols;
      FType gi   = fop::sigmoid(fop::add(fop::add(xWrow[k], sUrow[k]), b[k]));

      int l = i + 2 * cols;
      FType gc   = fop::tanh(fop::add(fop::add(xWrow[l], sUrow[l]), b[l]));

      FType cout = fop::add(fop::mul(gf, rowCell[i]), fop::mul(gi, gc));
      rowOut[i]  = fop::add(fop::mul(m, cout), fop::mul(fop::sub(1.f, m), rowCell[i]));
    }
  }
}

void LSTMCellForward(Tensor out, std::vector<Tensor> inputs) {
  int cols = out->shape()[-1];
#ifdef __AVX__
  if(cols % 8 == 0)
    LSTMCellForwardTyped<float32x8>(out, inputs);
  else
#endif
  if(cols % 4 == 0)
    LSTMCellForwardTyped<float32x4>(out, inputs);
  else
    LSTMCellForwardTyped<float>(out, inputs);
}

template <typename FType>
void LSTMOutputForwardTyped(Tensor out_, const std::vector<Tensor>& inputs) {
  int rows = out_->shape().elements() / out_->shape()[-1];
  
  int fVecSize = sizeof(FType) / sizeof(float);
  int cols = out_->shape()[-1] / fVecSize;

  FType* out = out_->data<FType>();
  const FType* cell = inputs[0]->data<FType>();
  const FType* xW   = inputs[1]->data<FType>();
  const FType* sU   = inputs[2]->data<FType>();
  const FType* b    = inputs[3]->data<FType>();

  using fop = functional::Ops<FType>;

  for(int j = 0; j < rows; ++j) {
    FType* rowOut = out + j * cols;
    const FType* rowCell = cell + j * cols;

    const FType* xWrow = xW + j * cols * 4;
    const FType* sUrow = sU + j * cols * 4;

    for(int i = 0; i < cols; ++i) {
      int k = i + 3 * cols;
      FType go  = fop::sigmoid(fop::add(fop::add(xWrow[k], sUrow[k]), b[k]));
      rowOut[i] = fop::mul(go, fop::tanh(rowCell[i]));
    }
  }
}

void LSTMOutputForward(Tensor out, std::vector<Tensor> inputs) {
  int cols = out->shape()[-1];

#ifdef __AVX__
  if(cols % 8 == 0)
    LSTMOutputForwardTyped<float32x8>(out, inputs);
  else 
#endif
  if(cols % 4 == 0)
    LSTMOutputForwardTyped<float32x4>(out, inputs);
  else
    LSTMOutputForwardTyped<float>(out, inputs);
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
      float gf = functional::Ops<float>::sigmoid(xWrow[i] + sUrow[i] + b[i]);

      int k = i + cols;
      float gi = functional::Ops<float>::sigmoid(xWrow[k] + sUrow[k] + b[k]);

      int l = i + 2 * cols;
      float gc = std::tanh(xWrow[l] + sUrow[l] + b[l]);

      float a = rowAdj[i];

      // dc/dx_{t-1}
      if(outCell) {
        rowOutCell[i] += (m * gf - m + 1) * a;
      }

      // dc/d(b_f) = dc/d(xW_f) ...
      float dcdxf = m * rowCell[i] * gf * (1 - gf) * a;
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
      float dcdb_i = m * gc * gi * (1 - gi) * a;
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
      float dcdxc = m * gi * (1 - gc * gc) * a;
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
      float go = functional::Ops<float>::sigmoid(xWrow[k] + sUrow[k] + b[k]);

      float t = std::tanh(rowCell[i]);

      float a = rowAdj[i];

      // dc/dc_{t-1}
      if(outCell) {
        rowOutCell[i] += go * (1 - t * t) * a;
      }

      // dc/d(b_o) = dc/d(xW_f) ...
      float dcdxo = t * go * (1 - go) * a;
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

void SinusoidalPositionEmbeddings(marian::Tensor t, int start) {
  int dimEmb   = t->shape()[-1];
  int dimWords = (int)t->size() / dimEmb;

  float numTimescales = (float)dimEmb / 2;
  float logTimescaleIncrement = std::log(10000.f) / (numTimescales - 1.f);

  for(int j = 0; j < dimWords; ++j) {
    for(int i = 0; i < dimEmb; ++i) {
      float v = (j + start) * std::exp((i % (int)numTimescales) * -logTimescaleIncrement);
      t->data()[j * dimEmb + i] = i < (int)numTimescales ? std::sin(v) : std::cos(v);
    }
  }
}

void HighwayForward(Tensor out,
                   const Tensor in1,
                   const Tensor in2,
                   const Tensor t) {
  using namespace functional;
  cpu::Element(_1 = sigmoid(_2), out, t);
  cpu::Element(_1 = _1 * _2 + (1.f - _1) * _3, out, in1, in2);
}

void HighwayBackward(Tensor out1,
                     Tensor out2,
                     Tensor outt,
                     const Tensor in1,
                     const Tensor in2,
                     const Tensor t,
                     const Tensor adj) {
  using namespace functional;
  cpu::Element(_1 +=        sigmoid(_2)  * _3, out1, t, adj);
  cpu::Element(_1 += (1.f - sigmoid(_2)) * _3, out2, t, adj);
  cpu::Element(_1 += sigmoid(_2) * (1.f - sigmoid(_2)) * (_3 - _4) * _5, outt, t, in1, in2, adj);
}

void PoolingWithMaskingForward(Tensor /*out*/,
                               Tensor /*in*/,
                               Tensor /*mask*/,
                               int /*width*/,
                               bool /*isEven*/) {
  ABORT("Not implemented!");
}

void PoolingWithMaskingBackward(Tensor /*adj*/,
                                Tensor /*adjIn*/,
                                Tensor /*in*/,
                                Tensor /*mask*/,
                                int /*width*/,
                                bool /*isEven*/) {
  ABORT("Not implemented!");
}
}  // namespace cpu
}  // namespace marian
