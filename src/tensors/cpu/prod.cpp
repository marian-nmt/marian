/* All or part of this file was contributed by Intel under license:
 *   Copyright (C) 2017-2018 Intel Corporation
 *   SPDX-License-Identifier: MIT
 */

#include "tensors/cpu/backend.h"
#include "tensors/tensor.h"
#include "tensors/tensor_allocator.h"

#if MKL_FOUND
#include <mkl.h>
#else
#if BLAS_FOUND
#include <cblas.h>
#endif
#endif

#include "integer_common.h"
#include "prod_blas.h"


namespace marian {

namespace cpu {

void Prod(marian::Tensor C,
          const marian::Tensor& A,
          const marian::Tensor& B,
          bool transA,
          bool transB,
          float beta,
          float scalar) {
#if BLAS_FOUND
  float alpha = scalar;

  int m = A->shape().elements() / A->shape()[-1];
  int k = A->shape().back();
  if(transA)
    std::swap(m, k);

  int l = B->shape().elements() / B->shape()[-1];
  int n = B->shape()[-1];
  if(transB)
    std::swap(l, n);

  int lda = A->shape()[-1];
  int ldb = B->shape()[-1];
  int ldc = B->shape()[-1];

  if(transB)
    ldc = B->shape().elements() / B->shape()[-1];

  sgemm(transA,
        transB,
        m,
        n,
        k,
        alpha,
        A->data(),
        lda,
        B->data(),
        ldb,
        beta,
        C->data(),
        ldc);
#else
  C; A; B; transA; transB; beta; scalar;
  ABORT("You need to compile with MKL in order to use the CPU version");
#endif
}

// dummy implementation, computeType doesn't do anything on CPU
void Prod(marian::Tensor C,
          const marian::Tensor& A,
          const marian::Tensor& B,
          bool transA,
          bool transB,
          float beta,
          float scalar,
          Type computeType) {
  computeType; // make compiler happy
  cpu::Prod(C, A, B, transA, transB, beta, scalar);
}

void ProdBatched(marian::Tensor C,
                 Ptr<Allocator> /*allocator*/,
                 const marian::Tensor A,
                 const marian::Tensor B,
                 bool transA,
                 bool transB,
                 float beta,
                 float scalar) {
#if BLAS_FOUND
  float alpha = scalar;

  // determine meta-shape of bdot operation. Essentially treat the last two dimensions as single elements
  // such that (..., m, k) x (..., k, n) -> (..., m, n) where ... is a broadcastable shape as in element-wise kernels.

  auto aShape = A->shape();
  auto bShape = B->shape();

  // make sure both shape have the same number of dimensions via broadcasting
  size_t maxLength = std::max(aShape.size(), bShape.size());
  if(aShape.size() != bShape.size()) {
    Shape ones(std::vector<int>(maxLength, 1));
    aShape = Shape::broadcast({aShape, ones});
    bShape = Shape::broadcast({bShape, ones});
  }

  // Create meta-shapes without last 2 dimensions
  Shape aShapeMeta, bShapeMeta, cShapeMeta;
  aShapeMeta.resize(maxLength - 2);
  bShapeMeta.resize(maxLength - 2);
  for(size_t i = 0; i < maxLength - 2; ++i) {
    aShapeMeta.set(i, aShape[i]);
    bShapeMeta.set(i, bShape[i]);
  }
  cShapeMeta = Shape::broadcast({aShapeMeta, bShapeMeta});

  int m = aShape[-2];
  int k = aShape[-1];
  if(transA)
    std::swap(m, k);

  int l = bShape[-2];
  int n = bShape[-1];
  if(transB)
    std::swap(l, n);

  int lda = aShape[-1];
  int ldb = bShape[-1];
  int ldc = bShape[-1];

  if(transB)
    ldc = bShape[-2];

  int strideA = m * k;
  int strideB = n * k;
  int strideC = n * m;

  int batchC = cShapeMeta.elements();

  // Convert to functional shapes to be able to map dimensions. @TODO merge this
  functional::Shape aShapeMetaF = aShapeMeta;
  functional::Shape bShapeMetaF = bShapeMeta;
  functional::Shape cShapeMetaF = cShapeMeta;

#if MKL_FOUND
  CBLAS_TRANSPOSE transA_forarr = CblasNoTrans;
  CBLAS_TRANSPOSE transB_forarr = CblasNoTrans;

  if(transA)
    transA_forarr = CblasTrans;

  if(transB)
    transB_forarr = CblasTrans;

  /* cblas_sgemm_batch allows us to group all the small GEMMs that are done in a for loop with sgemm and compute
   * them in only one MKL call. For the API documentation refer to
   * https://software.intel.com/content/www/us/en/develop/documentation/mkl-developer-reference-c/top/blas-and-sparse-blas-routines/blas-like-extensions/cblas-gemm-batch.html
   * The API supports dependencies, where you can specify one "group" of GEMMs to be computed after another. (This controlled by the group_count parameter).
   * In our case, the operations are not dependent on one another so we hardcode one group. The rest of the arguments (with the exception of group_size) are
   * the same as the ones that cblas_sgemm expects, with the difference that we are supposed to provide an array pointer (One element per group).
   * Weirdly enough, we are required to to provide all of the integer arguments as the MKL_INT datatype
   */

  static const constexpr size_t group_count = 1; // We have one group
  const std::vector<CBLAS_TRANSPOSE> transa_arr(group_count, transA_forarr);
  const std::vector<CBLAS_TRANSPOSE> transb_arr(group_count, transB_forarr);
  const std::vector<MKL_INT> m_arr(group_count, (MKL_INT)m);
  const std::vector<MKL_INT> n_arr(group_count, (MKL_INT)n);
  const std::vector<MKL_INT> k_arr(group_count, (MKL_INT)k);
  const std::vector<float> alpha_arr(group_count, alpha);
  const std::vector<float> beta_arr(group_count, beta);
  const std::vector<MKL_INT> lda_arr(group_count, (MKL_INT)lda);
  const std::vector<MKL_INT> ldb_arr(group_count, (MKL_INT)ldb);
  const std::vector<MKL_INT> ldc_arr(group_count, (MKL_INT)ldc);
  const std::vector<MKL_INT> group_size(group_count, (MKL_INT)batchC); // Group size specifies number of GEMM operations per group (Which is batchC)

  std::vector<const float *> a_array(batchC, nullptr);
  std::vector<const float *> b_array(batchC, nullptr);
  std::vector<float *> c_array(batchC, nullptr);

  // This loop initializes the array pointers in the same way as the for loop
  // in the normal sgemm version a few lines below
  functional::Array<int, functional::Shape::size()> dims;
  for(int i = 0; i < batchC; ++i) {
    cShapeMetaF.dims(i, dims);
    auto aIndex = aShapeMetaF.bindex(dims);
    auto bIndex = bShapeMetaF.bindex(dims);

    a_array[i] = A->data() + aIndex * strideA;
    b_array[i] = B->data() + bIndex * strideB;
    c_array[i] = C->data() + i * strideC;
  }
  cblas_sgemm_batch (CblasRowMajor,
    &transa_arr[0],
    &transb_arr[0],
    &m_arr[0],
    &n_arr[0],
    &k_arr[0],
    &alpha_arr[0],
    &a_array[0],
    &lda_arr[0],
    &b_array[0],
    &ldb_arr[0],
    &beta_arr[0],
    &c_array[0],
    &ldc_arr[0],
    group_count,
    &group_size[0]);
#else
  functional::Array<int, functional::Shape::size()> dims;
  for(size_t i = 0; i < batchC; ++i) {
    cShapeMetaF.dims(i, dims);
    auto aIndex = aShapeMetaF.bindex(dims);
    auto bIndex = bShapeMetaF.bindex(dims);

    sgemm(transA,
          transB,
          m,
          n,
          k,
          alpha,
          A->data() + aIndex * strideA,
          lda,
          B->data() + bIndex * strideB,
          ldb,
          beta,
          C->data() + i * strideC,
          ldc);
  }
#endif
#else
  C; A; B; transA; transB; beta; scalar;
  ABORT("You need to compile with MKL in order to use the CPU version");
#endif
}


void ProdBatchedLegacy(marian::Tensor C,
                       Ptr<Allocator> /*allocator*/,
                       const marian::Tensor A,
                       const marian::Tensor B,
                       bool transA,
                       bool transB,
                       float beta,
                       float scalar) {
#if BLAS_FOUND
  float alpha = scalar;

  size_t batchA = A->shape().elements() / (A->shape()[-1] * A->shape()[-2]);
  size_t batchB = B->shape().elements() / (B->shape()[-1] * B->shape()[-2]);

  size_t m = A->shape()[-2];
  size_t k = A->shape()[-1];
  if(transA)
    std::swap(m, k);

  size_t l = B->shape()[-2];
  size_t n = B->shape()[-1];
  if(transB)
    std::swap(l, n);

  size_t lda = A->shape()[-1];
  size_t ldb = B->shape()[-1];
  size_t ldc = B->shape()[-1];

  if(transB)
    ldc = B->shape()[-2];

  auto strideB = batchB == 1 ? 0 : n * k;
  auto strideA = batchA == 1 ? 0 : m * k;
  auto strideC = n * m;

  auto batchC = std::max(batchA, batchB);
#if MKL_FOUND
  CBLAS_TRANSPOSE transA_forarr = CblasNoTrans;
  CBLAS_TRANSPOSE transB_forarr = CblasNoTrans;

  if(transA)
    transA_forarr = CblasTrans;

  if(transB)
    transB_forarr = CblasTrans;

  /* cblas_sgemm_batch allows us to group all the small GEMMs that are done in a for loop with sgemm and compute
   * them in only one MKL call. For the API documentation refer to
   * https://software.intel.com/content/www/us/en/develop/documentation/mkl-developer-reference-c/top/blas-and-sparse-blas-routines/blas-like-extensions/cblas-gemm-batch.html
   * The API supports dependencies, where you can specify one "group" of GEMMs to be computed after another. (This controlled by the group_count parameter).
   * In our case, the operations are not dependent on one another so we hardcode one group. The rest of the arguments (with the exception of group_size) are
   * the same as the ones that cblas_sgemm expects, with the difference that we are supposed to provide an array pointer (One element per group).
   * Weirdly enough, we are required to to provide all of the integer arguments as the MKL_INT datatype
   */

  static const constexpr size_t group_count = 1; // We have one group
  const std::vector<CBLAS_TRANSPOSE> transa_arr(group_count, transA_forarr);
  const std::vector<CBLAS_TRANSPOSE> transb_arr(group_count, transB_forarr);
  const std::vector<MKL_INT> m_arr(group_count, (MKL_INT)m);
  const std::vector<MKL_INT> n_arr(group_count, (MKL_INT)n);
  const std::vector<MKL_INT> k_arr(group_count, (MKL_INT)k);
  const std::vector<float> alpha_arr(group_count, alpha);
  const std::vector<float> beta_arr(group_count, beta);
  const std::vector<MKL_INT> lda_arr(group_count, (MKL_INT)lda);
  const std::vector<MKL_INT> ldb_arr(group_count, (MKL_INT)ldb);
  const std::vector<MKL_INT> ldc_arr(group_count, (MKL_INT)ldc);
  const std::vector<MKL_INT> group_size(group_count, (MKL_INT)batchC); // Group size specifies number of GEMM operations per group (Which is batchC)

  std::vector<const float *> a_array(batchC, nullptr);
  std::vector<const float *> b_array(batchC, nullptr);
  std::vector<float *> c_array(batchC, nullptr);

  // This loop initializes the array pointers in the same way as the for loop
  // in the normal sgemm version a few lines below
  for(size_t i = 0; i < batchC; ++i) {
    a_array[i] = A->data() + (i % batchA) * strideA;
    b_array[i] = B->data() + (i % batchB) * strideB;
    c_array[i] = C->data() + i * strideC;
  }
  cblas_sgemm_batch (CblasRowMajor,
    &transa_arr[0],
    &transb_arr[0],
    &m_arr[0],
    &n_arr[0],
    &k_arr[0],
    &alpha_arr[0],
    &a_array[0],
    &lda_arr[0],
    &b_array[0],
    &ldb_arr[0],
    &beta_arr[0],
    &c_array[0],
    &ldc_arr[0],
    group_count,
    &group_size[0]);
#else
  for(size_t i = 0; i < batchC; ++i) {
    sgemm(transA,
          transB,
          (int)m,
          (int)n,
          (int)k,
          alpha,
          A->data() + (i % batchA) * strideA,
          (int)lda,
          B->data() + (i % batchB) * strideB,
          (int)ldb,
          beta,
          C->data() + i * strideC,
          (int)ldc);
  }
#endif
#else
  C; A; B; transA; transB; beta; scalar;
  ABORT("You need to compile with MKL in order to use the CPU version");
#endif
}

void ProdWithBias(marian::Tensor C,
                  const marian::Tensor& A,
                  const marian::Tensor& B,
                  const marian::Tensor& bias,
                  bool transA,
                  bool transB,
                  float beta,
                  float scalar) {
  cpu::Prod(C, A, B, transA, transB, beta, scalar);
  cpu::integer::AddBias(C, bias);
}

void Affine(marian::Tensor C,
            Ptr<Allocator> /*allocator*/,
            const marian::Tensor& A,
            const marian::Tensor& B,
            const marian::Tensor& bias,
            bool transA,
            bool transB,
            float beta,
            float scalar,
            bool reluPostprocess) {
  using namespace functional;
  ProdWithBias(C, A, B, bias, transA, transB, beta, scalar);
  if(reluPostprocess)
    cpu::Element(_1 = ReLU(_1), C); // @TODO: also fuse with AddBias
}


void CSRProd(marian::Tensor C,
             Ptr<Allocator> /*allocator*/,
             const marian::Tensor& S_values,
             const marian::Tensor& S_indices,
             const marian::Tensor& S_offsets,
             const marian::Tensor& D,
             bool transS,
             bool swapOperands,
             float beta) {
  C, S_values, S_indices, S_offsets, D;

  // Note: The CPU implementation currently only implements what's needed for decoding.

  // interpret tensor dimensions as matrix dimensions
  const auto& shapeC = C->shape();
  const auto& shapeD = D->shape();
  // If swapOperands, S and D are swapped (C = D x S instead of C = S x D).
  // In that case, in the next 6 lines, please read all dimensions as if they were reversed in order.
  auto rowsC = shapeC[-(int)swapOperands];
  auto colsC = shapeC.elements() / rowsC;
  auto rowsD = shapeD[-(int)swapOperands];
  auto colsD = shapeD.elements() / rowsD;
  auto rowsS = transS ? rowsD : rowsC;
  auto colsS = transS ? rowsC : rowsD;
  ABORT_IF(colsD != colsC, "Inconsistent outer dimensions in CSR product");
  if (swapOperands) { // make rowsX actual row dimensions again, likewise colsX
    std::swap(rowsC, colsC);
    std::swap(rowsD, colsD);
    std::swap(rowsS, colsS);
  }
  // sparse arrays
  auto numOffsets = S_offsets->shape().elements() - 1; // -1 since last value is length
  ABORT_IF(numOffsets != rowsS, "Unexpected number of rows in CSR argument"); numOffsets;
  ABORT_IF(S_values->shape() != S_indices->shape(), "CSR values and indices must have the same size");
  if (!transS && !swapOperands) {
    // C = S * D, where D = CSR matrix
    const auto* offsets = S_offsets->data<IndexType>();
    const auto* indices = S_indices->data<IndexType>();
    const auto* values  = S_values->data<float>();
    const auto* dataD   = D->data<float>();
    auto*       dataC   = C->data<float>();
    ABORT_IF(beta != 0 && beta != 1, "cpu::CSRProd only supports beta = 0 or 1");
    for (size_t i = 0; i < rowsC; i++) {
      auto add = (beta == 1); // first element: overwrite or add according to beta; subsequent elements: add
      for (size_t kk = offsets[i]; kk < offsets[i + 1]; kk++) {
        auto k = indices[kk];    // fetch the non-zero row
        auto valS = values[kk]; // and the value from that row
        // This code is written with the hope for good vectorization, and the hope
        // that adding to memory will be done efficiently by the caching system.
        if (valS == 1)
          if (!add)
            for (size_t j = 0; j < colsC; j++)
              dataC[i * colsC/*==colsD*/ + j] = dataD[k * colsC/*==colsD*/ + j]; // this is a memcpy()
          else
            for (size_t j = 0; j < colsC; j++)
              dataC[i * colsC/*==colsD*/ + j] += dataD[k * colsC/*==colsD*/ + j]; // this is a contiguous-vector addition
        else
          if (!add)
            for (size_t j = 0; j < colsC; j++)
              dataC[i * colsC/*==colsD*/ + j] = valS * dataD[k * colsC/*==colsD*/ + j];
          else
            for (size_t j = 0; j < colsC; j++)
              dataC[i * colsC/*==colsD*/ + j] += valS * dataD[k * colsC/*==colsD*/ + j]; // notice the +=
        add = true; // next iteration will add to existing result
      }
    }
  }
  else
    ABORT("CSRProd for transS={}, swapOperands={} is not yet implemented for CPU", transS, swapOperands);
}

}  // namespace cpu
}  // namespace marian
