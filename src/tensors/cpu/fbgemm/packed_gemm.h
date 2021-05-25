#pragma once

#include "tensors/tensor.h"

namespace marian {
namespace cpu {
namespace variant { // Variants of GEMM implementations

// Returns the byte size of packed matrix in fp16. It's calculated by fbgemm's internal logic due to the paddings and different layouts.
// Packing with fp16 only targets AVX2 instruction sets for now.
// See '3rd_party/fbgemm/include/fbgemm/FbgemmFP16.h'.
// shape: shape of the tensor to be packed
// transpose: the matrix is transposed
// packsize (out): the size of the packed matrix in byte
void fbgemmPacked16PackInfo(const marian::Shape& shape,
                            const bool transpose,
                            /*out*/uint64_t& packsize);

// Returns the byte size of packed matrix in fp16. It's calculated by fbgemm's internal logic due to the paddings and different layouts.
// This function returns some other extra variables
// Packing with fp16 only targets AVX2 instruction sets for now.
// See '3rd_party/fbgemm/include/fbgemm/FbgemmFP16.h'.
// shape: shape of the tensor to be packed
// transpose: the matrix is transposed
// nrow (out): the number of rows
// ncol (out): the number of columns
// kernel_ncol_blocks (out): the number of column blocks
// brow (out): the number of rows in a block
// bcol (out): the number of columns in a block
// last_brow (out): the number of rows in the last block
// nbrow (out): row index in a block
// nbcol (out): column index in a block
// packsize (out): the size of the packed matrix in byte
void fbgemmPacked16PackInfo(const marian::Shape& shape,
                          const bool transpose,
                          /*out*/int& nrow,
                          /*out*/int& ncol,
                          /*out*/int& kernel_ncol_blocks,
                          /*out*/int& brow,
                          /*out*/int& bcol,
                          /*out*/int& last_brow,
                          /*out*/int& nbrow,
                          /*out*/int& nbcol,
                          /*out*/uint64_t& packsize); // @TODO: change to size_t where appropriate

// Returns the byte size of packed matrix in int8. It's calculated by fbgemm's internal logic due to the paddings and different layouts.
// See '3rd_party/fbgemm/src/PackBMatrix.cc'.
// shape: shape of the tensor to be packed
// packType: Type to be packed - packed8avx2 or packed8avx512
// transpose: the matrix is transposed
// nrow (out): the number of rows
// ncol (out): the number of columns
// packsize (out): the size of the packed matrix in byte
void fbgemmPacked8PackInfo(const marian::Shape& shape,
                           const marian::Type packType,
                           const bool transpose,
                           /*out*/int& nrow,
                           /*out*/int& ncol,
                           /*out*/uint64_t& packsize);

// Pack a matrix (fp16) into cache utilization efficient way (block format) into fp16
// out: output tensor - packed format
// inData: input tensor data - pointer of float data
// transpose: the matrix is transposed
// nrow: the number of rows
// ncol: the number of columns
// kernel_ncol_blocks: the number of column blocks
// brow: the number of rows in a block
// bcol: the number of columns in a block
// last_brow: the number of rows in the last block
// nbrow: row index in a block
// nbcol: column index in a block
// packsize: the size of the packed matrix
//          (the number of fp16 elements + padding (1024) + extra temporary memory (256))
void fbgemmPacked16Pack(marian::Tensor out,
                        const float* inData,
                        const bool transpose,
                        const int nrow,
                        const int ncol,
                        const int kernel_ncol_blocks,
                        const int brow,
                        const int bcol,
                        const int last_brow,
                        const int nbrow,
                        const int nbcol,
                        const uint64_t packsize); // @TODO: change to size_t where appropriate

// Pack a matrix (int8) into cache utilization efficient way (block format) together with quantization into int8
// out: output tensor - packed format and quantized into int8
// inData: input tensor data - pointer of float data
// packType: Type to be packed - packed8avx2 or packed8avx512
// transpose: the matrix is transposed
// nrow: the number of rows
// ncol: the number of columns
// packsize: the size of the packed matrix
//          (the size of int8 packed B from fbgemm:PackAWithQuantRowOffset + quantization scale, offset and zero point)
// quantRangeStdDevs: the range to be quantized for the original float data in multiples standard deviation
//                    the default value is 0.0f which means min/max quantization
//                    only a half range of normal int8 which is [-64, 63] used to avoid overflow
//                    during the accumulation in VPMADDUBSW instruction 
//                    https://intel.github.io/mkl-dnn/dev_guide_int8_computations.html
//                    (e.g. 3.f means the original tensor is quantized
//                    from [mean - 3.f * standard deviation, mean + 3.f * standard deviation] to [-64, 63])
void fbgemmPacked8Pack(marian::Tensor out,
                       const float* inData,
                       const marian::Type packType,
                       const bool transpose,
                       const int nrow,
                       const int ncol,
                       const uint64_t packsize,
                       const float quantRangeStdDevs = 0.f); // @TODO: change to size_t where appropriate

// GEMM operation on the packed B matrix
// C: output matrix
// A: A matrix
// B: B matrix (packed)
// m: the number of rows in A and C
// n: the number of columns in B and C
// transA: transpose of A matrix
// B is already packed. So, we don't need transB
void fbgemmPacked16Gemm(marian::Tensor C,
                        const marian::Tensor A,
                        const marian::Tensor B,
                        const marian::Tensor bias,
                        const size_t m,
                        const size_t n,
                        const int transA = 0);

// GEMM operation on the packed B matrix in 8 bit integers
// C: output matrix
// A: A matrix
// B: B matrix (packed)
// m: the number of rows in A and C
// n: the number of columns in B and C
// k: the number of columns in A and rows in B
// transA: transpose of A matrix
// transB: transpose of B matrix
void fbgemmPacked8Gemm(Type packType,
                       marian::Tensor C,
                       const marian::Tensor A,
                       const marian::Tensor B,
                       const size_t m,
                       const size_t n,
                       const size_t k,
                       const int transA = 0,
                       const int transB = 0);

}  // namespace variant
}  // namespace cpu
}  // namespace marian
