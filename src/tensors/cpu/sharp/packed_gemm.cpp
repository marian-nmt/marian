#include "packed_gemm.h"
#include "tensors/tensor_allocator.h"
#include "tensors/tensor_operators.h"

#include <emmintrin.h>
#include <immintrin.h>
#include <tmmintrin.h>
#include <xmmintrin.h>
#include <cassert>
#include <cstddef>
#include <unordered_map>
//#include <chrono>

#ifdef _MSC_VER
#pragma warning(disable: 4505) // warning C4505: 'fbgemmAlignedAlloc' in fbgemm.h: unreferenced local function has been removed (missing 'static inline')
#endif

#if (USE_FBGEMM && MKL_FOUND)
#include "3rd_party/fbgemm/include/fbgemm/FbgemmFP16.h"
#include "3rd_party/fbgemm/include/fbgemm/QuantUtils.h"
#include "3rd_party/fbgemm/include/fbgemm/Fbgemm.h"

#include <mkl.h>
#include <mkl_types.h>
//#include "mkl_vsl.h"

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace fbgemm;
#endif // USE_FBGEMM && MKL_FOUND

namespace marian {
namespace cpu {
namespace variant {

#if (USE_FBGEMM && MKL_FOUND)
// initialize with a dummy
static PackedGemmMatrixFP16 packedPlaceholder(1, 1, 1, 1, 1, 1, 1, 1);

// This is copied from FBGEMM code
// A better way?
// blocked row-major format address arithmetic
inline uint64_t addr(const int r_,
                     const int c_,
                     const int brow_,
                     const int bcol_,
                     const int nbrow_,
                     const int nbcol_,
                     const int last_brow_) {
  uint64_t r = (uint64_t)r_;
  uint64_t c = (uint64_t)c_;

  uint64_t block_row_id = r / brow_;
  uint64_t brow_offset = (block_row_id * nbcol_) * (brow_ * bcol_);
  uint64_t block_col_id = c / bcol_;
  uint64_t bcol_offset
      = block_col_id * ((block_row_id != nbrow_ - 1) ? (brow_ * bcol_) : (last_brow_ * bcol_));
  uint64_t block_offset = brow_offset + bcol_offset;
  uint64_t inblock_offset = r % brow_ * bcol_ + c % bcol_;

  uint64_t index = block_offset + inblock_offset;
  return index;
}

void PackFp32(marian::Tensor out,
              const marian::Tensor in,
              bool transpose,
              int nrow,
              int ncol,
              int kernel_ncol_blocks,
              int brow,
              int bcol,
              int last_brow,
              int nbrow,
              int nbcol,
              uint64_t packsize) {
  //auto t_start = std::chrono::high_resolution_clock::now();
  // for the last embedding layer, pack it into int8
  // initialize memory
  uint8_t* outmemorg = out->data<uint8_t>();
  for(auto i = 0; i < packsize; i++) {
    outmemorg[i] = 0;
  }
  // save the other auxiliary variables
  uint64_t* auxmemsize = (uint64_t*)outmemorg;
  auxmemsize[0] = packsize;
  int* auxmem = (int*)(auxmemsize + 1);
  auxmem[0] = nrow;
  auxmem[1] = ncol;
  auxmem[2] = kernel_ncol_blocks;
  auxmem[3] = brow;
  auxmem[4] = bcol;
  auxmem[5] = last_brow;
  auxmem[6] = nbrow;
  auxmem[7] = nbcol;
  // cast to float16
  fbgemm::float16* outmem = (fbgemm::float16*)(outmemorg + 256);
  fbgemm::float16* dummy = new fbgemm::float16;
  // pack the matrix
  float* inmem = in->data();
  for(int i = 0; i < nrow; i++) {
    for(int j = 0; j < ncol; j++) {
      outmem[addr(i, j, brow, bcol, nbrow, nbcol, last_brow)]
          = tconv(!transpose ? inmem[i * ncol + j] : inmem[i + nrow * j], *dummy);
    }
  }
  delete dummy;

  //auto t_end = std::chrono::high_resolution_clock::now();
  //packingTime += (float) std::chrono::duration<double, std::milli>(t_end-t_start).count();
  //std::cout << "Packing time: " << packingTime << std::endl;
}

void GemmPackFp32(marian::Tensor C,
                  const marian::Tensor A,
                  const marian::Tensor B,
                  const marian::Tensor bias,
                  const int64_t m,
                  const int64_t n,
                  int transA) {
  // row major
  // keep the original mem
  fbgemm::float16* pmat = packedPlaceholder.pmat_;
  // retreive aux fields from the memory
  uint64_t* packedmemSize = (uint64_t*)B->data();
  packedPlaceholder.size_ = packedmemSize[0];
  int* packedmemAux = (int*)(packedmemSize + 1);
  packedPlaceholder.nrow_ = packedmemAux[0];
  packedPlaceholder.ncol_ = packedmemAux[1];
  packedPlaceholder.kernel_ncol_blocks_ = packedmemAux[2];
  packedPlaceholder.brow_ = packedmemAux[3];
  packedPlaceholder.bcol_ = packedmemAux[4];
  packedPlaceholder.last_brow_ = packedmemAux[5];
  packedPlaceholder.nbrow_ = packedmemAux[6];
  packedPlaceholder.nbcol_ = packedmemAux[7];

  // packed matrix
  packedPlaceholder.pmat_ = (fbgemm::float16*)(B->data<uint8_t>() + 256);

  for(int i = 0; i < m; ++i) {
    mkl_somatcopy('R', 'N', 1, n, 1, bias->data(), n, C->data() + n * i, n);
  }

#ifdef _OPENMP
#pragma omp parallel
#endif
  {
#ifdef _OPENMP
    int num_threads = omp_get_num_threads();
    int tid = omp_get_thread_num();
#else
    int num_threads = 1;
    int tid = 0;
#endif
    fbgemm::cblas_gemm_compute(transA ? matrix_op_t::Transpose : matrix_op_t::NoTranspose,
                      (int)m,
                      A->data(),
                      packedPlaceholder,
                      1,
                      C->data(),
                      tid,
                      num_threads);
  }

  // return back the original mem
  packedPlaceholder.pmat_ = pmat;
}
#else // USE_FBGEMM && MKL_FOUND
void PackFp32(marian::Tensor out,
              const marian::Tensor in,
              bool transpose,
              int nrow,
              int ncol,
              int kernel_ncol_blocks,
              int brow,
              int bcol,
              int last_brow,
              int nbrow,
              int nbcol,
              uint64_t packsize) {
                // does nothing. supports only FBGEMM based packed gemm at this moment.
}
void GemmPackFp32(marian::Tensor C,
                  const marian::Tensor A,
                  const marian::Tensor B,
                  const marian::Tensor bias,
                  const int64_t m,
                  const int64_t n,
                  int transA) {
                // does nothing. supports only FBGEMM based packed gemm at this moment.
}
#endif // USE_FBGEMM && MKL_FOUND

// This operates on floats after processing so doesn't care about int8_t vs
// int16_t.
void AddBias(marian::Tensor C, const marian::Tensor Bias) {
  float* y = C->data();
  const float* x = C->data();
  const float* bias = Bias->data();

  int m = C->shape().elements() / C->shape()[-1];
  int n = C->shape()[-1];
#ifdef __AVX512F__
  int n16 = n & ~15;
#else
  int n4 = (n / 4) * 4;
#endif

  for(int j = 0; j < m; ++j) {
    int i = 0;
#ifdef __AVX512F__
    for(; i < n16; i += 16) {
      __m512 ai = _mm512_loadu_ps(x + j * n + i);
      __m512 bi = _mm512_loadu_ps(bias + i);
      __m512 yi = _mm512_add_ps(ai, bi);
      _mm512_storeu_ps(y + j * n + i, yi);
    }
#else
    for(; i < n4; i += 4) {
      __m128 ai = _mm_loadu_ps(x + j * n + i);
      __m128 bi = _mm_loadu_ps(bias + i);
      __m128 yi = _mm_add_ps(ai, bi);
      _mm_storeu_ps(y + j * n + i, yi);
    }
#endif
    for(; i < n; i++) {
      y[j * n + i] = x[j * n + i] + bias[i];
    }
  }

  // std::cout << "Output: " << std::endl;
  // for (int ii = 0; ii < n; ii++) {
  //   std::cout << y[ii] << ",";
  // }
  // std::cout << std::endl;
}

}  // namespace variant
}  // namespace cpu
}  // namespace marian
