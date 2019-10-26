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

#if USE_FBGEMM
#ifdef _MSC_VER
#pragma warning(disable: 4505) // 'fbgemmAlignedAlloc' in fbgemm.h: unreferenced local function has been removed (missing 'static inline')
#pragma warning(disable: 4251) // 'fbgemm::CompressedSparseColumn::colptr_': class 'std::vector<int,std::allocator<_Ty>>' needs to have dll-interface to be used by clients of class 'fbgemm::CompressedSparseColumn'
// the following does not work; need to manually disable them in Linker options
//#pragma comment(linker, "/ignore:4049") // locally defined symbol ...asmjit... imported
//#pragma comment(linker, "/ignore:4217") // locally defined symbol ...asmjit... imported
#endif


#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
#endif

#include "3rd_party/fbgemm/include/fbgemm/FbgemmFP16.h"
#include "3rd_party/fbgemm/include/fbgemm/QuantUtils.h"
#include "3rd_party/fbgemm/include/fbgemm/Fbgemm.h"

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#if MKL_FOUND
#include <mkl.h>
#include <mkl_types.h>
#endif

using namespace fbgemm;
#endif // USE_FBGEMM

namespace marian {
namespace cpu {
namespace variant { // Variants of GEMM implementations

#if USE_FBGEMM
// initialize with a dummy
// When this class is instantiated,
// the actual packing operation is happening. If we create this instance every time we call GEMM,
// we are doing packing every time and very slow.
// In Caffe2, the operator is stateful and hold an instance of this.
// But, we don't have any logic for this in marian. We can only cache a tensor (which means a memory chunk).
// So, for now, we keep the packed memory on our own 1D tensor, then when we call GEMM,
// we just reuse this instance again and again by replacing the class members (including memory pointer). Eventually,
// I will add a new constructor to the class in FBGEMM which accepts
// pre - allocated and pre - packed memory as a parameter.After it's done,
// this temporary buffer will be removed.
// When constructing this dummy buffer, ones are used for all the parameters to allocate minimum amount of memory.
//
// In a multi marian instance setting (as a dynamic library),
// different marian instances should not share this variable.
static thread_local PackedGemmMatrixFP16 packedPlaceholder(1, 1, 1, 1, 1, 1, 1, 1);

// This is copied from FBGEMM code
// A better way?
// will be removed, when FBGEMM api is changed
// blocked row-major format address arithmetic
/**
 * Returns the memory address in the packed (block formatted) matrix array of a specific element 
 * indexed by the original non-packed array.
 *
 * @param r_ row index in the original matrix
 * @param c_ column index in the original matrix
 * @param brow_ row wide block index
 * @param bcol_ column wide block index
 * @param nbrow_ number of blocks in row
 * @param nbcol_ number of blocks in column
 * @param last_brow_ row number of the last block
 */
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
              const bool transpose,
              const int nrow,
              const int ncol,
              const int kernel_ncol_blocks,
              const int brow,
              const int bcol,
              const int last_brow,
              const int nbrow,
              const int nbcol,
              const uint64_t packsize) {
  // initialize memory
  uint8_t* outmemorg = out->data<uint8_t>();
  for(auto i = 0; i < packsize; i++) {
    outmemorg[i] = 0;
  }
  // save the other auxiliary variables
  uint64_t* auxmemsize = (uint64_t*)outmemorg;
  auxmemsize[0] = packsize;
  // save FBGEMM related parameters into the header of the allocated memory by marian
  int32_t header[8];
  header[0] = nrow;
  header[1] = ncol;
  header[2] = kernel_ncol_blocks;
  header[3] = brow;
  header[4] = bcol;
  header[5] = last_brow;
  header[6] = nbrow;
  header[7] = nbcol;
  memcpy(auxmemsize + 1, header, sizeof(header));
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
}

// GEMM operation on the packed B matrix
// C: output matrix
// A: A matrix
// B: B matrix (packed)
// m: the number of rows in A and C
// n: the number of columns in B and C
// transA: transpose of A matrix
// B is already packed. So, we don't need transB
void GemmPackFp32(marian::Tensor C,
                  const marian::Tensor A,
                  const marian::Tensor B,
                  const marian::Tensor bias,
                  const size_t m,
                  const size_t n,
                  const int transA) {
  // row major
  // keep the original mem
  fbgemm::float16* pmat = packedPlaceholder.pmat_;
  // retreive aux fields from the memory
  uint64_t* packedmemSize = (uint64_t*)B->data();
  packedPlaceholder.size_ = packedmemSize[0];
  int32_t header[8];
  memcpy(header, packedmemSize + 1, sizeof(header));
  packedPlaceholder.nrow_ = header[0];
  packedPlaceholder.ncol_ = header[1];
  packedPlaceholder.kernel_ncol_blocks_ = header[2];
  packedPlaceholder.brow_ = header[3];
  packedPlaceholder.bcol_ = header[4];
  packedPlaceholder.last_brow_ = header[5];
  packedPlaceholder.nbrow_ = header[6];
  packedPlaceholder.nbcol_ = header[7];

  // packed matrix
  packedPlaceholder.pmat_ = (fbgemm::float16*)(B->data<uint8_t>() + 256);

#if MKL_FOUND
  for(int i = 0; i < m; ++i) {
    mkl_somatcopy('R', 'N', 1, n, 1, bias->data(), n, C->data() + n * i, n);
  }
#else
  for(int i = 0; i < m; ++i) {
    std::copy(bias->data(), bias->data() + n, C->data() + n * i);
  }
#endif

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
#else // USE_FBGEMM
void PackFp32(marian::Tensor out,
              const marian::Tensor in,
              const bool transpose,
              const int nrow,
              const int ncol,
              const int kernel_ncol_blocks,
              const int brow,
              const int bcol,
              const int last_brow,
              const int nbrow,
              const int nbcol,
              const uint64_t packsize) {
                // does nothing. supports only FBGEMM based packed gemm at this moment.
                ABORT("FBGEMM is needed to use packed GEMM.");
}
void GemmPackFp32(marian::Tensor C,
                  const marian::Tensor A,
                  const marian::Tensor B,
                  const marian::Tensor bias,
                  const size_t m,
                  const size_t n,
                  const int transA) {
                // does nothing. supports only FBGEMM based packed gemm at this moment.
                ABORT("FBGEMM is needed to use packed GEMM.");
}
#endif // USE_FBGEMM

}  // namespace variant
}  // namespace cpu
}  // namespace marian
