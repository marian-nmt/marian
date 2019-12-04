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
#pragma warning(disable: 4661) // 'fbgemm::PackMatrix<fbgemm::PackBMatrix<int8_t,int32_t>,int8_t,int32_t>::PackMatrix(int32_t,int32_t,inpType *,int,const fbgemm::BlockingFactors *)': no suitable definition provided for explicit template instantiation request
#pragma warning(disable: 4244) // fbgemm\quantutils.h(51): warning C4244: 'return': conversion from 'const _Ty' to 'T2', possible loss of data
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

// Copied code from fbgemm. It's padding required from some kernel in FBGEMM
// Verbatim - 'required by sw pipelined kernels'
// https://github.com/marian-nmt/FBGEMM/blob/master/include/fbgemm/FbgemmFP16.h#L109
const int PACK16_PADDING = 1024;  

// This is a memory space to store auxiliary variables for FBGEMM (e.g. block row, block column, kernel_ncol_blocks and etc.)
const int PACK16_SPECIALMEM = 256;

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

void fbgemmPacked16PackInfo(const marian::Shape& shape,
                            const bool transpose,
                            uint64_t& packsize) {
  int nrow, ncol, kernel_ncol_blocks, brow = 512, bcol, last_brow, nbrow, nbcol;
  fbgemmPacked16PackInfo(shape, transpose, nrow, ncol, kernel_ncol_blocks, brow, bcol, last_brow, nbrow, nbcol, packsize);
}

void fbgemmPacked16PackInfo(const marian::Shape& shape,
                            const bool transpose,
                            int& nrow,
                            int& ncol,
                            int& kernel_ncol_blocks,
                            int& brow,
                            int& bcol,
                            int& last_brow,
                            int& nbrow,
                            int& nbcol,
                            uint64_t& packsize) {
  nrow = transpose ? shape[1] : shape[0];
  ncol = transpose ? shape[0] : shape[1];
  kernel_ncol_blocks = 2;
  brow = 512;
  bcol = 8 * kernel_ncol_blocks;
  last_brow = nrow % brow == 0 ? brow : nrow % brow;
  nbrow = nrow % brow == 0 ? nrow / brow : (nrow + brow) / brow;
  nbcol = ncol % bcol == 0 ? ncol / bcol : (ncol + bcol) / bcol;
  ABORT_IF(ncol % bcol != 0, "ncol (number of columns) should be multiple of 16. {}", ncol);
  packsize = ((nbrow * brow) * (nbcol * bcol)) * sizeof(fbgemm::float16) + PACK16_PADDING
             + PACK16_SPECIALMEM;
}

void fbgemmPacked8PackInfo(const marian::Shape& shape,
                           const bool transpose,
                           int& nrow,
                           int& ncol,
                           uint64_t& packsize) {
    // Should be 2D - weight matrix
    ABORT_IF(shape.size() != 2,
            "Weight Matrix should be 2D");
    nrow = transpose ? shape[1] : shape[0];
    ncol = transpose ? shape[0] : shape[1];
    packsize = fbgemm::PackMatrix<fbgemm::PackBMatrix<int8_t>, int8_t>::packedBufferSize(
        transpose ? shape[1] : shape[0],
        transpose ? shape[0] : shape[1]);
    // add extra space for storing some other variables specific to B matrix
    // quantization sacles: 1 per column and float
    // quantization offset: 1 per column and int32
    // column offsets: 1 per column and int32
    packsize += ncol * (sizeof(float) + sizeof(int32_t) + sizeof(int32_t));
}

// This function computes the offset values for each column which are used for compensating the remainders of quantized values
// More detailed math is avilable in the FBGEMM's blog - https://engineering.fb.com/ml-applications/fbgemm/
inline void col_offsets_with_zero_pt_s8acc32(
    bool transpose,
    int K,
    int N,
    const int8_t* Bint8,
    const int32_t* B_zero_point,
    int32_t* col_offsets,
    int ncols_per_quant_group) {
  for (int n = 0; n < N; ++n) {
    int32_t sum = 0;
    for (int k = 0; k < K; ++k) {
      sum += transpose ? Bint8[k + n * K] : Bint8[k * N + n];
    }
    col_offsets[n] = sum - B_zero_point[n / ncols_per_quant_group] * K;
  }
}

void fbgemmPacked16Pack(marian::Tensor out,
                        const float* inData, // Packing is only available for 2D weight matrix in Marian. Otherwise, it's aborted in expanded_gemm.h.
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
  for(int i = 0; i < nrow; i++) {
    for(int j = 0; j < ncol; j++) {
      outmem[addr(i, j, brow, bcol, nbrow, nbcol, last_brow)]
          = tconv(!transpose ? inData[i * ncol + j] : inData[i + nrow * j], *dummy);
    }
  }
  delete dummy;
}

void fbgemmPacked8Pack(marian::Tensor out,
                       const float* inData,
                       const bool transpose,
                       const int nrow,
                       const int ncol,
                       const uint64_t packsize) {
  int k = nrow;
  int n = ncol;
  int len = k * n;

  // 1. collect stats for each column
  float* bqScale = new float[n];
  int32_t* bqZeropoint = new int32_t[n];

  const float* data = inData;
  float val = 0;

  if (transpose) {
    for (int jj = 0; jj < n; jj++) {
      float min = std::numeric_limits<float>::max(), max = std::numeric_limits<float>::min();
      double mean = 0, sqrsum = 0;
      for (int ii = 0; ii < k; ii++) {
        val = data[jj * k + ii];
        mean += val;
        sqrsum += val * val;
      }
      mean /= k;
      sqrsum /= k;
      sqrsum -= mean * mean;
      sqrsum = sqrt(sqrsum);

      min = (float)(mean - 7.0f*sqrsum);
      max = (float)(mean + 7.0f*sqrsum);
      bqScale[jj] = (max - min) / 255;
      bqZeropoint[jj] = (int32_t)(127 - max / bqScale[jj]);
    }
  } else {
    for (int jj = 0; jj < n; jj++) {
      float min = std::numeric_limits<float>::max(), max = std::numeric_limits<float>::min();
      double mean = 0, sqrsum = 0;
      for (int ii = 0; ii < k; ii++) {
        val = data[jj + ii * n];
        mean += val;
        sqrsum += val * val;
      }
      mean /= k;
      sqrsum /= k;
      sqrsum -= mean * mean;
      sqrsum = sqrt(sqrsum);

      min = (float)(mean - 7.0f*sqrsum);
      max = (float)(mean + 7.0f*sqrsum);
      bqScale[jj] = (max - min) / 255;
      bqZeropoint[jj] = (int32_t)(127 - max / bqScale[jj]);
    }
  }

  // 2. quantize
  int8_t* quantized = 0;
#ifdef _MSC_VER
  quantized = (int8_t*)_aligned_malloc(len, 256);
#else
  int result = posix_memalign((void**)&quantized, 256, len); result;
  assert(result == 0);
#endif
  for (int jj = 0; jj < n; jj++) {
    TensorQuantizationParams bQuantParam;
    bQuantParam.scale = bqScale[jj];
    bQuantParam.zero_point = bqZeropoint[jj];
    bQuantParam.precision = 8;

    if (transpose)
      fbgemm::Quantize<int8_t>(data + jj * k, quantized + jj * k, k, bQuantParam);
    else {
      for (int ii = 0; ii < k; ii++) {
        quantized[ii*n + jj] = fbgemm::Quantize<int8_t>(data[ii*n + jj], bQuantParam);
      }
    }
  }

  // 3. compute column offsets
  int32_t* col_offsets = new int32_t[n];
  col_offsets_with_zero_pt_s8acc32(transpose, k, n, quantized, bqZeropoint, col_offsets, 1);


  int8_t* packedbuf = out->data<int8_t>();
  for(auto i = 0; i < packsize; i++) {
    packedbuf[i] = 0;
  }

  // 4. packing
  PackBMatrix<int8_t> packedBN(
      transpose ? matrix_op_t::Transpose : matrix_op_t::NoTranspose,
      nrow, ncol, quantized, transpose ? nrow : ncol, packedbuf, 1);

  // copy quantization scale
  memcpy(packedbuf + (packsize - n * (sizeof(float) + sizeof(int32_t) + sizeof(int32_t))), bqScale, n * sizeof(float));
  // copy quantization offset
  memcpy(packedbuf + (packsize - n * (sizeof(int32_t) + sizeof(int32_t))), bqZeropoint, n * sizeof(int32_t));
  // copy column offsets to the memory
  memcpy(packedbuf + (packsize - n * sizeof(int32_t)), col_offsets, n * sizeof(int32_t));

#ifdef _MSC_VER
  _aligned_free(quantized);
#else
  free(quantized);
#endif
  delete[] col_offsets;
  delete[] bqScale;
  delete[] bqZeropoint;
}

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

  if(bias != nullptr) {
#if MKL_FOUND
    for(int i = 0; i < m; ++i) {
      mkl_somatcopy('R', 'N', 1, n, 1, bias->data(), n, C->data() + n * i, n);
    }
#else
    for(int i = 0; i < m; ++i) {
      std::copy(bias->data(), bias->data() + n, C->data() + n * i);
    }
#endif
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
                      bias != nullptr ? 1.0f : 0.0f,
                      C->data(),
                      tid,
                      num_threads);
  }

  // return back the original mem
  packedPlaceholder.pmat_ = pmat;
}

// GEMM operation on the packed B matrix in 8 bit integers
// C: output matrix
// A: A matrix
// B: B matrix (packed)
// m: the number of rows in A and C
// n: the number of columns in B and C
// k: the number of columns in A and the number of rows in B
// transA: whether A matrix is transposed or not
// transB: whether B matrix is transposed or not
void fbgemmPacked8Gemm(marian::Tensor C,
                       const marian::Tensor A,
                       const marian::Tensor B,
                       const size_t m,
                       const size_t n,
                       const size_t k,
                       const int transA,
                       const int transB) {
  // compute range to quantize A (activations) - (min/max quantization)
  float min_est = std::numeric_limits<float>::max(), max_est = std::numeric_limits<float>::min();

  int elem = A->shape().elements();
  float* data = A->data();
  // AVX based find min/max
  FindMinMax(data, &min_est, &max_est, elem);

  float ascale = (max_est - min_est) / 255;
  int32_t azeropoint = (int32_t)(255 - max_est / ascale);

  std::vector<int32_t> row_offset_buf(PackAWithQuantRowOffset<uint8_t>::rowOffsetBufferSize());
  PackAWithQuantRowOffset<uint8_t> packAN(
      transA ? matrix_op_t::Transpose : matrix_op_t::NoTranspose,
      (int32_t) (transA ? k : m),
      (int32_t) (transA ? m : k),
      A->data(),
      (int32_t) (transA ? m : k),
      nullptr, /*buffer for packed matrix*/
      ascale,
      azeropoint,
      1, /*groups*/
      row_offset_buf.data());

  // packed matrix size of B
  int bPackSize = PackMatrix<PackBMatrix<int8_t>, int8_t>::packedBufferSize((int32_t)k, (int32_t)n);

  // retrieve B matrix
  int8_t* bdata = B->data<int8_t>();
  float* bqScale = new float[n];
  memcpy(bqScale, bdata + bPackSize, n * sizeof(float));

  int32_t* bqZeropoint = new int32_t[n];
  memcpy(bqZeropoint, bdata + bPackSize + n * sizeof(float), n * sizeof(int32_t));

  int32_t* col_offsets = new int32_t[n];
  memcpy(col_offsets, bdata + bPackSize + n * (sizeof(float) + sizeof(int32_t)), n * sizeof(int32_t));

  DoNothing<float, float> doNothingObj{};
  ReQuantizeForFloat<false, QuantizationGranularity::OUT_CHANNEL> outputProcObj(
      doNothingObj,
      ascale,
      bqScale,
      azeropoint,
      bqZeropoint,
      packAN.getRowOffsetBuffer(),
      col_offsets,
      nullptr,
      (std::uint32_t) n);

  PackBMatrix<int8_t> repackedBN(
    transB ? matrix_op_t::Transpose : matrix_op_t::NoTranspose, (int32_t) k, (int32_t) n, bdata, (int32_t) (transB ? k : n, 1));

  // gemm computation
  fbgemmPacked(packAN, repackedBN, C->data(), (int32_t*)C->data(), (int32_t) n, outputProcObj, 0, 1);

  delete[] col_offsets;
  delete[] bqZeropoint;
  delete[] bqScale;
}

#endif // USE_FBGEMM

}  // namespace variant
}  // namespace cpu
}  // namespace marian
