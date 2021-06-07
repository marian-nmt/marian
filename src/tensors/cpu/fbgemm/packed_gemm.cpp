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
#pragma warning(disable: 4717) // 'fbgemm::PackMatrix<fbgemm::PackAWithQuantRowOffset<unsigned char,int>,unsigned char,int>::isThisLastKBlock': recursive on all control paths, function will cause runtime stack overflow
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

// This is the maximum value of FP16 type. There is a template type implementation, but it doesn't work on windows.
// To keep the consistent result, just use the constant value instead of #ifdef _MSC_VER.
// Template type implementation: float FP16_MAX = NumericLimits<float>(Type::float16).max;
const float FP16_MAX = 65504.f;

// This function clips a value into a [min, max] range
inline float clip(float value, float min, float max) {
  return std::max(min, std::min(value, max));
}

// This is copied from FBGEMM code
// A better way?
// will be removed, when FBGEMM api is changed
// blocked row-major format address arithmetic
//
// Returns the memory address in the packed (block formatted) matrix array of a specific element 
// indexed by the original non-packed array.
//
// @param r_ row index in the original matrix
// @param c_ column index in the original matrix
// @param brow_ row wide block index
// @param bcol_ column wide block index
// @param nbrow_ number of blocks in row
// @param nbcol_ number of blocks in column
// @param last_brow_ row number of the last block
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

// Returns a value in 2D array with the row, column index (i, j) and transposed flag.
// The number of rows and columns needs to be passed.
// The transposed flag indicates if the underlying data needs to be accessed in a tranposed layout or not.
inline float getVal2dArr(const float* data, size_t i, size_t j, size_t rows, size_t cols, bool transposed) {
  ABORT_IF(i >= rows, "Row index {} exceeds the number of rows {}.", i, rows);
  ABORT_IF(j >= cols, "Column index {} exceeds the number of columns {}.", j, cols);
  return transposed ? data[j * rows + i] : data[i * cols + j];
}

// Memory blocking factors (parameters) for packing into AVX2 int8
static const fbgemm::BlockingFactors Packed8Avx2BlockingFactors = {
    PackingTraits<int8_t, int32_t, inst_set_t::avx2>::MR,
    PackingTraits<int8_t, int32_t, inst_set_t::avx2>::NR,
    PackingTraits<int8_t, int32_t, inst_set_t::avx2>::NR_MIN,
    PackingTraits<int8_t, int32_t, inst_set_t::avx2>::ROW_INTERLEAVE,
    PackingTraits<int8_t, int32_t, inst_set_t::avx2>::MCB,
    PackingTraits<int8_t, int32_t, inst_set_t::avx2>::KCB,
    PackingTraits<int8_t, int32_t, inst_set_t::avx2>::NCB
};

// Memory blocking factors (parameters) for packing into AVX512 int8
static const fbgemm::BlockingFactors Packed8Avx512BlockingFactors = {
    PackingTraits<int8_t, int32_t, inst_set_t::avx512>::MR,
    PackingTraits<int8_t, int32_t, inst_set_t::avx512>::NR,
    PackingTraits<int8_t, int32_t, inst_set_t::avx512>::NR_MIN,
    PackingTraits<int8_t, int32_t, inst_set_t::avx512>::ROW_INTERLEAVE,
    PackingTraits<int8_t, int32_t, inst_set_t::avx512>::MCB,
    PackingTraits<int8_t, int32_t, inst_set_t::avx512>::KCB,
    PackingTraits<int8_t, int32_t, inst_set_t::avx512>::NCB
};

// This function returns the correct blocking factors structure for given packing type.
inline const fbgemm::BlockingFactors* getBlockingFactors(marian::Type packType) {
  if(packType == Type::packed8avx2) {
    return &Packed8Avx2BlockingFactors;
  } else if(packType == Type::packed8avx512) {
    return &Packed8Avx512BlockingFactors;
  } else {
    ABORT("Only avx2 and avx512 instruction sets are supported for int8. {}", packType);
  }
}

// Returns the byte size of packed matrix in fp16. It's calculated by fbgemm's internal logic due to the paddings and different layouts.
// Packing with fp16 only targets AVX2 instruction sets for now.
// See '3rd_party/fbgemm/include/fbgemm/FbgemmFP16.h'.
// shape: shape of the tensor to be packed
// transpose: the matrix is transposed
// packsize (out): the size of the packed matrix in byte
void fbgemmPacked16PackInfo(const marian::Shape& shape,
                            const bool transpose,
                            uint64_t& packsize) {
  int nrow, ncol, kernel_ncol_blocks, brow = 512, bcol, last_brow, nbrow, nbcol;
  fbgemmPacked16PackInfo(shape, transpose, nrow, ncol, kernel_ncol_blocks, brow, bcol, last_brow, nbrow, nbcol, packsize);
}

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
                           int& nrow,
                           int& ncol,
                           uint64_t& packsize) {
    // Should be 2D - weight matrix
    ABORT_IF(shape.size() != 2,
            "Weight Matrix should be 2D");
    nrow = transpose ? shape[1] : shape[0];
    ncol = transpose ? shape[0] : shape[1];

    const fbgemm::BlockingFactors* params = getBlockingFactors(packType);

    packsize = fbgemm::PackMatrix<fbgemm::PackBMatrix<int8_t>, int8_t>::packedBufferSize(
        transpose ? shape[1] : shape[0],
        transpose ? shape[0] : shape[1], params);
    // add extra space for storing some other variables specific to B matrix
    // quantization sacles: 1 per column and float
    // quantization offset: 1 per column and int32
    // column offsets: 1 per column and int32
    packsize += ncol * (sizeof(float) + sizeof(int32_t) + sizeof(int32_t));
}

// This function computes the offset values for each column which are used for compensating the remainders of quantized values
// More detailed math is avilable in the FBGEMM's blog - https://engineering.fb.com/ml-applications/fbgemm/
inline void colOffsetsWithZeroPtS8acc32(
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
      float src = clip(transpose ? inData[i + nrow * j] : inData[i * ncol + j], -FP16_MAX, FP16_MAX);
      outmem[addr(i, j, brow, bcol, nbrow, nbcol, last_brow)] = tconv(src, *dummy);
    }
  }
  delete dummy;
}

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
                       const float quantRangeStdDevs) {
  int k = nrow;
  int n = ncol;
  int len = k * n;

  // 1. collect stats for each column
  float* quantScaleB = new float[n];
  int32_t* quantZeropointB = new int32_t[n];

  const float* data = inData;
  float val = 0;

  // Use half of the quantization range to prevent overflow of VPMADDUBSW
  constexpr static int quantizedRange = 127;
  constexpr static int quantizedMax = 63;

  // This routine compute the quantization range for each column - either one of min/max range or quantRangeStdDevs sigma range.
  for (size_t jj = 0; jj < n; jj++) { // for each column, collect stats (min/max or mean/std.dev.)
    float min = std::numeric_limits<float>::max(), max = std::numeric_limits<float>::lowest();
    double mean = 0, sqrSum = 0;
    for (size_t ii = 0; ii < k; ii++) { // in a column, go throuhg all the rows and collect stats
      val = getVal2dArr(data, ii, jj, k, n, transpose);
      // If quantRangeStdDevs is 0.f, min/max values of the columns is used as a quantization range
      if(quantRangeStdDevs == 0.f) {
        if(min > val)
          min = val;
        if(max < val)
          max = val;
      } else {
        // Quantize by std.dev. range
        mean += val;
        sqrSum += val * val;
      }
    }
    // If a quantization range (in multiples of std. dev.) is given with a non-zero value,
    // it calculate the range for this column (different quantization scale/offset are used for each column)
    if(quantRangeStdDevs != 0.f) {
      mean /= k;
      sqrSum /= k;
      sqrSum -= mean * mean;
      sqrSum = sqrt(sqrSum);
      min = (float)(mean - quantRangeStdDevs * sqrSum);
      max = (float)(mean + quantRangeStdDevs * sqrSum);
    }
    // based on the quantization range, this computes the scale and offset for the quantization
    quantScaleB[jj] = (max - min) / quantizedRange;
    quantZeropointB[jj] = (int32_t)(quantizedMax - max / quantScaleB[jj]);
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
    bQuantParam.scale = quantScaleB[jj];
    bQuantParam.zero_point = quantZeropointB[jj];
    bQuantParam.precision = 7;  // Use half of the quantization range to prevent overflow of VPMADDUBSW

    if (transpose)
      fbgemm::Quantize<int8_t>(data + jj * k, quantized + jj * k, k, bQuantParam);
    else {
      for (int ii = 0; ii < k; ii++) {
        quantized[ii*n + jj] = fbgemm::Quantize<int8_t>(data[ii*n + jj], bQuantParam);
      }
    }
  }

  // 3. compute column offsets
  int32_t* colOffsets = new int32_t[n];
  colOffsetsWithZeroPtS8acc32(transpose, k, n, quantized, quantZeropointB, colOffsets, 1);


  int8_t* packedBuf = out->data<int8_t>();
  for(auto i = 0; i < packsize; i++) {
    packedBuf[i] = 0;
  }

  // 4. packing
  const fbgemm::BlockingFactors* params = getBlockingFactors(packType);
  
  PackBMatrix<int8_t> packedBN(
      transpose ? matrix_op_t::Transpose : matrix_op_t::NoTranspose,
      nrow, ncol, quantized, transpose ? nrow : ncol, packedBuf, 1, params);

  // copy quantization scale
  memcpy(packedBuf + (packsize - n * (sizeof(float) + sizeof(int32_t) + sizeof(int32_t))), quantScaleB, n * sizeof(float));
  // copy quantization offset
  memcpy(packedBuf + (packsize - n * (sizeof(int32_t) + sizeof(int32_t))), quantZeropointB, n * sizeof(int32_t));
  // copy column offsets to the memory
  memcpy(packedBuf + (packsize - n * sizeof(int32_t)), colOffsets, n * sizeof(int32_t));

#ifdef _MSC_VER
  _aligned_free(quantized);
#else
  free(quantized);
#endif
  delete[] colOffsets;
  delete[] quantScaleB;
  delete[] quantZeropointB;
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
void fbgemmPacked8Gemm(Type packType,
                       marian::Tensor C,
                       const marian::Tensor A,
                       const marian::Tensor B,
                       const size_t m,
                       const size_t n,
                       const size_t k,
                       const int transA,
                       const int transB) {
  const fbgemm::BlockingFactors* params = getBlockingFactors(packType);

  // Check if the packed format matches with the available AVX instruction set in the machine
  const bool avx512Support = fbgemmHasAvx512Support();
  if((packType == Type::packed8avx2 && avx512Support)
     || (packType == Type::packed8avx512 && !avx512Support)) {
    ABORT("FBGEMM doesn't allow to use {} packing order on {} CPUs",
          packType == Type::packed8avx2 ? "AVX2" : "AVX512",
          avx512Support ? "AVX512" : "AVX2");
  }

  // compute range to quantize A (activations) - (min/max quantization)
  float minA = std::numeric_limits<float>::max(), maxA = std::numeric_limits<float>::lowest();

  int elemA = A->shape().elements();
  float* dataA = A->data();
  // AVX based find min/max
  FindMinMax(dataA, &minA, &maxA, elemA);

  float quantScaleA = (maxA - minA) / 255;
  int32_t quantZeropointA = (int32_t)(255 - maxA / quantScaleA);

  // To avoid any repeated memory allocation and deallocation, make the scratch buffer variables static thread_local
  // In a multi-threaded situation, heap access lock for the memory allocation/free could
  // makes all the threads are blocked by each other. (heap contention)
  const size_t sizeBufA = params->KCB * params->MCB;
  static thread_local std::vector<uint8_t> packedBufA;
  if (packedBufA.size() < sizeBufA)
	  packedBufA.resize(sizeBufA);
  const size_t sizeRowOffsetBufA = PackAWithQuantRowOffset<uint8_t>::rowOffsetBufferSize();
  static thread_local std::vector<int32_t> rowOffsetBufA;
  if (rowOffsetBufA.size() < sizeRowOffsetBufA)
	  rowOffsetBufA.resize(sizeRowOffsetBufA);

  PackAWithQuantRowOffset<uint8_t> packA(
      transA ? matrix_op_t::Transpose : matrix_op_t::NoTranspose,
      (int32_t)(transA ? k : m),
      (int32_t)(transA ? m : k),
      A->data(),
      (int32_t)(transA ? m : k),
      // buffer for packed matrix, pass a pre-allocated memory to avoid additional allocation/deallocation inside fbgemm
      packedBufA.data(),
      quantScaleA,
      quantZeropointA,
      1, /*groups*/
      rowOffsetBufA.data(),
      params);

  // packed matrix size of B
  int packSizeB = PackMatrix<PackBMatrix<int8_t>, int8_t>::packedBufferSize((int32_t)k, (int32_t)n);

  // retrieve B matrix
  int8_t* dataB = B->data<int8_t>();

  // To avoid any repeated memory allocation and deallocation, make the scratch buffer variables static thread_local
  // In a multi-threaded situation, heap access lock for the memory allocation/free could
  // makes all the threads are blocked by each other. (heap contention)
  static thread_local std::vector<float> quantScaleB;
  if (quantScaleB.size() < n)
    quantScaleB.resize(n);
  memcpy(quantScaleB.data(), dataB + packSizeB, n * sizeof(float));

  static thread_local std::vector<int32_t> quantZeropointB;
  if (quantZeropointB.size() < n)
    quantZeropointB.resize(n);
  memcpy(quantZeropointB.data(), dataB + packSizeB + n * sizeof(float), n * sizeof(int32_t));

  static thread_local std::vector<int32_t> colOffsetsB;
  if (colOffsetsB.size() < n)
    colOffsetsB.resize(n);
  memcpy(colOffsetsB.data(), dataB + packSizeB + n * (sizeof(float) + sizeof(int32_t)), n * sizeof(int32_t));

  DoNothing<float, float> doNothingObj{};
  ReQuantizeForFloat<false, QuantizationGranularity::OUT_CHANNEL> outputProcObj(
      doNothingObj,
      quantScaleA,
      quantScaleB.data(),
      quantZeropointA,
      quantZeropointB.data(),
      packA.getRowOffsetBuffer(),
      colOffsetsB.data(),
      nullptr,
      (std::uint32_t) n);

  PackBMatrix<int8_t> repackedB(
    transB ? matrix_op_t::Transpose : matrix_op_t::NoTranspose, (int32_t) k, (int32_t) n, dataB, (int32_t) (transB ? k : n), 1, params);

  // gemm computation
  fbgemmPacked(packA, repackedB, C->data(), (int32_t*)C->data(), (int32_t) n, outputProcObj, 0, 1, params);
}

#endif // USE_FBGEMM

}  // namespace variant
}  // namespace cpu
}  // namespace marian
