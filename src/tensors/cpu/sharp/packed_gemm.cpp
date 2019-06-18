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
namespace pack {

//static float packingTime = 0;

#if (USE_FBGEMM && MKL_FOUND)
// initialize with a dummy
static PackedGemmMatrixFP16 packedPlaceholder(1, 1, 1, 1, 1, 1, 1, 1);

// temporary variable for int 8
//static PackBMatrix<int8_t>* packedBint8 = nullptr;
// transformer base wmt 2017
// static float bqScale = 0.39/128;
// static int32_t bqZeropoint = 0;
// old student de-en
//static float* bqScale;
//static int32_t* bqZeropoint;
// static float bqScale = 0.683/106;
// static int32_t bqZeropoint = 21;
// static float bqScale = 0.9672/128;
// static int32_t bqZeropoint = 0;
//static std::vector<int32_t>* col_offsets = nullptr;

//inline void col_offsets_with_zero_pt_s8acc32_ref(
//    bool transpose,
//    int K,
//    int N,
//    const int8_t* Bint8,
//    const int32_t* B_zero_point,
//    int32_t* col_offsets,
//    int ncols_per_quant_group) {
//  for (int n = 0; n < N; ++n) {
//    int32_t sum = 0;
//    for (int k = 0; k < K; ++k) {
//      sum += transpose ? Bint8[k + n * K] : Bint8[k * N + n];
//    }
//    col_offsets[n] = sum - B_zero_point[n / ncols_per_quant_group] * K;
//  }
//}

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
  // if (in->shape().size() == 2 && (in->shape()[0] >= 32000 || in->shape()[1] >= 32000)) {
 if (false) {
#if 0 
   if (packedBint8 == nullptr) {
      int k = transpose ? in->shape()[1] : in->shape()[0];
      int n = transpose ? in->shape()[0] : in->shape()[1];
      // std::cout << "transpose: " << transpose << ", k: " << k << ", n: " << n << std::endl;
      // std::cout << ", bqScale: " << bqScale << ", bqZeropoint: " << bqZeropoint << std::endl;
      // two steps
      // 0. quantize --> this should be done outside
      int len = in->shape()[0]*in->shape()[1];

      // 0-1. collect stats for each class
      bqScale = new float[n];
      bqZeropoint = new int32_t[n];

      // int numBin = 20;
      // float denum = 2/(float)numBin;

      // int hist[numBin] = { 0, };

      // Transposed only
      float* data = in->data();
      float val = 0;
      for (int jj = 0; jj < n; jj++) {
        float min = 1000000, max = -10000000;
        for (int ii = 0; ii < k; ii++) {
          val = data[jj*k + ii];
          if (val < min) min = val;
          if (val > max) max = val;
          // hist[(int)((val + 1)/denum)]++;
        }
        bqScale[jj] = (max - min)/255;
        bqZeropoint[jj] = (int32_t)(127 - max / bqScale[jj]);
        // bqScale[jj] = (0.3 + 0.4)/255;
        // bqZeropoint[jj] = (int32_t)(127 - 0.3 / bqScale[jj]);
      }

      // std::cout << "hist: ";
      // for (int ii = 0; ii < numBin; ii++) {
      //   std::cout << hist[ii] << ", ";
      // }
      // std::cout << std::endl;
      // std::cout << "len: " << len << ", bqScale: " << bqScale << ", bqZeropoint: " << bqZeropoint << std::endl;
      //int8_t quantized[len]; // aligned malloc?
      int8_t* quantized = (int8_t*)aligned_alloc(256, len);
      // std::cout << "len: " << len << ", bqScale: " << bqScale << ", bqZeropoint: " << bqZeropoint << std::endl;
      for (int ii = 0; ii < n; ii++) {
        TensorQuantizationParams bQuantParam;
        // std::cout << "len: " << len << ", bqScale: " << bqScale << ", bqZeropoint: " << bqZeropoint << std::endl;
        bQuantParam.scale = bqScale[ii];
        bQuantParam.zero_point = bqZeropoint[ii];
        bQuantParam.precision = 8;
        // std::cout << "len: " << len << ", bqScale: " << bqScale << ", bqZeropoint: " << bqZeropoint << std::endl;

        fbgemm::Quantize<int8_t>(in->data() + ii * k, quantized + ii * k, k, bQuantParam);
      }
      // std::cout << "original" << std::endl;
      // for (int ii = 0; ii < n; ii++) {
      //   for (int jj = 0; jj < 1; jj++) {
      //     std::cout << in->data()[ii * k + jj] << ","; 
      //   }
      //   std::cout << std::endl;
      // }
      // std::cout << "quantized" << std::endl;
      // for (int ii = 0; ii < 1; ii++) {
      //   for (int jj = 0; jj < k; jj++) {
      //     std::cout << (int32_t)quantized[ii * k + jj] << ","; 
      //   }
      //   std::cout << std::endl;
      // }
      // 1. compute column offsets
      col_offsets = new std::vector<int32_t>(n);
      col_offsets_with_zero_pt_s8acc32_ref(transpose, k, n, quantized, bqZeropoint, col_offsets->data(), 1);
      // for (int ii = 0; ii < n; ii++) {
      //   std::cout << (int32_t)col_offsets->data()[ii] << ","; 
      // }
      // std::cout << std::endl;
      // std::cout << "calc offset done" << std::endl;
      // 2. packing
      // uint8_t* packedmem = aligned_alloc(256, len);
      packedBint8 = new PackBMatrix<int8_t>(transpose ? matrix_op_t::Transpose : matrix_op_t::NoTranspose,
                                            k, // in->shape()[0],
                                            n, // in->shape()[1],
                                            quantized,
                                            in->shape()[1]);
      // std::cout << "packing B done" << std::endl;
      // int k = transpose ? in->shape()[1] : in->shape()[0];
      // int n = transpose ? in->shape()[0] : in->shape()[1];
      // std::cout << "transpose: " << transpose << ", k: " << k << ", n: " << n << std::endl;
      // std::cout << ", bqScale: " << bqScale << ", bqZeropoint: " << bqZeropoint << std::endl;
      // // two steps
      // // 0. quantize --> this should be done outside
      // int len = in->shape()[0]*in->shape()[1];
      // std::cout << "len: " << len << ", bqScale: " << bqScale << ", bqZeropoint: " << bqZeropoint << std::endl;
      // //int8_t quantized[len]; // aligned malloc?
      // int8_t* quantized = (int8_t*)aligned_alloc(256, len);
      // std::cout << "len: " << len << ", bqScale: " << bqScale << ", bqZeropoint: " << bqZeropoint << std::endl;
      // TensorQuantizationParams bQuantParam;
      // std::cout << "len: " << len << ", bqScale: " << bqScale << ", bqZeropoint: " << bqZeropoint << std::endl;
      // bQuantParam.scale = bqScale;
      // bQuantParam.zero_point = bqZeropoint;
      // bQuantParam.precision = 8;
      // std::cout << "len: " << len << ", bqScale: " << bqScale << ", bqZeropoint: " << bqZeropoint << std::endl;

      // fbgemm::Quantize<int8_t>(in->data(), quantized, len, bQuantParam);
      // std::cout << "original" << std::endl;
      // // for (int ii = 0; ii < n; ii++) {
      // //   for (int jj = 0; jj < 1; jj++) {
      // //     std::cout << in->data()[ii * k + jj] << ","; 
      // //   }
      // //   std::cout << std::endl;
      // // }
      // std::cout << "quantized" << std::endl;
      // // for (int ii = 0; ii < 1; ii++) {
      // //   for (int jj = 0; jj < k; jj++) {
      // //     std::cout << (int32_t)quantized[ii * k + jj] << ","; 
      // //   }
      // //   std::cout << std::endl;
      // // }
      // // 1. compute column offsets
      // col_offsets = new std::vector<int32_t>(n);
      // col_offsets_with_zero_pt_s8acc32_ref(k, n, n, quantized, &bqZeropoint, col_offsets->data(), n);
      // // for (int ii = 0; ii < n; ii++) {
      // //   std::cout << (int32_t)col_offsets->data()[ii] << ","; 
      // // }
      // std::cout << std::endl;
      // std::cout << "calc offset done" << std::endl;
      // 2. packing
      // uint8_t* packedmem = aligned_alloc(256, len);
      // packedBint8 = new PackBMatrix<int8_t>(transpose ? matrix_op_t::Transpose : matrix_op_t::NoTranspose,
      //                                       in->shape()[0],
      //                                       in->shape()[1],
      //                                       quantized,
      //                                       transpose ? in->shape()[1] : in->shape()[0]);
      // std::cout << "packing B done" << std::endl;
    }
#endif
  } else {
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

  // std::cout << "B transposed: " << transpose << std::endl;
  // for (int i = 0; i < in->shape().size(); i++) {
  //   std::cout << "size " << i << ": " << in->shape()[i] << std::endl;
  // }
  // // compute statistics for quantization
  // if (in->shape().size() == 2 && (in->shape()[0] >= 32000 || in->shape()[1] >= 32000)) {
  //   float mins[ncol] = {0};
  //   float maxs[ncol] = {0};
  //   float means[ncol] = {0};
  //   float stds[ncol] = {0};
  //   for(int i = 0; i < nrow; i++) {
  //     for(int j = 0; j < ncol; j++) {
  //       float val = !transpose ? in->data()[i * ncol + j] : in->data()[i + nrow * j];
  //       if (val < mins[j])
  //         mins[j] = val;
  //       if (val > maxs[j])
  //         maxs[j] = val;
        
  //       means[j] += val;
  //       stds[j] += val*val;
  //     }
  //   }
  //   for(int j = 0; j < ncol; j++) {
  //       std::cout << mins[j] << ", " << maxs[j] << ", " << means[j] << ", " << stds[j] << std::endl;
  //   }
  // }
}

void GemmPackFp32(marian::Tensor C,
                  const marian::Tensor A,
                  const marian::Tensor B,
                  const marian::Tensor bias,
                  const int64_t m,
                  const int64_t n,
                  int transA) {
  // use int 8 packed gemm
  // if (A->shape().size() == 4 && B->shape().size() == 1 && B->shape()[0] == 1) {
 if (false) {
    // quantize & pack A
    // transformer base wmt 2017
    // float ascale = 7.8/104;
    // int32_t azeropoint = 151;
    // old student de-en
    // float ascale = 14.85/117;
    // int32_t azeropoint = 138;

#if 0
    // compute range
    float min_est=1000000, max_est=-10000000;
    // VSLSSTaskPtr task;
    // MKL_INT task_p, task_n, xstorage;

    // /* Parameters of the task and initialization */
    // task_p = 1;
    // task_n = A->shape().elements();
    // xstorage = VSL_SS_MATRIX_STORAGE_ROWS;
    // min_est = max_est = A->data()[0];
    // /* Create a task */
    // vslsSSNewTask( &task, &task_p, &task_n, &xstorage, (float*)A->data(), 0, 0 );
    // /* Initialize the task parameters */
    // vslsSSEditTask( task, VSL_SS_ED_MIN, &min_est );
    // vslsSSEditTask( task, VSL_SS_ED_MAX, &max_est );
    // /* Compute the minimum and maximum values in observations */
    // vslsSSCompute( task, VSL_SS_MIN|VSL_SS_MAX, VSL_SS_METHOD_FAST );
    // /* Deallocate the task resources */
    
    // vslSSDeleteTask( &task );

    int elem = A->shape().elements();
    float* data = A->data();
    for (int ii = 0; ii < elem; ii++) {
      if (data[ii] < min_est) min_est = data[ii];
      if (data[ii] > max_est) max_est = data[ii];
    }
 
    std::vector<int32_t> row_offset_buf(PackAWithQuantRowOffset<uint8_t>::rowOffsetBufferSize());

    float ascale = (max_est - min_est)/255;
    int32_t azeropoint = (int32_t)(255 - max_est / ascale);
    PackAWithQuantRowOffset<uint8_t> packAN(
        transA ? matrix_op_t::Transpose : matrix_op_t::NoTranspose,
        transA ? k : m,
        transA ? m : k,
        A->data(),
        transA ? m : k,
        nullptr, /*buffer for packed matrix*/
        ascale,
        azeropoint,
        1, /*groups*/
        row_offset_buf.data());

    DoNothing<float, float> doNothingObj{};
    ReQuantizeForFloat<false, QuantizationGranularity::OUT_CHANNEL> outputProcObj(
        doNothingObj,
        ascale,
        bqScale,
        azeropoint,
        bqZeropoint,
        packAN.getRowOffsetBuffer(),
        col_offsets->data(),
        nullptr,
        n);

    // gemm
    fbgemmPacked(
        packAN,
        *packedBint8,
        C->data(),
        (int32_t*)C->data(),
        n,
        outputProcObj,
        0,
        1);

    // std::cout << "lowp gemm: " << std::endl;
    // for (int ii = 0; ii < n; ii++) {
    //   std::cout << C->data()[ii] << std::endl;
    // }
    // std::cout << std::endl;

#endif
  } else {
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

    if (true) {
    // if (A->shape().size() == 4 && B->shape()[0] > 20480000) {
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
    } else {
        fbgemm::cblas_gemm_compute(transA ? matrix_op_t::Transpose : matrix_op_t::NoTranspose,
                          (int)m,
                          A->data(),
                          packedPlaceholder,
                          1,
                          C->data(),
                          0,
                          1);
    }

    // if (B->shape().size() == 1 && B->shape()[0] >= 32000 * 512) {
    //   std::cout << "packed gemm: " << std::endl;
    //   for (int ii = 0; ii < n; ii++) {
    //     std::cout << C->data()[ii] << std::endl;
    //   }
    //   std::cout << std::endl;
    // }

    // return back the original mem
    packedPlaceholder.pmat_ = pmat;
  }

  // std::cout << "A transposed: " << transA << std::endl;
  // for (int i = 0; i < A->shape().size(); i++) {
  //   std::cout << "size " << i << ": " << A->shape()[i] << std::endl;
  // }
  // compute statistics for quantization
  // if (A->shape().size() == 4 && B->shape()[0] > 20480000) {
  //   int bsize = A->shape().elements() / A->shape()[-1];
  //   int hsize = A->shape()[3];
  //   float mins[bsize] = {0};
  //   float maxs[bsize] = {0};
  //   float means[bsize] = {0};
  //   float stds[bsize] = {0};
  //   float* inmem = A->data();
  //   for(int i = 0; i < bsize; i++) {
  //     for(int j = 0; j < hsize; j++) {
  //       float val = !transA ? inmem[i * hsize + j] : inmem[i + bsize * j];
  //       if (val < mins[i])
  //         mins[i] = val;
  //       if (val > maxs[i])
  //         maxs[i] = val;
        
  //       means[i] += val;
  //       stds[i] += val*val;
  //     }
  //   }
  //   for(int i = 0; i < bsize; i++) {
  //       std::cout << mins[i] << ", " << maxs[i] << ", " << means[i] << ", " << stds[i] << std::endl;
  //   }
  // }
}
#else // USE_FBGEMM && MKL_FOUND
void PackFp32(marian::Tensor out,
              const marian::Tensor in,
              bool tranpose,
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
                  const int64_t k,
                  const float beta,
                  const int layout,
                  const int transA,
                  const int transB,
                  size_t idx) {
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

}  // namespace pack
}  // namespace cpu
}  // namespace marian
