#include <cstdlib>
#include <cublas_v2.h>
#include <iostream>
#include <vector>
#include <curand.h>
#include <cuda_fp16.h>
#include <chrono>

using namespace std;

cudaStream_t stream;
cublasHandle_t handle;

void HandleError(cudaError_t err, const char *file, int line ) {
  if (err != cudaSuccess) {
    std::cerr << "ERROR: " << cudaGetErrorString(err) << " in " << file << " at line " << line << std::endl;
    exit( EXIT_FAILURE );
  }
}

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

///////////////////////////////////////////////////////////////////////////

#define __fp16 half

__fp16 uint16_as_fp16 (uint16_t a)
{
    __fp16 res;
#if defined (__cplusplus)
    memcpy (&res, &a, sizeof (res));
#else /* __cplusplus */
    volatile union {
        __fp16 f;
        uint16_t i;
    } cvt;
    cvt.i = a;
    res = cvt.f;
#endif /* __cplusplus */
    return res;
}

uint32_t fp32_as_uint32 (float a)
{
    uint32_t res;
#if defined (__cplusplus)
    memcpy (&res, &a, sizeof (res));
#else /* __cplusplus */
    volatile union {
        float f;
        uint32_t i;
    } cvt;
    cvt.f = a;
    res = cvt.i;
#endif /* __cplusplus */
    return res;
}

/* host version of device function __float2half_rn() */
__fp16 float2half_rn (float a)
{
    uint32_t ia = fp32_as_uint32 (a);
    uint16_t ir;

    ir = (ia >> 16) & 0x8000;
    if ((ia & 0x7f800000) == 0x7f800000) {
        if ((ia & 0x7fffffff) == 0x7f800000) {
            ir |= 0x7c00; /* infinity */
        } else {
            ir = 0x7fff; /* canonical NaN */
        }
    } else if ((ia & 0x7f800000) >= 0x33000000) {
        int shift = (int)((ia >> 23) & 0xff) - 127;
        if (shift > 15) {
            ir |= 0x7c00; /* infinity */
        } else {
            ia = (ia & 0x007fffff) | 0x00800000; /* extract mantissa */
            if (shift < -14) { /* denormal */
                ir |= ia >> (-1 - shift);
                ia = ia << (32 - (-1 - shift));
            } else { /* normal */
                ir |= ia >> (24 - 11);
                ia = ia << (32 - (24 - 11));
                ir = ir + ((14 + shift) << 10);
            }
            /* IEEE-754 round to nearest of even */
            if ((ia > 0x80000000) || ((ia == 0x80000000) && (ir & 1))) {
                ir++;
            }
        }
    }
    return uint16_as_fp16 (ir);
}

///////////////////////////////////////////////////////////////////////////

__global__
void gSetElement(const float val, half *arr, size_t ind)
{
  half h = __float2half(val);
  arr[ind] = h;
}

void SetElement(const float &val, half *arr, size_t ind)
{
  HANDLE_ERROR( cudaStreamSynchronize(stream));
  gSetElement<<<1,1,0, stream>>>(val, arr, ind);
  HANDLE_ERROR( cudaStreamSynchronize(stream));
}

///////////////////////////////////////////////////////////////////////////

__global__
void gSetElement(const float val1, const float val2, half2 *arr, size_t ind)
{
  half2 h =  __floats2half2_rn(val1, val2);
  arr[ind] = h;
}

void SetElement(const float &val1, const float &val2, half2 *arr, size_t ind)
{
  HANDLE_ERROR( cudaStreamSynchronize(stream));
  gSetElement<<<1,1,0, stream>>>(val1, val2, arr, ind);
  HANDLE_ERROR( cudaStreamSynchronize(stream));
}

///////////////////////////////////////////////////////////////////////////
__global__
void gHalf2Float(float *out, const half *in, size_t size)
{
  for (size_t i = 0; i <  size; ++i) {
    const half &h = in[i];
    float f = __half2float(h);
    out[i] = f;
  }
}

void Output(const half *in, size_t size)
{
  vector<float> vec(size);
  float *d_out;
  HANDLE_ERROR( cudaMalloc(&d_out, size * sizeof(float)) );

  HANDLE_ERROR( cudaStreamSynchronize(stream));
  gHalf2Float<<<1,1,0, stream>>>(d_out, in, size);
  HANDLE_ERROR( cudaMemcpyAsync(vec.data(),
                                d_out,
                                size * sizeof(float),
                                cudaMemcpyDeviceToHost,
                                stream) );
  HANDLE_ERROR( cudaStreamSynchronize(stream));

  HANDLE_ERROR(cudaFree(d_out));

  // Copy (and print) the result on host memory
  for (size_t i = 0; i < size; ++i) {
    cerr << vec[i] << " ";
  }
  cerr << endl;

}


///////////////////////////////////////////////////////////////////////////

void GPU_fill_rand(half *A, int nr_rows_A, int nr_cols_A) {
     // Create a pseudo-random number generator
     curandGenerator_t prng;
     curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);

     // Set the seed for the random number generator using the system clock
     curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long) clock());

     // Fill the array with random numbers on the device
     /* curandGenerateUniform(prng, A, nr_rows_A * nr_cols_A); */
}

void gpu_blas_mmul(const half *A,
                  const half *B,
                  half *C,
                  const int m,
                  const int k,
                  const int n)
{
  int lda=m,
     ldb=k,
     ldc=m;

  half alpha = float2half_rn(1.0);
  half beta = float2half_rn(0.0);

  std::chrono::time_point<std::chrono::system_clock> start, end;

  // Do the actual multiplication
  start = std::chrono::system_clock::now();
  cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha,
        A, lda, B, ldb, &beta, C, ldc);
  cudaStreamSynchronize(stream);

  end = std::chrono::system_clock::now();

  std::chrono::duration<double> elapsed_seconds = end-start;
  std::time_t end_time = std::chrono::system_clock::to_time_t(end);
  std::cout << "finished computation at " << std::ctime(&end_time)
           << "elapsed time: " << elapsed_seconds.count() << "s\n";
}

void testSpeed()
{
  // Allocate 3 arrays on CPU
  int nr_rows_A, nr_cols_A, nr_rows_B, nr_cols_B, nr_rows_C, nr_cols_C;

  // for simplicity we are going to use square arrays
  nr_rows_A = 12;
  nr_cols_A = 500;
  nr_rows_B = 500;
  nr_cols_B = 90000;
  nr_rows_C = 12;
  nr_cols_C = 90000;

  // Allocate 3 arrays on GPU
  half *d_A, *d_B, *d_C;
  cudaMalloc(&d_A,nr_rows_A * nr_cols_A * sizeof(half));
  cudaMalloc(&d_B,nr_rows_B * nr_cols_B * sizeof(half));
  cudaMalloc(&d_C,nr_rows_C * nr_cols_C * sizeof(half));

  // Fill the arrays A and B on GPU with random numbers
  GPU_fill_rand(d_A, nr_rows_A, nr_cols_A);
  GPU_fill_rand(d_B, nr_rows_B, nr_cols_B);

  // Multiply A and B on GPU
  gpu_blas_mmul(d_A, d_B, d_C, nr_rows_A, nr_cols_A, nr_cols_B);

  // Copy (and print) the result on host memory

  //Free GPU memory
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
}

void testResult()
{
  // Allocate 3 arrays on CPU
  int nr_rows_A, nr_cols_A, nr_rows_B, nr_cols_B, nr_rows_C, nr_cols_C;

  // Allocate 3 arrays on GPU
  half *d_A, *d_B, *d_C;

  // small matrices to eyeball
  nr_rows_A = 2;
  nr_cols_A = 3;
  nr_rows_B = 3;
  nr_cols_B = 4;
  nr_rows_C = 2;
  nr_cols_C = 4;

  // Allocate 3 arrays on GPU
  cudaMalloc(&d_A,nr_rows_A * nr_cols_A * sizeof(half));
  cudaMalloc(&d_B,nr_rows_B * nr_cols_B * sizeof(half));
  cudaMalloc(&d_C,nr_rows_C * nr_cols_C * sizeof(half));

  SetElement(1, d_A, 0);
  SetElement(4, d_A, 1);
  SetElement(2, d_A, 2);
  SetElement(5, d_A, 3);
  SetElement(3, d_A, 4);
  SetElement(6, d_A, 5);

  SetElement(7, d_B, 0);
  SetElement(11, d_B, 1);
  SetElement(15, d_B, 2);
  SetElement(8, d_B, 3);
  SetElement(12, d_B, 4);
  SetElement(16, d_B, 5);
  SetElement(9, d_B, 6);
  SetElement(13, d_B, 7);
  SetElement(17, d_B, 8);
  SetElement(10, d_B, 9);
  SetElement(14, d_B, 10);
  SetElement(18, d_B, 11);

  // Multiply A and B on GPU
  gpu_blas_mmul(d_A, d_B, d_C, nr_rows_A, nr_cols_A, nr_cols_B);

  Output(d_A, 6);
  Output(d_B, 12);
  Output(d_C, 8);

  //Free GPU memory
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

}

void testResultHalf2()
{
  cerr << "half=" << sizeof(half) << endl;
  cerr << "half2=" << sizeof(half2) << endl;

  // Allocate 3 arrays on CPU
  int nr_rows_A, nr_cols_A, nr_rows_B, nr_cols_B, nr_rows_C, nr_cols_C;

  // Allocate 3 arrays on GPU
  half2 *d_A, *d_B, *d_C;

  // small matrices to eyeball
  nr_rows_A = 2;
  nr_cols_A = 3;
  nr_rows_B = 3;
  nr_cols_B = 4;
  nr_rows_C = 2;
  nr_cols_C = 4;

  // Allocate 3 arrays on GPU
  cudaMalloc(&d_A,nr_rows_A * nr_cols_A * sizeof(half2) / 2);
  cudaMalloc(&d_B,nr_rows_B * nr_cols_B * sizeof(half2) / 2);
  cudaMalloc(&d_C,nr_rows_C * nr_cols_C * sizeof(half2) / 2);

  SetElement(1, 4, d_A, 0);
  SetElement(2, 5, d_A, 1);
  SetElement(3, 6, d_A, 2);

  SetElement(7, 11, d_B, 0);
  SetElement(15, 8, d_B, 1);
  SetElement(12, 16, d_B, 2);
  SetElement(9, 13, d_B, 3);
  SetElement(17, 10, d_B, 4);
  SetElement(14, 18, d_B, 5);

  // Multiply A and B on GPU
  gpu_blas_mmul((half*) d_A, (half*) d_B, (half*) d_C, nr_rows_A, nr_cols_A, nr_cols_B);

  Output((half*)d_A, 6);
  Output((half*)d_B, 12);
  Output((half*)d_C, 8);

}

int main() {
  HANDLE_ERROR( cudaStreamCreate(&stream));

  cublasStatus_t stat;
  stat = cublasCreate(&handle);
  if (stat != CUBLAS_STATUS_SUCCESS) {
    printf ("cublasCreate initialization failed\n");
    abort();
  }

  stat = cublasSetStream(handle, stream);
  if (stat != CUBLAS_STATUS_SUCCESS) {
    printf ("cublasSetStream initialization failed\n");
    abort();
  }

  testSpeed();
  testResult();
  testResultHalf2();

  cublasDestroy(handle);
  HANDLE_ERROR(cudaStreamDestroy(stream));

  return 0;
}
