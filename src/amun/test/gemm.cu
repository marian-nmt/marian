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

void Output(const float *in, size_t size)
{
  vector<float> vec(size);

  HANDLE_ERROR( cudaStreamSynchronize(stream));
  HANDLE_ERROR( cudaMemcpyAsync(vec.data(),
                                in,
                                size * sizeof(float),
                                cudaMemcpyDeviceToHost,
                                stream) );
  HANDLE_ERROR( cudaStreamSynchronize(stream));

  // Copy (and print) the result on host memory
  for (size_t i = 0; i < size; ++i) {
    cerr << vec[i] << " " << flush;
  }
  cerr << endl;

}

///////////////////////////////////////////////////////////////////////////

__global__
void gSetElement(const float val, float *arr, size_t ind)
{
  arr[ind] = val;
}

void SetElement(const float &val, float *arr, size_t ind)
{
  HANDLE_ERROR( cudaStreamSynchronize(stream));
  gSetElement<<<1,1,0, stream>>>(val, arr, ind);
  HANDLE_ERROR( cudaStreamSynchronize(stream));
}

///////////////////////////////////////////////////////////////////////////

__global__
void gProd(const float *A,
          const float *B,
          float *C,
          const int m,
          const int k,
          const int n)
{
  extern __shared__ float share[];

  cublasHandle_t handle;
  cublasCreate(&handle);

  int lda=m,ldb=k,ldc=m;

  float alpha = 1.0;
  float beta = 0.0;

  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha,
      A, lda, B, ldb, &beta, C, ldc);
  //cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha,
  //    A, lda, B, ldb, &beta, share, ldc);

  cublasDestroy(handle);
}

///////////////////////////////////////////////////////////////////////////

void GPU_fill_rand(float *A, int nr_rows_A, int nr_cols_A) {
     // Create a pseudo-random number generator
     curandGenerator_t prng;
     curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);

     // Set the seed for the random number generator using the system clock
     curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long) clock());

     // Fill the array with random numbers on the device
     /* curandGenerateUniform(prng, A, nr_rows_A * nr_cols_A); */
}

void gpu_blas_mmul(const float *A,
                   const float *B,
                   float *C,
                   const int m,
                   const int k,
                   const int n)
{
  int lda=m,ldb=k,ldc=m;

  float alpha = 1.0;
  float beta = 0.0;

  std::chrono::time_point<std::chrono::system_clock> start, end;

  // Do the actual multiplication
  start = std::chrono::system_clock::now();

  cudaStreamSynchronize(stream);
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha,
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
  float *d_A, *d_B, *d_C;
  cudaMalloc(&d_A,nr_rows_A * nr_cols_A * sizeof(float));
  cudaMalloc(&d_B,nr_rows_B * nr_cols_B * sizeof(float));
  cudaMalloc(&d_C,nr_rows_C * nr_cols_C * sizeof(float));

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
  float *d_A, *d_B, *d_C;

  // small matrices to eyeball
  nr_rows_A = 2;
  nr_cols_A = 3;
  nr_rows_B = 3;
  nr_cols_B = 4;
  nr_rows_C = 2;
  nr_cols_C = 4;

  // Allocate 3 arrays on GPU
  cudaMalloc(&d_A,nr_rows_A * nr_cols_A * sizeof(float));
  cudaMalloc(&d_B,nr_rows_B * nr_cols_B * sizeof(float));
  cudaMalloc(&d_C,nr_rows_C * nr_cols_C * sizeof(float));

  SetElement(1.0f, d_A, 0);
  SetElement(4.0f, d_A, 1);
  SetElement(2.0f, d_A, 2);
  SetElement(5.0f, d_A, 3);
  SetElement(3.0f, d_A, 4);
  SetElement(6.0f, d_A, 5);

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

  cerr << "testResult1" << endl;
  // Multiply A and B on GPU
  gpu_blas_mmul(d_A, d_B, d_C, nr_rows_A, nr_cols_A, nr_cols_B);
  cerr << "testResult2" << endl;

  Output(d_A, 6);
  Output(d_B, 12);
  Output(d_C, 8);

  cerr << "testResult3" << endl;
  HANDLE_ERROR( cudaMemsetAsync(d_C, 0, 8 * sizeof(float), stream) );
  Output(d_C, 8);

  cerr << "testResult4" << endl;

  // in-kernel multiplication
  int shared = 8 * sizeof(float);

  gProd<<<1,1, shared>>>(d_A, d_B, d_C, nr_rows_A, nr_cols_A, nr_cols_B);
  //gpu_blas_mmul(d_A, d_B, d_C, nr_rows_A, nr_cols_A, nr_cols_B);
  cerr << "testResult5" << endl;
  cudaStreamSynchronize(stream);
  cerr << "testResult6" << endl;

  Output(d_C, 8);
  cerr << "testResult7" << endl;

  //Free GPU memory
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

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

  cublasDestroy(handle);
  HANDLE_ERROR(cudaStreamDestroy(stream));

  return 0;
 }
