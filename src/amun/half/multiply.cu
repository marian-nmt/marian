#include <iostream>
#include <vector>
//#include <cuda.h>
#include <stdio.h>
#include <chrono>
#include <curand.h>
#include <cublas_v2.h>

using namespace std;


///////////////////////////////////////////////////////////////////////////////

cudaStream_t stream;
cublasHandle_t handle;

///////////////////////////////////////////////////////////////////////////////

void HandleError(cudaError_t err, const char *file, int line ) {
  if (err != cudaSuccess) {
    std::cerr << "ERROR: " << cudaGetErrorString(err) << " in " << file << " at line " << line << std::endl;
    exit( EXIT_FAILURE );
  }
}

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

///////////////////////////////////////////////////////////////////////////////
void GPU_fill_rand(float *A, int nr_rows_A, int nr_cols_A) {
     // Create a pseudo-random number generator
     curandGenerator_t prng;
     curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);

     // Set the seed for the random number generator using the system clock
     curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long) clock());

     // Fill the array with random numbers on the device
     /* curandGenerateUniform(prng, A, nr_rows_A * nr_cols_A); */
}

///////////////////////////////////////////////////////////////////////////////

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

  // Do the actual multiplication
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha,
      A, lda, B, ldb, &beta, C, ldc);

}

///////////////////////////////////////////////////////////////////////////////
void testBatchMultiply(int batchSize, int numIter, cublasMath_t mathMode) 
{
  cublasStatus_t stat = cublasSetMathMode(handle, mathMode);
  if (stat != CUBLAS_STATUS_SUCCESS) {
    printf ("cublasSetMathMode failed\n");
    abort();
  }

  // Allocate 3 arrays on CPU
  int nr_rows_A, nr_cols_A, nr_rows_B, nr_cols_B, nr_rows_C, nr_cols_C;

  nr_rows_A = batchSize;
  nr_cols_A = 512;
  nr_rows_B = 512;
  nr_cols_B = 85000;
  nr_rows_C = batchSize;
  nr_cols_C = 85000;

  // Allocate 3 arrays on GPU
  float *d_A, *d_B, *d_C;
  cudaMalloc(&d_A,nr_rows_A * nr_cols_A * sizeof(float));
  cudaMalloc(&d_B,nr_rows_B * nr_cols_B * sizeof(float));
  cudaMalloc(&d_C,nr_rows_C * nr_cols_C * sizeof(float));

  // Fill the arrays A and B on GPU with random numbers
  GPU_fill_rand(d_A, nr_rows_A, nr_cols_A);
  GPU_fill_rand(d_B, nr_rows_B, nr_cols_B);

  cudaStreamSynchronize(stream);

  std::chrono::time_point<std::chrono::system_clock> start, end;
  start = std::chrono::system_clock::now();

  for (size_t i = 0; i < numIter; ++i) {
    // Multiply A and B on GPU
    gpu_blas_mmul(d_A, d_B, d_C, nr_rows_A, nr_cols_A, nr_cols_B);
  }

  cudaStreamSynchronize(stream);
  end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end-start;
  std::cout << "batchSize=" << batchSize 
            << " mathMode=" << mathMode
            << " time: " << elapsed_seconds.count() << endl;
 
  //Free GPU memory
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);  

}

///////////////////////////////////////////////////////////////////////////////

int main()
{
  cerr << "Starting" << endl;

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

  for (int batchSize = 640; batchSize > 0; --batchSize) {
    testBatchMultiply(batchSize, 10000, CUBLAS_DEFAULT_MATH);
    testBatchMultiply(batchSize, 10000, CUBLAS_TENSOR_OP_MATH);
  }

  cublasDestroy(handle);
  HANDLE_ERROR(cudaStreamDestroy(stream));

  cerr << "Finished" << endl;
}
