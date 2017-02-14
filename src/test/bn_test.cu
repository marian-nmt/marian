#include <iostream>
#include <cuda.h>
#include <string>
#include <vector>
#include <cmath>

#include <common/shape.h>

#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    exit(1);}} while(0)

using namespace marian;

__global__ void gLNormalization(float* out, const float* in, const float* alpha, const float* beta,
                                const Shape outShape, float eps=0.00001) {
  int rows = outShape[0];
  int cols = outShape[1];

  extern __shared__ float _share[];

  for (int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if (j < rows) {
      float* so = out + j * cols;
      const float* sp = in + j * cols;

      float* _sum = _share + blockDim.x;
      _sum[threadIdx.x] = sp[threadIdx.x]; // mask
      for (int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if (id < cols) {
          _sum[threadIdx.x] = sp[id];
        }
      }
      __syncthreads();
      int len = blockDim.x;
      while(len != 1) {
        __syncthreads();
        int skip = (len + 1) >> 1;
        if (threadIdx.x < (len >> 1)) {
             _sum[threadIdx.x] += _sum[threadIdx.x + skip];
        }
        len = (len + 1) >> 1;
      }
      __syncthreads();
      float mean = _sum[0] / cols;
      __syncthreads();

      float* _sqSum = _share + blockDim.x;

      _sqSum[threadIdx.x] = 0.0;
      for (int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols) {
          float ex = sp[id] - mean;
          so[id] = ex;
          _sqSum[threadIdx.x] += ex * ex;
        }
      }
      __syncthreads();
      len = blockDim.x;
      while(len != 1) {
        __syncthreads();
        int skip = (len + 1) >> 1;
        if(threadIdx.x < (len >> 1))
          _sqSum[threadIdx.x] += _sqSum[threadIdx.x + skip];
        len = (len + 1) >> 1;
      }
      __syncthreads();
      float sigma = sqrtf(eps + (_sqSum[0] / cols));
      __syncthreads();

      for (int tid = 0; tid < cols; tid += blockDim.x) {
        int id = tid + threadIdx.x;
        if(id < cols) {
          so[id] = alpha[id] * (so[id] / sigma) + beta[id];
        }
      }
    }
  }
}

void cpuNormalization(const std::vector<float>& inData,
                      const std::vector<float>& alpha,
                      const std::vector<float>& beta,
                      float eps, int cols,
                      std::vector<float>& outData) {
  int rows = inData.size() / cols;

  for (int rowIdx = 0; rowIdx < rows; ++rowIdx) {
    float sum = 0.0f;
    for (size_t colIdx = 0; colIdx < cols; ++colIdx) {
      sum += inData[rowIdx * cols + colIdx];
    }

    float mean = sum / cols;

    float sqSum = 0.0f;
    for (size_t colIdx = 0; colIdx < cols; ++colIdx) {
      sqSum += (inData[rowIdx * cols + colIdx] - mean) * (inData[rowIdx * cols + colIdx] - mean);
    }

    float sigma = std::sqrt(eps + (sqSum / cols));

    outData.resize(inData.size());

    for (size_t i = 0; i < cols; ++i) {
      outData[rowIdx * cols + i] = alpha[i] * ((inData[rowIdx * cols + i] - mean) / sigma) + beta[i];
    }
  }
}


void gpuNormalization(const std::vector<float>& in,
                      const std::vector<float>& alpha,
                      const std::vector<float>& beta,
                      float eps, int cols,
                      std::vector<float>& out) {
  float* d_indata;
  float* d_outdata;

  float* d_alpha;
  float* d_beta;


  CUDA_CALL( cudaMalloc(&d_indata, in.size() * sizeof(float)) );
  CUDA_CALL( cudaMalloc(&d_outdata, in.size() * sizeof(float)) );
  CUDA_CALL( cudaMalloc(&d_alpha, alpha.size() * sizeof(float)) );
  CUDA_CALL( cudaMalloc(&d_beta, beta.size() * sizeof(float)) );

  CUDA_CALL( cudaMemcpy(d_indata, in.data(), in.size() * sizeof(float), cudaMemcpyHostToDevice) );
  CUDA_CALL( cudaMemcpy(d_alpha, alpha.data(), alpha.size() * sizeof(float), cudaMemcpyHostToDevice) );
  CUDA_CALL( cudaMemcpy(d_beta, beta.data(), beta.size() * sizeof(float), cudaMemcpyHostToDevice) );

  int numThreads = std::min(cols, 512);

  int rows = in.size() / cols;
  int numBlocks = rows;
  int shared = numThreads * sizeof(float) * 2;

  Shape shape({rows, cols, 1, 1});

  gLNormalization<<<numBlocks, numThreads, shared>>>(d_outdata, d_indata, d_alpha, d_beta, shape, eps);

  out.resize(in.size());
  cudaDeviceSynchronize();

  CUDA_CALL( cudaMemcpy(out.data(), d_outdata, in.size() * sizeof(float), cudaMemcpyDeviceToHost) );

  CUDA_CALL( cudaFree(d_indata) );
  CUDA_CALL( cudaFree(d_outdata) );
}

void test_normalization(std::vector<float>& inData,
                        std::vector<float>& alpha,
                        std::vector<float>& beta,
                        int cols, float eps) {
  std::vector<float> cpuOut;
  std::vector<float> gpuOut;

  cpuNormalization(inData, alpha, beta, eps, cols, cpuOut);
  gpuNormalization(inData, alpha, beta, eps, cols, gpuOut);

  for (int i = 0; i < inData.size(); ++i) {
    std::cerr << "CPU: " << cpuOut[i] << " :: GPU: " << gpuOut[i] << std::endl;
  }
}

int main() {
  const float eps = 0.f;
  std::vector<float> in({1,2,3,4,5,6,7,8});
  std::vector<float> alpha({2,4,6,8});
  std::vector<float> beta({2,4,6,8});


  test_normalization(in, alpha, beta, 4, eps);

  return 0;
}


