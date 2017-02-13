#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <curand.h>

#include "tensors/tensor.h"


#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    exit(1);}} while(0)

#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    exit(1);}} while(0)


__global__
void gScalled(float* data, int n, float p) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;

  while (index < n) {
    data[index] = (data[index] < p) / p;
    index += gridDim.x * blockDim.x;
  }
}

namespace marian {

class DropoutGenerator {
  public:
    DropoutGenerator(cudaStream_t stream=0, unsigned long long seed = 1234ULL) {
      CURAND_CALL(curandCreateGenerator(&generator_, CURAND_RNG_PSEUDO_DEFAULT));
      CURAND_CALL(curandSetPseudoRandomGeneratorSeed(generator_, seed));
      CURAND_CALL(curandSetStream(generator_, stream));
    }

    void Generate(Tensor& tensor, float p) {
      Generate(tensor->data(), tensor->size(), p);
    }


    void Generate(float* data, int n, float p) {
      CURAND_CALL(curandGenerateUniform(generator_, data, n));
      int numThreads = std::min(n, 512);
      int numBlocks = n / numThreads + (n % numThreads != 0);

      gScalled<<<numBlocks, numThreads>>>(data, n, p);
    }

    ~DropoutGenerator() {
      CURAND_CALL(curandDestroyGenerator(generator_));
    }

  private:
    curandGenerator_t generator_;

};

}
