#pragma once

#include <cublas_v2.h>
#include <cuda.h>
#include <curand.h>

#include "common/config.h"
#include "graph/backend.h"

#define CURAND_CALL(x)                                \
  do {                                                \
    if((x) != CURAND_STATUS_SUCCESS) {                \
      printf("Error at %s:%d\n", __FILE__, __LINE__); \
      exit(1);                                        \
    }                                                 \
  } while(0)

namespace marian {

class BackendGPU : public Backend {
public:
  void setDevice(size_t device) { cudaSetDevice(device); }

  void setHandles(size_t device, size_t seed) {
    cublasHandle_ = create_handle(device);
    curandGenerator_ = createCurandGenerator(device, Config::seed);
  }

  cublasHandle_t getCublasHandle() { return cublasHandle_; }

  curandGenerator_t getCurandGenerator() { return curandGenerator_; }

private:
  cublasHandle_t cublasHandle_;
  curandGenerator_t curandGenerator_;

  curandGenerator_t createCurandGenerator(size_t device, size_t seed) {
    cudaSetDevice(device);
    curandGenerator_t generator;
    CURAND_CALL(curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(generator, seed));

    // cudaStream_t stream = 0;
    // CURAND_CALL(curandSetStream(generator, stream));
    // CURAND_CALL(curandDestroyGenerator(generator));
    return generator;
  }

  cublasHandle_t create_handle(size_t device) {
    cudaSetDevice(device);
    cublasHandle_t cublasHandle;
    cublasCreate(&cublasHandle);
    return cublasHandle;
  }
};
}
