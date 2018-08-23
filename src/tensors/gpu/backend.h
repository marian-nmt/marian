#pragma once

#include <cublas_v2.h>
#include <cuda.h>
#include <curand.h>

#include "common/config.h"
#include "tensors/backend.h"

#define CURAND_CALL(x)                                \
  do {                                                \
    if((x) != CURAND_STATUS_SUCCESS) {                \
      printf("Error at %s:%d\n", __FILE__, __LINE__); \
      exit(1);                                        \
    }                                                 \
  } while(0)

namespace marian {
namespace gpu {

class Backend : public marian::Backend {
public:
  Backend(DeviceId deviceId, size_t seed) : marian::Backend(deviceId, seed) {
    setDevice();
    setHandles();
  }

  void setDevice() override { cudaSetDevice(deviceId_.no); }

  void synchronize() override { cudaStreamSynchronize(0); }

  cublasHandle_t getCublasHandle() { return cublasHandle_; }

  curandGenerator_t getCurandGenerator() { return curandGenerator_; }

private:
  cublasHandle_t cublasHandle_;
  curandGenerator_t curandGenerator_;

  void setHandles() {
    cublasHandle_ = create_handle();
    curandGenerator_ = createCurandGenerator();
  }

  curandGenerator_t createCurandGenerator() {
    cudaSetDevice(deviceId_.no);
    curandGenerator_t generator;
    CURAND_CALL(curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(generator, seed_));

    // cudaStream_t stream = 0;
    // CURAND_CALL(curandSetStream(generator, stream));
    // CURAND_CALL(curandDestroyGenerator(generator));
    return generator;
  }

  cublasHandle_t create_handle() {
    cudaSetDevice(deviceId_.no);
    cublasHandle_t cublasHandle;
    cublasCreate(&cublasHandle);
    return cublasHandle;
  }
};
}  // namespace gpu
}  // namespace marian
