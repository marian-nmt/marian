#pragma once

#include "common/config.h"
#include "tensors/backend.h"
#include "tensors/gpu/cuda_helpers.h"

#include <cublas_v2.h>
#include <cuda.h>
#include <curand.h>

namespace marian {
namespace gpu {

class Backend : public marian::Backend {
public:
  Backend(DeviceId deviceId, size_t seed) : marian::Backend(deviceId, seed) {
    setDevice();
    setHandles();
  }

  void setDevice() override { cudaSetDevice((int)deviceId_.no); }

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
    cudaSetDevice((int)deviceId_.no);
    curandGenerator_t generator;
    CURAND_CHECK(curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(generator, seed_));

    // cudaStream_t stream = 0;
    // CURAND_CHECK(curandSetStream(generator, stream));
    // CURAND_CHECK(curandDestroyGenerator(generator));
    return generator;
  }

  cublasHandle_t create_handle() {
    cudaSetDevice((int)deviceId_.no);
    cublasHandle_t cublasHandle;
    cublasCreate(&cublasHandle);
    return cublasHandle;
  }
};
}  // namespace gpu
}  // namespace marian
