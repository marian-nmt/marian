#pragma once

#include "common/config.h"
#include "tensors/backend.h" // note: this is one folder up
#include "tensors/gpu/cuda_helpers.h"

#include <cuda.h>
#include <cublas_v2.h>
#include <curand.h>

namespace marian {
namespace gpu {

class Backend : public marian::Backend {
public:
  Backend(DeviceId deviceId, size_t seed) : marian::Backend(deviceId, seed) {
    setDevice();
    setHandles();
  }

  ~Backend() {
    setDevice();
    cublasDestroy(cublasHandle_);
  }

  void setDevice() override { cudaSetDevice((int)deviceId_.no); }

  void synchronize() override { cudaStreamSynchronize(0); }

  cublasHandle_t getCublasHandle() { return cublasHandle_; }

private:
  cublasHandle_t cublasHandle_;

  void setHandles() {
    cublasHandle_ = create_handle();
  }

  cublasHandle_t create_handle() {
    cublasHandle_t cublasHandle;
    cublasCreate(&cublasHandle);
    return cublasHandle;
  }
};
}  // namespace gpu
}  // namespace marian
