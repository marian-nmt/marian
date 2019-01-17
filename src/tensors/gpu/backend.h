#pragma once

#include "common/config.h"
#include "tensors/backend.h" // note: this is one folder up
#include "tensors/gpu/cuda_helpers.h"

#include <cuda.h>
#include <cublas_v2.h>
#include <cusparse.h>
#include <curand.h>

namespace marian {
namespace gpu {

class Backend : public marian::Backend {
public:
  Backend(DeviceId deviceId, size_t seed) : marian::Backend(deviceId, seed) {
    setDevice();
    cublasCreate(&cublasHandle_);
    cusparseCreate(&cusparseHandle_);
  }

  ~Backend() {
    setDevice();
    cusparseDestroy(cusparseHandle_);
    cublasDestroy(cublasHandle_);
  }

  void setDevice() override { cudaSetDevice((int)deviceId_.no); }

  void synchronize() override { cudaStreamSynchronize(0); }

  cublasHandle_t getCublasHandle() { return cublasHandle_; }
  cusparseHandle_t getCusparseHandle() { return cusparseHandle_; }

private:
  cublasHandle_t cublasHandle_;
  cusparseHandle_t cusparseHandle_;
};
}  // namespace gpu
}  // namespace marian
