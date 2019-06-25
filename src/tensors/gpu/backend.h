#pragma once

#include "common/config.h"
#include "tensors/backend.h"  // note: this is one folder up
#include "tensors/gpu/cuda_helpers.h"
#include "common/logging.h"

#include <cublas_v2.h>
#include <cuda.h>
#include <curand.h>
#include <cusparse.h>

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

  // for CPU & inference only, sets to use optimized code for inference. Does nothing for GPU.
  void setOptimized(bool optimize) override { ABORT("Not supported for GPU_{}", optimize); }
  bool isOptimized() override {
    ABORT("Not supported for GPU");
    return false;
  }
  // for CPU only, selects different GEMM types for the inference. Does nothing for GPU.
  void setGemmType(std::string gemmType) override { ABORT("Not supported for GPU_{}", gemmType); }
  GemmType getGemmType() override {
    ABORT("Not supported for GPU");
    return GemmType::Auto;
  }

private:
  cublasHandle_t cublasHandle_;
  cusparseHandle_t cusparseHandle_;
};
}  // namespace gpu
}  // namespace marian
