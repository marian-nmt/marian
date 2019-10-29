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

// @TODO: in the future this should pobably become a fully fledged CudaInfo class with many attributes
struct CudaCompute {
  int major;
  int minor;
};

class Backend : public marian::Backend {
private:
  void setCudaComputeCapability() {
    cudaDeviceGetAttribute(&compute_.major, cudaDevAttrComputeCapabilityMajor, (int)deviceId_.no);
    cudaDeviceGetAttribute(&compute_.minor, cudaDevAttrComputeCapabilityMinor, (int)deviceId_.no);
  }

public:
  Backend(DeviceId deviceId, size_t seed) : marian::Backend(deviceId, seed) {
    setDevice();
    cublasCreate(&cublasHandle_);
    cusparseCreate(&cusparseHandle_);
    setCudaComputeCapability();
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

  CudaCompute getCudaComputeCapability() { return compute_; }

  // for CPU, sets to use optimized code for inference.
  // for GPU, this is invalid. for gpu, isOptimized() function always returns false.
  void setOptimized(bool optimize) override {
    LOG_ONCE(info, "setOptimized() not supported for GPU_{}", optimize);
  }
  
  bool isOptimized() override {
    return false;
  }

  // for CPU, selects different GEMM types for the inference.
  // for GPU, there's no gemm type. so, it does nothing.
  void setGemmType(std::string gemmType) override {
    LOG_ONCE(info, "setGemmType() not supported for GPU_{}", gemmType);
  }
  GemmType getGemmType() override {
    LOG_ONCE(info, "getGemmType() not supported for GPU");
    return GemmType::Auto;
  }

private:
  cublasHandle_t cublasHandle_;
  cusparseHandle_t cusparseHandle_;
  CudaCompute compute_;
};
}  // namespace gpu
}  // namespace marian
