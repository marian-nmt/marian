#pragma once

#include <functional>
#include <random>

#include "common/config.h"
#include "tensors/backend.h"

namespace marian {
namespace cpu {

class Backend : public marian::Backend {
protected:
  bool optimized_{false};
  GemmType gemmType_{GemmType::Auto};

public:
  Backend(DeviceId deviceId, size_t seed) : marian::Backend(deviceId, seed) {}
  void setDevice() override {}
  void synchronize() override {}

  // for CPU & inference only, sets to use optimized code for inference. Does nothing for GPU.
  void setOptimized(bool optimize) { optimized_ = optimize; }
  bool isOptimized() override { return optimized_; }
  // for CPU only, selects different GEMM types for the inference. Does nothing for GPU.
  void setGemmType(std::string gemmType) override {
    if(gemmType == "auto") gemmType_ = GemmType::Auto;
    else if(gemmType == "mkl") gemmType_ = GemmType::Mkl;
    else if(gemmType == "int16") gemmType_ = GemmType::Int16;
    else if(gemmType == "packed") gemmType_ = GemmType::PackedFb;
    else if(gemmType == "int8") gemmType_ = GemmType::Int8Fb;
    else ABORT("Unknow GEMM type");
  }
  GemmType getGemmType() override { return gemmType_; }
};
}  // namespace cpu
}  // namespace marian
