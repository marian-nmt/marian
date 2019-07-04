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
  void setOptimized(bool optimize) override { optimized_ = optimize; }
  bool isOptimized() override { return optimized_; }
  // for CPU only, selects different GEMM types for the inference. Does nothing for GPU.
  void setGemmType(std::string gemmType) override {
    if      (gemmType == "auto")        gemmType_ = GemmType::Auto;
    else if (gemmType == "mklfp32")     gemmType_ = GemmType::MklFp32;
    else if (gemmType == "intrinint16") gemmType_ = GemmType::IntrinInt16;
#if USE_FBGEMM
    else if (gemmType == "fp16packed")  gemmType_ = GemmType::FbFp16Packed;
    else if (gemmType == "int8packed")  gemmType_ = GemmType::FbInt8Packed;
#endif // USE_FBGEMM
    else ABORT("Unknown GEMM type - '{}'", gemmType);
  }
  GemmType getGemmType() override { return gemmType_; }
};
}  // namespace cpu
}  // namespace marian
