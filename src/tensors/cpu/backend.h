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
  GemmType gemmType_{GemmType::Float32};
  float quantizeRange_{0.f};

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
    else if (gemmType == "float32")     gemmType_ = GemmType::Float32;
#if USE_FBGEMM
    else if (gemmType == "packed16")    gemmType_ = GemmType::FbFp16Packed;
    else if (gemmType.find("packed8") == 0)  gemmType_ = GemmType::FbInt8Packed;
#endif // USE_FBGEMM
    else ABORT("Unknown GEMM type - '{}'", gemmType);
  }
  GemmType getGemmType() override { return gemmType_; }
  // for CPU, sets quantization range of weight matrices for the inference.
  // for GPU, there's no quantization. so, it does nothing.
  void setQuantizeRange(float range) override { quantizeRange_ = range; }
  float getQuantizeRange() override { return quantizeRange_; }
};

}  // namespace cpu
}  // namespace marian
