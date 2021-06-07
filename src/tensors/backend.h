#pragma once

#include "common/definitions.h"
#include "tensors/rand.h"

namespace marian {

// GEMM type enum
typedef enum {
  Auto = 0,            // auto tuning between available GEMMs
  Float32 = 1,         // MKL based GEMM, fp32
  FbFp16Packed = 10,   // FBGEMM based fp16 GEMM with packing
  FbInt8Packed = 11    // FBGEMM based int8 GEMM with packing
} GemmType;

class Backend {
protected:
  DeviceId deviceId_;
  size_t seed_;
  Ptr<RandomGenerator> randomGenerator_;
  
public:
  Backend(DeviceId deviceId, size_t seed)
      : deviceId_(deviceId), seed_(seed), randomGenerator_(createRandomGenerator(seed, deviceId)) {}
  virtual ~Backend() {};
  virtual DeviceId getDeviceId() { return deviceId_; };
  virtual Ptr<RandomGenerator> getRandomGenerator() { return randomGenerator_; }

  // for GPU only, calls cudaSetDevice, does nothing on CPU. Maybe change name.
  virtual void setDevice() = 0;
  virtual void synchronize() = 0;

  // for CPU, sets to use optimized code for inference.
  // for GPU, this is invalid. for gpu, isOptimized() function always returns false.
  virtual void setOptimized(bool optimize) = 0;
  virtual bool isOptimized() = 0;
  // for CPU, selects different GEMM types for the inference.
  // for GPU, there's no gemm type. so, it does nothing.
  virtual void setGemmType(std::string gemmType) = 0;
  virtual GemmType getGemmType() = 0;
  // for CPU, sets quantization range of weight matrices for the inference.
  // for GPU, there's no quantization. so, it does nothing.
  virtual void setQuantizeRange(float range) = 0;
  virtual float getQuantizeRange() = 0;
};

Ptr<Backend> BackendByDeviceId(DeviceId deviceId, size_t seed);

}  // namespace marian
