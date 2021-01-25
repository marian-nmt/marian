#pragma once

#include "common/definitions.h"
#include "tensors/rand.h"

namespace marian {

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
};

Ptr<Backend> BackendByDeviceId(DeviceId deviceId, size_t seed);

}  // namespace marian
