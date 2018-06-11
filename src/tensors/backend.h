#pragma once

#include "common/definitions.h"

namespace marian {

class Backend {
protected:
  DeviceId deviceId_;
  size_t seed_;
  float clip_{0.f};

public:
  Backend(DeviceId deviceId, size_t seed) : deviceId_(deviceId), seed_(seed) {}

  virtual DeviceId getDevice() { return deviceId_; };
  virtual void setDevice() = 0;
  virtual void synchronize() = 0;

  virtual void setClip(float clip) {
    clip_ = clip;
  }

  float getClip() {
    return clip_;
  }
};

Ptr<Backend> BackendByDevice(DeviceId deviceId, size_t seed);
}
