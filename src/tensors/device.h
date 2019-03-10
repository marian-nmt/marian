#pragma once

#include <cmath>
#include <cstdint>

#include "common/definitions.h"

namespace marian {

class Device {
protected:
  DeviceId deviceId_;

  uint8_t* data_{0};
  size_t size_{0};
  size_t alignment_;

  size_t align(size_t size) {
    return (size_t)(ceil(size / (float)alignment_) * alignment_);
  }

public:
  Device(DeviceId deviceId, size_t alignment = 256)
      : deviceId_(deviceId), data_(0), size_(0), alignment_(alignment) {}

  virtual ~Device(){};

  virtual void reserve(size_t size) = 0;

  virtual uint8_t* data() { return data_; }

  virtual size_t size() { return size_; }

  virtual DeviceId getDeviceId() { return deviceId_; }
};

namespace gpu {
class Device : public marian::Device {
public:
  Device(DeviceId deviceId, size_t alignment = 256)
      : marian::Device(deviceId, alignment) {}

  ~Device();

  void reserve(size_t size) override;
};
}  // namespace gpu

namespace cpu {
class Device : public marian::Device {
public:
  Device(DeviceId deviceId, size_t alignment = 256)
      : marian::Device(deviceId, alignment) {}

  ~Device();

  void reserve(size_t size) override;
};

class WrappedDevice : public marian::Device {
public:
  WrappedDevice(DeviceId deviceId, size_t alignment = 256)
      : marian::Device(deviceId, alignment) {}
  ~WrappedDevice() {}

  void set(uint8_t* data, size_t size) {
    marian::Device::data_ = data;
    marian::Device::size_ = size;
  }

  // doesn't allocate anything, just checks size.
  void reserve(size_t size) override {
    ABORT_IF(size > size_,
             "Requested size {} is larger than pre-allocated size {}",
             size,
             size_);
  }
};

}  // namespace cpu

static inline Ptr<Device> DispatchDevice(DeviceId deviceId,
                                         size_t alignment = 256) {
#ifdef CUDA_FOUND
  if(deviceId.type == DeviceType::gpu)
    return New<gpu::Device>(deviceId, alignment);
  else
    return New<cpu::Device>(deviceId, alignment);
#else
  if(deviceId.type == DeviceType::gpu)
    ABORT("CUDA support not compiled into marian");
  else
    return New<cpu::Device>(deviceId, alignment);
#endif
}
}  // namespace marian
