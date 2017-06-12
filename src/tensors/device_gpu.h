#pragma once

#include <cstdint>
#include <cmath>

namespace marian {

class DeviceGPU {
private:
  uint8_t* data_;
  size_t size_;
  size_t device_;
  size_t alignment_;

  size_t align(size_t size) {
    return ceil(size / (float)alignment_) * alignment_;
  }

public:
  DeviceGPU(size_t device, size_t alignment=256)
   : data_(0), size_(0),
     device_(device),
     alignment_(alignment) {}

  ~DeviceGPU();

  void reserve(size_t size);

  uint8_t* data() { return data_; }

  size_t size() { return size_; }

  size_t getDevice() { return device_; }
};

}