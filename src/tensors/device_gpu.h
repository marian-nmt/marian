#pragma once

#include <cstdint>

namespace marian {

class DeviceGPU {
private:
  uint8_t* data_;
  size_t size_;
  size_t device_;

public:
  DeviceGPU(size_t device) : data_(0), size_(0), device_(device) {}

  ~DeviceGPU();

  void reserve(size_t size);

  uint8_t* data() { return data_; }

  size_t size() { return size_; }

  size_t getDevice() { return device_; }
};

}