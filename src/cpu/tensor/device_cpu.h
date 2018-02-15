/* All or part of this file was contributed by Intel under license:
 *   Copyright (C) 2017-2018 Intel Corporation
 *   SPDX-License-Identifier: MIT
 */

#pragma once

#include <memory>
#include <cstdint>

namespace marian {

using std::size_t;
using std::uint8_t;

class DeviceCPU {
private:
  std::unique_ptr<uint8_t, void (*)(void*)> data_;
  size_t size_;
  size_t device_;
  size_t alignment_;

  size_t align(size_t size) const {
    return ((size + (alignment_-1)) / alignment_) * alignment_;
  }

public:
  DeviceCPU(size_t device, size_t alignment=2097152);

  void reserve(size_t size);

  uint8_t* data() { return data_.get(); }

  size_t size() { return size_; }

  size_t getDevice() { return device_; }
};

}
