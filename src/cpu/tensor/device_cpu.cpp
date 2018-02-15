/* All or part of this file was contributed by Intel under license:
 *   Copyright (C) 2017-2018 Intel Corporation
 *   SPDX-License-Identifier: MIT
 */

#include <algorithm>
#include "3rd_party/exception.h"
#include "tensors/device_cpu.h"

#include <x86intrin.h>

namespace marian {

DeviceCPU::DeviceCPU(size_t device, size_t alignment)
  : data_(nullptr, _mm_free), size_(0), device_(device), alignment_(alignment) {
}

void DeviceCPU::reserve(size_t size) {
  size = align(size);
  UTIL_THROW_IF2(size < size_, "New size must be larger than old size");
  if (size == size_) {
    return;
  }

  uint8_t* data = static_cast<uint8_t*>(_mm_malloc(size, alignment_));
  std::copy(data_.get(), data_.get() + size_, data);

  data_.reset(data);
  size_ = size;
}


}
