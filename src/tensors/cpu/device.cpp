#include "tensors/device.h"
#include "tensors/cpu/aligned.h"
#include <iostream>
namespace marian {
namespace cpu {

Device::~Device() {
  genericFree(data_);
}

void Device::reserve(size_t size) {
  size = align(size);
  ABORT_IF(size < size_ || size == 0,
           "New size must be larger than old size and larger than 0");

  uint8_t *temp = static_cast<uint8_t*>(genericMalloc(alignment_, size));
  if(data_) {
    std::copy(data_, data_ + size_, temp);
    genericFree(data_);
  }
  data_ = temp;
  size_ = size;
}
}  // namespace cpu
}  // namespace marian
