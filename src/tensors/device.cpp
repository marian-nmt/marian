#include <iostream>
#include "tensors/device.h"

namespace marian {
namespace cpu {
  
  Device::~Device() {
    delete[] data_;
  }
  
  void Device::reserve(size_t size) {
    size = align(size);
    ABORT_IF(size < size_, "New size must be larger than old size");

    if(data_) {
      uint8_t *temp = new uint8_t[size_];
      std::copy(data_, data_ + size_, temp);
      delete[] data_;
      data_ = temp;
    } else {
      data_ = new uint8_t[size];
    }

    size_ = size;
  }
  
}
}
