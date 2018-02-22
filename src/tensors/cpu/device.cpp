#include <iostream>
#include "tensors/device.h"

namespace marian {
namespace cpu {

  Device::~Device() {
    delete[] data_;
    data_ = nullptr;
    size_ = 0;
  }

  void Device::reserve(size_t size) {
    size = align(size);
    ABORT_IF(size < size_ || size == 0, "New size must be larger than old size and larger than 0");

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
