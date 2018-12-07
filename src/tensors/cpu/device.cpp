#include "tensors/device.h"
#include <iostream>

#ifdef _WIN32
#include <malloc.h>
#endif
#include <stdlib.h>

namespace marian {
namespace cpu {

Device::~Device() {
  free(data_);
  data_ = nullptr;
  size_ = 0;
}

#ifdef _WIN32
#define MALLOC(size) _aligned_alloc(size, alignment_)
#elif __GNUC__
#define MALLOC(size) aligned_alloc(alignment_, size)
#else
#define MALLOC(size) malloc(size)
#endif

void Device::reserve(size_t size) {
  size = align(size);
  ABORT_IF(size < size_ || size == 0,
           "New size must be larger than old size and larger than 0");

  if(data_) {
    uint8_t *temp = static_cast<uint8_t*>(MALLOC(size));
    std::copy(data_, data_ + size_, temp);
    free(data_);
    data_ = temp;
  } else {
    data_ = static_cast<uint8_t*>(MALLOC(size));
  }
  size_ = size;
}
}  // namespace cpu
}  // namespace marian
