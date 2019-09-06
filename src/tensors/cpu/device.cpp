#include "tensors/device.h"
#include <iostream>

#ifdef _WIN32
#include <malloc.h>
#endif
#include <stdlib.h>

namespace marian {
namespace cpu {

// allocate function for tensor reserve() below. 
// Needed for AVX512, while not available on all compilers. It seems clang
// does not have aligned_alloc for all cstlib versions. If AVX512 is not used
// a simple malloc is probably fine. 
// Should generate a runtime error otherwise as we have a check in the AVX512 
// functions which tests for alignment. 
#ifdef _WIN32
#define MALLOC(size) _aligned_malloc(size, alignment_)
#elif __GNUC__
#define MALLOC(size) aligned_alloc(alignment_, size)
#else
#define MALLOC(size) malloc(size)
#endif

#ifdef _WIN32
#define FREE(ptr) _aligned_free(ptr)
#else
#define FREE(ptr) free(ptr)
#endif

Device::~Device() {
  FREE(data_);
}

void Device::reserve(size_t size) {
  size = align(size);
  ABORT_IF(size < size_ || size == 0,
           "New size must be larger than old size and larger than 0");

  if(data_) {
    uint8_t *temp = static_cast<uint8_t*>(MALLOC(size));
    std::copy(data_, data_ + size_, temp);
    FREE(data_);
    data_ = temp;
  } else {
    data_ = static_cast<uint8_t*>(MALLOC(size));
  }
  size_ = size;
}
}  // namespace cpu
}  // namespace marian
