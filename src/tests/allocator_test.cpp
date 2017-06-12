#include <boost/timer/timer.hpp>
#include <iostream>
#include <map>
#include <cmath>

#include "3rd_party/exception.h"
#include "tensors/allocator.h"
#include "tensors/device_gpu.h"

class DeviceCPU {
private:
  uint8_t* data_;
  size_t size_;
  size_t alignment_;

public:
  DeviceCPU(size_t device, size_t alignment = 256)
   : data_(0), size_(0), alignment_(alignment) {}

  ~DeviceCPU() { delete[] data_; }

  size_t align(size_t size) {
    return ceil(size / (float)alignment_) * alignment_;
  }

  void reserve(size_t size) {
    size = align(size);
    UTIL_THROW_IF2(size < size_, "New size must be larger than old size");

    if(data_) {
      // Allocate memory by going through host memory
      uint8_t *temp = new uint8_t[size_];
      std::copy(data_, data_ + size_, temp);
      std::fill(data_, data_ + size_, 0);
      delete[] data_;
      data_ = new uint8_t[size];
      std::copy(temp, temp + size_, data_);
      delete[] temp;
    } else {
      data_ = new uint8_t[size];
    }

    size_ = size;
  }

  uint8_t* data() { return data_; }

  size_t size() { return size_; }

  size_t getDevice() { return 0; }
};

int main(int argc, char** argv) {
  using namespace marian;

  auto a = New<Allocator<DeviceGPU>>(0, 0, 30000, 256);
  std::cerr << "Size: " << a->size() << std::endl;

  auto mem1 = a->alloc<int>(100000);
  std::cerr << "Size: " << a->size() << std::endl;
  std::cerr << "mem1: " << *mem1 << std::endl;

  //a->throwAtReallocation(true);

  auto mem2 = a->alloc(1000000);
  std::cerr << "Size: " << a->size() << std::endl;
  std::cerr << "mem2: " << *mem2 << std::endl;

  a->free(mem1);

  auto mem3 = a->alloc(100000);
  std::cerr << "mem2: " << *mem2 << std::endl;
  std::cerr << "mem3: " << *mem3 << std::endl;
  std::cerr << "Size: " << a->size() << std::endl;

  return 0;
}
