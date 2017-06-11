#include <boost/timer/timer.hpp>
#include <iostream>
#include <map>

#include "tensors/allocator.h"
#include "tensors/device_gpu.h"

class DeviceCPU {
private:
  uint8_t* data_;
  size_t size_;

public:
  DeviceCPU(size_t device) : data_(0), size_(0) {}

  ~DeviceCPU() { delete[] data_; }

  void reserve(size_t size) {
    //UTIL_THROW_IF2(size < size_, "New size must be larger than old size");

    if(data_) {
      // Allocate memory by going through host memory
      uint8_t *temp = new uint8_t[size_];
      std::copy(data_, data_ + size_, temp);
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

  size_t num = 10000;
  size_t len = 10000;

  auto a = New<Allocator<DeviceCPU>>(0, num * len, 1000);
  std::cerr << a->size() << std::endl;

  {
    boost::timer::auto_cpu_timer t;
    std::vector<Ptr<MemoryPiece>> mem(num);
    for(int i = 0; i < 1000000; ++i) {
      int idx = rand() % mem.size();
      mem[idx] = a->alloc(len);
      std::fill(mem[idx]->data(), mem[idx]->data() + mem[idx]->size(), (uint8_t)(i % 256));
    }
  }

  return 0;
}
