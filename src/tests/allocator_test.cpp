#include <boost/timer/timer.hpp>
#include <iostream>
#include <map>
#include <cmath>

#include "3rd_party/exception.h"
#include "tensors/allocator.h"

int main(int argc, char** argv) {
  using namespace marian;

  auto a = New<Allocator>({0, DeviceType::gpu}, 0, 30000, 256);
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
