#include <iostream>
#include <boost/timer/timer.hpp>

#include "tensors/tensor_allocator.h"
#include "tensors/tensor_gpu.h"
#include "tensor_operators.h"

using namespace marian;

int main() {
  TensorAllocator params = newTensorAllocator<DeviceGPU>();
  //params->allocate(100000000);

  Tensor in1;
  params->allocate(in1, {80, 85000, 25});
  in1->set(1);

  Tensor in2;
  params->allocate(in2, {80, 85000, 25});
  in2->set(1);

  Tensor sum;
  params->allocate(sum, {1, 85000, 1});

  boost::timer::cpu_timer timer;

  for(int i = 0; i < 1000; ++i) {
    Reduce(_1 * _2, sum, in1, in2);
  }

  std::cout << timer.format(5, "%ws") << std::endl;

  //std::cerr << in1->debug() << std::endl;
  //std::cerr << sum->debug() << std::endl;

  return 0;
}
