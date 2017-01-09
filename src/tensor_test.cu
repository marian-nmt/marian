#include <iostream>
#include <boost/timer/timer.hpp>

#include "tensors/tensor_allocator.h"
#include "tensors/tensor_gpu.h"
#include "tensor_operators.h"

using namespace marian;

int main() {
  TensorAllocator params = newTensorAllocator<DeviceGPU>();
  //params->allocate(100000000);

  std::vector<float> in1v(4096);
  std::vector<float> in2v(4096);

  for(int i = 0; i < 4096; ++i) {
    if(i < 2048) {
      in1v[i] = 1.5;
      in2v[i] = -1;
    }
    else {
      in1v[i] = 1.4;
      in2v[i] = 1;
    }
  }

  Tensor in1;
  params->allocate(in1, {1, 2048, 2});
  in1->set(in1v);

  Tensor in2;
  params->allocate(in2, {1, 2048, 2});
  in2->set(in2v);

  Tensor sum;
  params->allocate(sum, {1, 2048, 2});
  sum->set(0);


  Element(_1 = (0.f + _2) * _3, sum, in1, in2);


  std::cerr << in1->debug() << std::endl;
  std::cerr << in2->debug() << std::endl;
  std::cerr << sum->debug() << std::endl;

  return 0;
}
