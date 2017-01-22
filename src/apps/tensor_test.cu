#include <iostream>
#include <boost/timer/timer.hpp>

#include "tensors/tensor_allocator.h"
#include "tensors/tensor_gpu.h"
#include "kernels/tensor_operators.h"

using namespace marian;

int main() {
  TensorAllocator params = newTensorAllocator<DeviceGPU>();

  Tensor in1;
  params->allocate(in1, {3, 10, 7});
  in1->set(2);

  Tensor in2;
  params->allocate(in2, {3, 10, 1});
  std::vector<float> v(30, 0);
  for(int i = 0; i < 10; ++i)
    v[i] = 1;
  in2->set(v);

  Tensor out1;
  params->allocate(out1, {3, 1, 7});
  out1->set(0);

  Tensor out2;
  params->allocate(out2, {3, 1, 7});
  out2->set(0);

  Add(_1 * _2, out1, in1, in2);
  Reduce(_1 * _2, out2, in1, in2);

  std::cerr << in1->debug() << std::endl;
  std::cerr << in2->debug() << std::endl;
  std::cerr << out1->debug() << std::endl;
  std::cerr << out2->debug() << std::endl;

  return 0;
}
