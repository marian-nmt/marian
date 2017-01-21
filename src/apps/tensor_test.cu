#include <iostream>
#include <boost/timer/timer.hpp>

#include "tensors/tensor_allocator.h"
#include "tensors/tensor_gpu.h"
#include "kernels/tensor_operators.h"

using namespace marian;

int main() {
  TensorAllocator params = newTensorAllocator<DeviceGPU>();

  Tensor in;
  params->allocate(in, {4096, 2048});
  in->set(0.01);


  float norm = L2Norm(in);

  std::cerr << in->debug() << std::endl;
  std::cerr << norm << std::endl;

  //std::cerr << L2Norm(in) << std::endl;


  return 0;
}
