#include <iostream>

#include "tensors/tensor_allocator.h"
#include "tensors/tensor_gpu.h"
#include "tensor_operators.h"

using namespace marian;

int main() {
  TensorAllocator params = newTensorAllocator<DeviceGPU>();

  std::vector<Tensor> tensors;
  for (int i = 0; i < 20; ++i) {
    tensors.emplace_back();
    params->allocate(tensors.back(), {1000,1000});
    std::cerr << tensors.back()->size() << std::endl;
    tensors.back()->set(0, 3.14 * i);
    std::cerr << params->capacity() << " " << params->size() << std::endl;
  }

  for(int i = 0; i < 20; i++)
    std::cerr << tensors[i]->get(0) << std::endl;

  return 0;
}
