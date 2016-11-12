#include <iostream>

#include "tensors/tensor_allocator.h"
#include "tensors/tensor_gpu.h"
#include "tensor_operators.h"

using namespace marian;

int main() {
  TensorAllocator params = newTensorAllocator<DeviceGPU>();
  //params->allocate(100000000);

  std::vector<Tensor> tensors;
  for (int i = 0; i < 200; ++i) {
    std::cerr << i << std::endl;
    tensors.emplace_back();
    params->allocate(tensors.back(), {784,2048});
    std::cerr << tensors.back()->size() << std::endl;
    std::cerr << params->capacity() << " " << params->size() << std::endl;
  }

  for(int i = 0; i < 200; i++) {
    tensors[i]->set(0, 3.14 * i);
    std::cerr << tensors[i]->get(0) << std::endl;
  }

  return 0;
}
