#include <iostream>

#include "tensors/tensor_allocator.h"
#include "tensors/tensor_gpu.h"

using namespace marian;

int main() {
  TensorAllocator params = newTensorAllocator<DeviceGPU>();
  params->allocate(1000);
  std::vector<Tensor> tensors;
  for (int i = 0; i < 20; ++i) {
    Tensor t = params->tensor({1000,1000});
    std::cerr << t->size() << std::endl;
    t->set(0, 3.14 * i);
    tensors.push_back(t);
    std::cerr << params->capacity() << " " << params->size() << std::endl;
  }

  for(int i = 0; i < 20; i++)
    std::cerr << tensors[i]->get(0) << std::endl;

  return 0;
}
