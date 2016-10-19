#include <iostream>

#include "tensors/tensor_allocator.h"
#include "tensors/tensor_gpu.h"
#include "tensor_operators.h"

using namespace marian;

int main() {
  TensorAllocator params = newTensorAllocator<DeviceGPU>();

  auto t = params->tensor({9, 11});
  t->set(1);
  //std::cerr << t->debug() << std::endl;

  auto s = params->tensor({1, 11});
  s->set(0);
  Sum(s, t, 0, true);
  std::cerr << s->debug() << std::endl;


  //std::vector<Tensor> tensors;
  //for (int i = 0; i < 20; ++i) {
  //  Tensor t = params->tensor({1000,1000});
  //  std::cerr << t->size() << std::endl;
  //  t->set(0, 3.14 * i);
  //  tensors.push_back(t);
  //  std::cerr << params->capacity() << " " << params->size() << std::endl;
  //}
  //
  //for(int i = 0; i < 20; i++)
  //  std::cerr << tensors[i]->get(0) << std::endl;

  return 0;
}
