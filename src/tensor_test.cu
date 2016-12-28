#include <iostream>

#include "tensors/tensor_allocator.h"
#include "tensors/tensor_gpu.h"
#include "tensor_operators.h"

using namespace marian;

int main() {
  TensorAllocator params = newTensorAllocator<DeviceGPU>();
  //params->allocate(100000000);

  Tensor in1;
  params->allocate(in1, {2,2,5});
  in1->set(2);

  Tensor in2;
  params->allocate(in2, {2,1,5});
  in2->set(3);

  Tensor out;
  params->allocate(out, {2,2,1});

  Tensor sum;
  params->allocate(sum, {2,2,1});


  std::cerr << in1->debug() << std::endl;
  std::cerr << in2->debug() << std::endl;

  Reduce(_1, sum, in2);
  Reduce((_1 * _2), out, in1, in2);


  std::cerr << sum->debug() << std::endl;
  std::cerr << out->debug() << std::endl;


  //std::vector<Tensor> tensors;
  //for (int i = 0; i < 200; ++i) {
  //  std::cerr << i << std::endl;
  //  tensors.emplace_back();
  //  params->allocate(tensors.back(), {784,2048});
  //  std::cerr << tensors.back()->size() << std::endl;
  //  std::cerr << params->capacity() << " " << params->size() << std::endl;
  //}
  //
  //for(int i = 0; i < 200; i++) {
  //  tensors[i]->set(0, 3.14 * i);
  //  std::cerr << tensors[i]->get(0) << std::endl;
  //}

  return 0;
}
