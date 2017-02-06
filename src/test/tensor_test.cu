#include <iostream>
#include <boost/timer/timer.hpp>

#include "tensors/tensor_allocator.h"
#include "tensors/tensor.h"
#include "kernels/tensor_operators.h"
#include "kernels/thrust_functions.h"
#include "common/logging.h"

using namespace marian;

int main() {
  Logger memory{stderrLogger("memory", "[%Y-%m-%d %T] [memory] %v")};

  Ptr<TensorAllocator> params = New<TensorAllocator>(0);

  cublasHandle_t handle = create_handle(0);

  int words = 64;
  int batch = 128;
  int hidden = 4096;

  Tensor mappedState;
  params->allocate(mappedState, {batch, hidden, 1});
  mappedState->set(0.001);

  Tensor mappedContext;
  params->allocate(mappedContext, {batch, hidden, words});
  mappedContext->set(0.001);

  Tensor va;
  params->allocate(va, {hidden, 1});
  va->set(0.001);

  Tensor out1;
  params->allocate(out1, {batch, hidden, words});
  out1->set(0);

  Tensor gMappedState;
  params->allocate(gMappedState, {batch, hidden, 1});
  gMappedState->set(0);

  Tensor gMappedContext;
  params->allocate(gMappedContext, {batch, hidden, words});
  gMappedContext->set(0.001);

  Tensor gVa;
  params->allocate(gVa, {hidden, 1});
  va->set(0.001);

  Tensor gOut1;
  params->allocate(gOut1, {batch, hidden, words});
  out1->set(0);

  Tensor out2;
  params->allocate(out2, {batch, 1, words});
  out2->set(0);

  boost::timer::cpu_timer timer;
  for(int i = 0; i < 5000; ++i) {
    Element(_1 = Tanh(_2 + _3), out1, mappedState, mappedContext);
    Prod(handle, out2, out1, va, false, false, 0);
    Prod(handle, gOut1, out2, va, false, true, 1.0f);
    Prod(handle, gVa, out1, out2, true, false, 1.0f);
    Add(_1 * (1.f - (_2 *_2)), gMappedState, out1, out1);
    Add(_1 * (1.f - (_2 *_2)), gMappedContext, out1, out1);
    cudaStreamSynchronize(0);

    if(i % 100 == 0)
      std::cout << "." << std::flush;
  }
  std::cout << timer.format(5, "%ws") << std::endl;

  boost::timer::cpu_timer timer2;
  for(int i = 0; i < 5000; ++i) {
    Att(out2, mappedContext, mappedState, va);
    AttBack(gMappedContext, gMappedState, gVa,
        mappedContext, mappedState, va, out2);
    cudaStreamSynchronize(0);
    if(i % 100 == 0)
      std::cout << "." << std::flush;
  }
  std::cout << timer2.format(5, "%ws") << std::endl;

  return 0;
}
