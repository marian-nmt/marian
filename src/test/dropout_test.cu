#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <curand.h>
#include <vector>
#include <boost/timer/timer.hpp>
#include <boost/chrono.hpp>

#include "tensors/tensor_allocator.h"
#include "tensors/tensor.h"
#include "kernels/tensor_operators.h"

#include "layers/dropout.h"

#include "kernels/dropout_cudnn.h"

using namespace marian;

int main() {
  int cudaDevice = 0;
  TensorAllocator* params = new TensorAllocator(cudaDevice);

  cublasHandle_t handle = create_handle(cudaDevice);

  int rows = 64;
  int cols = 2048;
  int layers = 64;

  std::cerr << "Number of elements in tensor: " << rows * cols * layers << std::endl;
  int rep = 1000;
  const float prob = 0.5f;

  Tensor dropoutMatrix;
  params->allocate(dropoutMatrix, {rows, cols, layers});

  DropoutGenerator dropout(0);

  cudaStreamSynchronize(0);
  boost::timer::cpu_timer timer;

  for (int i = 0; i < rep;++i) {
    dropout.Generate(dropoutMatrix, prob);

  }

  cudaDeviceSynchronize();

  std::cerr << "DropoutGenerator: " << rep << " repetitions: " << timer.format(5, "%ws") << std::endl;

  Tensor cudnnInTensor, cudnnOutTensor;
  params->allocate(cudnnInTensor, {rows, cols, layers});
  params->allocate(cudnnOutTensor, {rows, cols, layers});

  void* states_;
  void* space_;
  size_t spaceSize_;
  cudnnDropoutDescriptor_t dropDesc_;

  CudnnDropoutPrepare(cudnnInTensor, prob, &dropDesc_, &space_, &spaceSize_, &states_, (size_t)1234);
  cudaStreamSynchronize(0);

  cudaDeviceSynchronize();
  timer.start();
  for (int i = 0; i < rep; ++i) {
    CudnnDropoutForward(dropDesc_, space_, spaceSize_, cudnnInTensor, cudnnOutTensor);
  }

  cudaDeviceSynchronize();
  std::cerr << "CUDNN Dropout: " << rep << " repetitions: " << timer.format(5, "%ws") << std::endl;


  return 0;
}
