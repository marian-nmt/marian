#include <algorithm>
#include <chrono>
#include <iomanip>
#include <string>
#include <cstdio>
#include <boost/timer/timer.hpp>

#include "marian.h"
#include "mnist.h"
#include "trainer.h"
#include "models/feedforward.h"

#include "tensors/tensor.h"
#include "tensors/tensor_gpu.h"
#include "tensors/tensor_allocator.h"

using namespace marian;
using namespace keywords;
using namespace data;
using namespace models;

int main(int argc, char** argv) {
  cudnnHandle_t cudnnHandle;
  cudnnCreate(&cudnnHandle);

  cudnnRNNDescriptor_t rnnDesc;
  cudnnCreateRNNDescriptor(&rnnDesc);

  int rows = 1;
  int cols = 500;

  int hiddenSize = 1024;
  int seqLength = 10;

  TensorAllocator alloc = newTensorAllocator<DeviceGPU>();
  std::vector<Tensor> tensors(seqLength);
  for(int i = 0; i < seqLength; ++i) {
    alloc->allocate(tensors[i], {rows, cols});
    tensors[i]->set(i);
  }

  cudnnTensorDescriptor_t xDesc[seqLength];
  for(int i = 0; i < seqLength; ++i) {
    xDesc[i] = std::static_pointer_cast<TensorGPU>(tensors[i])->cudnn();
  }

  // dropout
  float dropout = 0.0;
  cudnnDropoutDescriptor_t dropDesc;
  size_t statesSize;
  void* states;
  cudnnDropoutGetStatesSize(cudnnHandle, &statesSize);
  cudaMalloc((void**)&states, statesSize);
  cudnnCreateDropoutDescriptor(&dropDesc);
  cudnnSetDropoutDescriptor(
    dropDesc,
    cudnnHandle,
    dropout,
    (void*)states,
    statesSize,
    1234);
  // dropout

  cudnnSetRNNDescriptor(
    rnnDesc,
    hiddenSize,
    1,
    dropDesc,
    CUDNN_LINEAR_INPUT,
    CUDNN_UNIDIRECTIONAL,
    CUDNN_RNN_TANH,
    CUDNN_DATA_FLOAT);

  size_t workSpaceSize;
  cudnnGetRNNWorkspaceSize(
    cudnnHandle,
    rnnDesc,
    seqLength,
    xDesc,
    &workSpaceSize);

  std::cerr << workSpaceSize << std::endl;

  size_t trainingReserveSize;
  cudnnGetRNNTrainingReserveSize(
    cudnnHandle,
    rnnDesc,
    seqLength,
    xDesc,
    &trainingReserveSize);

  std::cerr << trainingReserveSize << std::endl;

  cudnnDestroyRNNDescriptor(rnnDesc);

  return 0;
}
