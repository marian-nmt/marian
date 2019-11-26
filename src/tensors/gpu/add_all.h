#pragma once

// This header file provides wrappers around NVidia's reduce_all kernel with our custom aggregation functionality
// This kernel reduces a tensor into a single value. We have modified it to allow for different types of aggregations
// like summing or max etc.
 
#include "tensors/gpu/cuda_helpers.h"
#include "tensors/tensor.h"
#include "tensors/allocator.h"
#include "functional/functional.h"
#include "functional/tensor.h"
#include "tensors/tensor_operators.h"
#include "3rd_party/reduce_all.h" // only works with CUDA >9.0, we are dropping CUDA 8.0 support, also changed in CMakeLists.txt

#include <cuda.h>


namespace marian {

#if COMPILE_FP16
// local overload to determine tensor type
template <> inline Type typeId<half>()  { return Type::float16; }
#endif

template <typename T, typename AccType, class Functor, class AggFunctor, class... Tensors>
void AggregateAll(Ptr<Allocator> allocator,
                  Functor functor, 
                  AccType aggInit,
                  AggFunctor aggFunctor,
                  AccType scale,
                  Tensor out, 
                  const Tensors... tensors) {
  cudaSetDevice(out->getDeviceId().no);

  static_assert(CUDA_VERSION >= 9000, "Marian requires CUDA_VERSION >= 9000 (9.0)");

  constexpr size_t K = sizeof...(Tensors);                         // obtain arity K of tensors...
  functional::Array<functional::Tensor<T>, K> gIns = {tensors...}; // convert to array of K objects of type functional::Tensor<T>
  functional::Shape full = marian::Shape::broadcast({tensors...}); // compute maximal broadcasted shape

  int size = full.elements();
  int threads = (size < MAX_THREADS * 2) ? nextPow2((size + 1) / 2) : MAX_THREADS; // suggested in NVidia example for the all_reduce kernel
  int blocks  = std::min(MAX_BLOCKS, (size + (threads * 2 - 1)) / (threads * 2));  // suggested in NVidia example for the all_reduce kernel

  // The all_reduce kernel by nivida needs to perform multiple passes if the number of blocks needed to perform the reduction is larger than 1.
  // Here we allocate the memory for the intermediate reductions for each block.
  Tensor blockMem;
  if(blocks > 1 || out->type() != typeId<AccType>()) { // if the out tensor does not have elementType AccType we need to allocate and convert later
    MemoryPiece::PtrType temporaryMemory;
    if(allocator) {
      temporaryMemory = allocator->alloc<AccType>(blocks);
    } else { // @TODO: get rid of this branch
      uint8_t* temporaryMemoryPtr = 0;
      CUDA_CHECK(cudaMalloc(&temporaryMemoryPtr, sizeof(AccType) * blocks)); 
      temporaryMemory = MemoryPiece::New(temporaryMemoryPtr, sizeof(AccType) * blocks); // @TODO: consider implementing MemoryPiece::cudaMalloc<T>(size) for managed memory
    }
    blockMem = TensorBase::New(temporaryMemory,
                               Shape({blocks}), 
                               typeId<AccType>(), 
                               out->getBackend());
    blockMem->set(aggInit); // set temporary memory to aggInit
  }
  else {            // we are reducing into a single element now and the type matches, just use out as memory
    blockMem = out; // do not set final output memory as we might be summing gradients... needs to be handled outside this function
  }

  functional::Tensor<AccType> gBlockMem = blockMem;
  reduceSinglePass<T, AccType>(functor, aggInit, aggFunctor, scale, full, /*out=*/gBlockMem, /*in=*/gIns, threads, blocks);  // First pass reduction into intermediate memory

  // If we actually needed more than one block to perform the first pass reduction, recursively run a second pass reduction over block memory until block memory has size 1.
  if(blocks > 1) {
    using namespace functional;
    auto identity = _1; // transformation was done in first pass, hence only identity
    AggregateAll<AccType, AccType>(allocator, identity, aggInit, aggFunctor, scale, out, /*in=*/blockMem); // Reducing AccType in AccType now (meta-reduction)
  } else if(out->type() != typeId<AccType>()) { // it's only a single block, but we need to convert to different type, as mentioned above
    CopyCast(out, blockMem);
  }

  if(blockMem != out) {
    // Free temporary memory whether allocated in allocator or via cudaMalloc
    if(allocator)
      allocator->free(blockMem->memory());
    else if(blockMem->memory()->data())
      CUDA_CHECK(cudaFree(blockMem->memory()->data()));
  }
} 

// Aggregates all values into a single tensor and returns the value of that tensor as a float
// This does a GPU to CPU memory copy via TensorBase::scalar().
// Used currently only for L2Norm computation
template <typename T, typename AccType, class Functor, class AggFunctor, class... Tensors>
AccType AggregateAllAndReturn(Ptr<Allocator> allocator, 
                              Functor functor, 
                              AccType aggInit,
                              AggFunctor aggFunctor,
                              AccType scale,
                              const Tensors... tensors) {
  MemoryPiece::PtrType temporaryMemory;
  if(allocator) {
    temporaryMemory = allocator->alloc<AccType>(1);
  } else { // @TODO: get rid of this branch
    uint8_t* temporaryMemoryPtr = 0;
    CUDA_CHECK(cudaMalloc(&temporaryMemoryPtr, sizeof(AccType))); 
    temporaryMemory = MemoryPiece::New(temporaryMemoryPtr, sizeof(AccType));
  }

  std::tuple<Tensors...> in(tensors...);

  // Create a temporary tensor of size 1 to reduce into
  auto out = TensorBase::New(temporaryMemory, 
                             Shape({1}), 
                             typeId<AccType>(), 
                             std::get<0>(in)->getBackend());
  out->set(aggInit); // init to aggInit

  AggregateAll<T, AccType>(allocator, functor, aggInit, aggFunctor, scale, out, tensors...);

  AccType outScalar = out->template scalar<AccType>(); // convert to float also if other underlying type
  
  if(allocator)
    allocator->free(out->memory());
  else if(out->memory()->data()) // @TODO: get rid of this branch
    CUDA_CHECK(cudaFree(out->memory()->data()));

  return outScalar;
} 

}