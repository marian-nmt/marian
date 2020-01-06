#pragma once

// This header file provides wrappers around NVidia's reduce_all kernel with our custom aggregation functionality
// This kernel reduces a tensor into a single value. We have modified it to allow for different types of aggregations
// like summing or max etc.

#include "tensors/gpu/cuda_helpers.h"
#include "tensors/tensor.h"
#include "tensors/allocator.h"
#include "functional/tensor.h"
#include "tensors/tensor_operators.h"

namespace marian {

// These function declarations are repeated as template specialization with variadic template arguments does not seem to work.
// Here I am just creating version for 1, 2, and 3 arguments. To be extended if required.
template <typename T, typename AccType, class Functor, class AggFunctor>
void AggregateAll(Ptr<Allocator> allocator,
                  Functor functor, 
                  AccType aggInit,
                  AggFunctor aggFunctor,
                  AccType scale,
                  Tensor out, 
                  const Tensor in1);

template <typename T, typename AccType, class Functor, class AggFunctor>
void AggregateAll(Ptr<Allocator> allocator,
                  Functor functor, 
                  AccType aggInit,
                  AggFunctor aggFunctor,
                  AccType scale,
                  Tensor out, 
                  const Tensor in1, 
                  const Tensor in2);

template <typename T, typename AccType, class Functor, class AggFunctor>
void AggregateAll(Ptr<Allocator> allocator,
                  Functor functor, 
                  AccType aggInit,
                  AggFunctor aggFunctor,
                  AccType scale,
                  Tensor out, 
                  const Tensor in1, 
                  const Tensor in2, 
                  const Tensor in3);

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