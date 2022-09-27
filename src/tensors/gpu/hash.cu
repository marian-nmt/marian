#include "tensors/gpu/add_all.h"
#include "functional/operators.h"
// clang-format on

#include <cstdint>

#if COMPILE_FP16
#include <cuda_fp16.h>
#endif

namespace marian {
namespace gpu {

// cpu-side conversion of float to uint32_t via bit-wise cast
uint32_t f2u(float f32) {
  uint32_t u32;
  std::memcpy(&u32, &f32, 4);
  return u32;
}

// cpu-side conversion of uint32_t to float via bit-wise cast
float u2f(uint32_t u32) {
  float f32;
  std::memcpy(&f32, &u32, 4);
  return f32;
}

// Computes a murmur3-ish hash value for a Marian tensor.
uint32_t hashTensor(Tensor tensor, uint32_t seed, Ptr<Allocator> allocator) {
  // we first accumulate into single value via a binary mumurhash3-like operator, 
  // see functional/operators.h for details.
  using namespace functional;
  uint32_t h = 0;
  if(tensor->type() == Type::float32)
    h = f2u(AggregateAllAndReturn<float, float>(allocator, _1, u2f(seed), murmur(_1, _2), 1, tensor));
#if COMPILE_FP16
  else if(tensor->type() == Type::float16)
    // internally, a half value gets cast to a float value before hashing or combining. These is the same
    // mechanics as for summing where we cast to a larger type for better precision.
    h = f2u(AggregateAllAndReturn<half,  float>(allocator, _1, u2f(seed), murmur(_1, _2), 1, tensor));
#endif
  else
    ABORT("Hashing of tensors not supported for type {}", tensor->type());
  
  // finalization according to murmurhash3 implementation
  uint32_t len = (uint32_t)tensor->size();
  h ^= len;
  h ^= h >> 16;
  h *= 0x85ebca6b;
  h ^= h >> 13;
  h *= 0xc2b2ae35;
  h ^= h >> 16;
  return h;
}

} // namespace gpu
} // namespace marian