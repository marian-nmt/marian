#include "tensors/gpu/algorithm.h"

// clang-format off
#include "tensors/tensor_operators.h"
#include "tensors/gpu/cuda_helpers.h"
// clang-format on

namespace marian {
namespace gpu {

template <typename T>
void copy(Ptr<Backend> backend, const T* begin, const T* end, T* dest) {
  CUDA_CHECK(cudaSetDevice(backend->getDevice().no));
  CudaCopy(begin, end, dest);
  CUDA_CHECK(cudaStreamSynchronize(0));
}

template void copy<int8_t>(Ptr<Backend>, const int8_t*, const int8_t*, int8_t*);
template void copy<int16_t>(Ptr<Backend>, const int16_t*, const int16_t*, int16_t*);
template void copy<int32_t>(Ptr<Backend>, const int32_t*, const int32_t*, int32_t*);
template void copy<int64_t>(Ptr<Backend>, const int64_t*, const int64_t*, int64_t*);

template void copy<uint8_t>(Ptr<Backend>, const uint8_t*, const uint8_t*, uint8_t*);
template void copy<uint16_t>(Ptr<Backend>, const uint16_t*, const uint16_t*, uint16_t*);
template void copy<uint32_t>(Ptr<Backend>, const uint32_t*, const uint32_t*, uint32_t*);
template void copy<uint64_t>(Ptr<Backend>, const uint64_t*, const uint64_t*, uint64_t*);

template void copy<float>(Ptr<Backend>, const float*, const float*, float*);
template void copy<double>(Ptr<Backend>, const double*, const double*, double*);


template <typename T>
__global__ void gFill(T* d_in, int size, T val) {
  for(int bid = 0; bid < size; bid += blockDim.x * gridDim.x) {
    int index = bid + threadIdx.x + blockDim.x * blockIdx.x;
    if(index < size) {
      d_in[index] = val;
    }
  }
}

template <typename T>
void fill(Ptr<Backend> backend, T* begin, T* end, T value) {
  CUDA_CHECK(cudaSetDevice(backend->getDevice().no));
  int size = end - begin;
  int threads = std::min(512, size);
  int blocks = (size / threads) + (size % threads != 0);
  gFill<<<blocks, threads>>>(begin, size, value);
  CUDA_CHECK(cudaStreamSynchronize(0));
}

template void fill<int8_t>(Ptr<Backend>, int8_t*, int8_t*, int8_t);
template void fill<int16_t>(Ptr<Backend>, int16_t*, int16_t*, int16_t);
template void fill<int32_t>(Ptr<Backend>, int32_t*, int32_t*, int32_t);
template void fill<int64_t>(Ptr<Backend>, int64_t*, int64_t*, int64_t);
template void fill<uint8_t>(Ptr<Backend>, uint8_t*, uint8_t*, uint8_t);
template void fill<uint16_t>(Ptr<Backend>, uint16_t*, uint16_t*, uint16_t);
template void fill<uint32_t>(Ptr<Backend>, uint32_t*, uint32_t*, uint32_t);
template void fill<uint64_t>(Ptr<Backend>, uint64_t*, uint64_t*, uint64_t);

template void fill<float>(Ptr<Backend>, float*, float*, float);
template void fill<double>(Ptr<Backend>, double*, double*, double);

void setSparse(Ptr<Backend> backend,
               const std::vector<size_t>& keys,
               const std::vector<float>& values,
               float* data) {
  CUDA_CHECK(cudaSetDevice(backend->getDevice().no));
  ABORT("no SetSparse");
  // gpu::SetSparse(data, keys, values);
  CUDA_CHECK(cudaStreamSynchronize(0));
}
}
}
