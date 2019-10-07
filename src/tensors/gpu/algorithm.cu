#include "tensors/gpu/algorithm.h"

// clang-format off
#include "tensors/tensor_operators.h"
#include "tensors/gpu/cuda_helpers.h"
// clang-format on

namespace marian {
namespace gpu {

template <typename T>
void copy(Ptr<Backend> backend, const T* begin, const T* end, T* dest) {
  CUDA_CHECK(cudaSetDevice(backend->getDeviceId().no));
  CudaCopy(begin, end, dest);
  CUDA_CHECK(cudaStreamSynchronize(0));
}

// clang-format off
template void copy<int8_t>(Ptr<Backend>, const int8_t*, const int8_t*, int8_t*);
template void copy<int16_t>(Ptr<Backend>, const int16_t*, const int16_t*, int16_t*);
template void copy<int32_t>(Ptr<Backend>, const int32_t*, const int32_t*, int32_t*);
template void copy<int64_t>(Ptr<Backend>, const int64_t*, const int64_t*, int64_t*);
template void copy<uint8_t>(Ptr<Backend>, const uint8_t*, const uint8_t*, uint8_t*);
template void copy<uint16_t>(Ptr<Backend>, const uint16_t*, const uint16_t*, uint16_t*);
template void copy<uint32_t>(Ptr<Backend>, const uint32_t*, const uint32_t*, uint32_t*);
template void copy<uint64_t>(Ptr<Backend>, const uint64_t*, const uint64_t*, uint64_t*);
template void copy<char>(Ptr<Backend>, const char*, const char*, char*);
template void copy<float16>(Ptr<Backend>, const float16*, const float16*, float16*);
template void copy<float>(Ptr<Backend>, const float*, const float*, float*);
template void copy<double>(Ptr<Backend>, const double*, const double*, double*);
// clang-format on

template <typename T>
__global__ void gFill(T* d_in, int size, T val) {
  //auto blocks = gridDim.x;
  auto threadsPerBlock = blockDim.x;
  //for(int bid = 0; bid < size; bid += threadsPerBlock * blocks) {
    int index = /*bid +*/ threadIdx.x + threadsPerBlock * blockIdx.x;
    if(index < size) {
      d_in[index] = val;
    }
  //}
}

template <typename T>
void fill(Ptr<Backend> backend, T* begin, T* end, T value) {
  int size = end - begin;
  if (size == 0)
    return;
  CUDA_CHECK(cudaSetDevice(backend->getDeviceId().no));
  int threadsPerBlock = std::min(MAX_THREADS, size);
  int blocks = (size / threadsPerBlock) + (size % threadsPerBlock != 0); // @TODO: (size+threadsPerBlock-1)/threadsPerBlock or CeilDiv(a,b)
  gFill<<<blocks, threadsPerBlock>>>(begin, size, value);
  CUDA_CHECK(cudaStreamSynchronize(0));
}

template <>
void fill<float16>(Ptr<Backend> backend, float16* begin, float16* end, float16 value) {
  int size = end - begin;
  if (size == 0)
    return;
#if COMPILE_FP16
  CUDA_CHECK(cudaSetDevice(backend->getDeviceId().no));
  int threadsPerBlock = std::min(MAX_THREADS, size);
  int blocks = (size / threadsPerBlock) + (size % threadsPerBlock != 0); // @TODO: (size+threadsPerBlock-1)/threadsPerBlock or CeilDiv(a,b)
  gFill<<<blocks, threadsPerBlock>>>((__half*)begin, size, (__half)value);
  CUDA_CHECK(cudaStreamSynchronize(0));
#else
   ABORT("FP16 not supported with current hardware or CUDA version");
#endif
}

template void fill<bool>(Ptr<Backend>, bool*, bool*, bool);
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
  CUDA_CHECK(cudaSetDevice(backend->getDeviceId().no));
  ABORT("no SetSparse");
  // gpu::SetSparse(data, keys, values);
  CUDA_CHECK(cudaStreamSynchronize(0));
}

template <typename T>
__global__ void gSwap(T* d_v1, T* d_v2, int size) {
  auto threadsPerBlock = blockDim.x;
  int index = threadIdx.x + threadsPerBlock * blockIdx.x;
  if(index < size) {
    T temp = d_v1[index];
    d_v1[index] = d_v2[index];
    d_v2[index] = temp;
  }
}

template <typename T>
void swap_ranges(Ptr<Backend> backend, T* begin, T* end, T* dest) {
  int size = end - begin;
  if (size == 0)
    return;

  CUDA_CHECK(cudaSetDevice(backend->getDeviceId().no));
  int threadsPerBlock = std::min(MAX_THREADS, size);
  int blocks = (size / threadsPerBlock) + (size % threadsPerBlock != 0); // @TODO: (size+threadsPerBlock-1)/threadsPerBlock or CeilDiv(a,b)
  gSwap<<<blocks, threadsPerBlock>>>(begin, dest, size);
  CUDA_CHECK(cudaStreamSynchronize(0));
}

template <>
void swap_ranges<float16>(Ptr<Backend> backend, float16* begin, float16* end, float16* dest) {
  int size = end - begin;
  if (size == 0)
    return;

#if COMPILE_FP16
  CUDA_CHECK(cudaSetDevice(backend->getDeviceId().no));
  int threadsPerBlock = std::min(MAX_THREADS, size);
  int blocks = (size / threadsPerBlock) + (size % threadsPerBlock != 0); // @TODO: (size+threadsPerBlock-1)/threadsPerBlock or CeilDiv(a,b)
  gSwap<<<blocks, threadsPerBlock>>>((__half*)begin, (__half*)dest, size);
  CUDA_CHECK(cudaStreamSynchronize(0));
#else
  ABORT("FP16 not supported with current hardware or CUDA version");
#endif
}

// clang-format off
template void swap_ranges<char>(Ptr<Backend>, char*, char*, char*);
template void swap_ranges<int8_t>(Ptr<Backend>, int8_t*, int8_t*, int8_t*);
template void swap_ranges<int16_t>(Ptr<Backend>, int16_t*, int16_t*, int16_t*);
template void swap_ranges<int32_t>(Ptr<Backend>, int32_t*, int32_t*, int32_t*);
template void swap_ranges<int64_t>(Ptr<Backend>, int64_t*, int64_t*, int64_t*);

template void swap_ranges<uint8_t>(Ptr<Backend>, uint8_t*, uint8_t*, uint8_t*);
template void swap_ranges<uint16_t>(Ptr<Backend>, uint16_t*, uint16_t*, uint16_t*);
template void swap_ranges<uint32_t>(Ptr<Backend>, uint32_t*, uint32_t*, uint32_t*);
template void swap_ranges<uint64_t>(Ptr<Backend>, uint64_t*, uint64_t*, uint64_t*);

template void swap_ranges<float>(Ptr<Backend>, float*, float*, float*);
template void swap_ranges<double>(Ptr<Backend>, double*, double*, double*);
// clang-format on

}  // namespace gpu
}  // namespace marian
