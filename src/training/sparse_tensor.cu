#include <curand.h>
#include <curand_kernel.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <memory>

#include "tensors/tensor.h"
#include "tensors/tensor_operators.h"
#include "training/sparse_tensor.h"

namespace marian {

// TODO: create actual sparse tensor class. This one is just minimal
__global__ void gScatterAdd(float* denseData,
                            float* sparseData,
                            int* sparseIndices,
                            int denseSize,
                            int sparseSize,
                            int offset) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if(idx >= sparseSize)
    return;
  if(sparseIndices[idx] + offset >= 0
     && sparseIndices[idx] + offset < denseSize)
    denseData[sparseIndices[idx] + offset] += sparseData[idx];
}

__global__ void gFindSubtensor(int* indices,
                               int size,
                               int targetStart,
                               int targetEnd,
                               int* resultStart,
                               int* resultEnd) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if(idx >= size)
    return;

  if(indices[idx] >= targetStart
     && (idx == 0 || indices[idx - 1] < targetStart)) {
    resultStart[0] = idx;
  }

  if(indices[idx] < targetEnd
     && (idx == size - 1 || indices[idx + 1] >= targetEnd))
    resultEnd[0] = idx;
}

SparseTensorBase::SparseTensorBase(int capacity, Ptr<Backend> backend)
: backend_(backend), capacity_(capacity) {
  cudaSetDevice(backend_->getDevice().no);
  CUDA_CHECK(cudaMalloc(&data_, sizeof(float) * capacity));
  CUDA_CHECK(cudaMalloc(&indices_, sizeof(int) * capacity));

  CUDA_CHECK(cudaMalloc(&gstart_, sizeof(int) * 100));
  CUDA_CHECK(cudaMalloc(&gend_, sizeof(int) * 100));
}

SparseTensorBase::SparseTensorBase(float* data,
                                   int* indices,
                                   int size,
                                   Ptr<Backend> backend)
: backend_(backend) {
  data_ = data;
  indices_ = indices;
  size_ = size;
  capacity_ = size;
}

int SparseTensorBase::capacity() {
  return capacity_;
}

int SparseTensorBase::size() {
  return size_;
}

float* SparseTensorBase::data() {
  return data_;
}

int* SparseTensorBase::indices() {
  return indices_;
}

void SparseTensorBase::copyFrom(float* data,
                                int* indices,
                                int size,
                                bool data_only) {
  if(capacity_ < size) {
    return;
    // NO enough capacity
  }
  size_ = size;
  if(size == 0)
    return;
  cudaSetDevice(backend_->getDevice().no);

  cudaMemcpy(data_, data, size * sizeof(float), cudaMemcpyDefault);
  if(!data_only)
    cudaMemcpy(indices_, indices, size * sizeof(int), cudaMemcpyDefault);
  cudaStreamSynchronize(0);
}

// copy from another sparse tensor
void SparseTensorBase::copyFrom(std::shared_ptr<SparseTensorBase> t,
                                bool data_only) {
  copyFrom(t->data(), t->indices(), t->size(), data_only);
}

Ptr<Backend> SparseTensorBase::getBackend() {
  return backend_;
}

void SparseTensorBase::setSize(int size) {
  size_ = size;
}

// return the dense representation of this tensor
void SparseTensorBase::toDense(Tensor t, int offset) {
  cudaSetDevice(backend_->getDevice().no);
  int threads = 512;
  int blocks = 1 + size_ / threads;
  t->set(0);
  gScatterAdd<<<blocks, threads>>>(
      t->data(), data_, indices_, t->size(), size_, offset);
  cudaStreamSynchronize(0);
}

void SparseTensorBase::scatterAdd(Tensor t, int offset) {
  cudaSetDevice(backend_->getDevice().no);
  cudaStreamSynchronize(0);
  int threads = 512;
  int blocks = 1 + size_ / threads;
  gScatterAdd<<<blocks, threads>>>(
      t->data(), data_, indices_, t->size(), size_, offset);
  cudaStreamSynchronize(0);
}

std::shared_ptr<SparseTensorBase> SparseTensorBase::subtensor(int pos,
                                                              int size,
                                                              int idx) {
  cudaSetDevice(backend_->getDevice().no);
  cudaStreamSynchronize(0);
  int* start = gstart_ + idx;
  int* end = gend_ + idx;

  int threads = 512;
  int blocks = 1 + size_ / threads;
  cudaMemset(start, -1, sizeof(int));
  cudaMemset(end, 0, sizeof(int));

  gFindSubtensor<<<blocks, threads>>>(
      indices_, size_, pos, pos + size, start, end);

  int startOffset;
  int endOffset;
  int tmp_dt;
  cudaMemcpy(&startOffset, start, sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(&endOffset, end, sizeof(int), cudaMemcpyDeviceToHost);

  if(startOffset != -1 && startOffset < size_)
    cudaMemcpy(
        &tmp_dt, indices_ + startOffset, sizeof(int), cudaMemcpyDeviceToHost);

  int subtensorSize = std::max(0, endOffset - startOffset + 1);
  cudaStreamSynchronize(0);
  return std::shared_ptr<SparseTensorBase>(new SparseTensorBase(
      data_ + startOffset, indices_ + startOffset, subtensorSize, backend_));
}
}
