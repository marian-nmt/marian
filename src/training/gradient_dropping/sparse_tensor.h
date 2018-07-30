#pragma once

#include <algorithm>
#include <memory>

#include "common/definitions.h"
#include "tensors/backend.h"
#include "tensors/device.h"
#include "tensors/tensor_operators.h"

#ifdef CUDA_FOUND
#include "tensors/gpu/algorithm.h"
#include "training/gradient_dropping/gpu/sparse_algorithm.h"
#endif

namespace marian {
class SparseTensorBase : public std::enable_shared_from_this<SparseTensorBase> {
  float* data_;
  int* indices_;

  int size_;
  int capacity_;
  Ptr<Backend> backend_;

  std::vector<Ptr<Device>> devices;

  template <typename T>
  T* newData(int size, Ptr<Backend> backend) {
    Ptr<Device> device = DispatchDevice(backend->getDevice());
    device->reserve(size * sizeof(T));
    devices.push_back(device);
    return (T*)device->data();
  }

public:
  SparseTensorBase(int capacity, Ptr<Backend> backend)
      : backend_(backend), capacity_(capacity) {
    data_ = newData<float>(capacity, backend);
    indices_ = newData<int>(capacity, backend);
  }

  SparseTensorBase(float* data, int* indices, int size, Ptr<Backend> backend)
      : backend_(backend) {
    data_ = data;
    indices_ = indices;
    size_ = size;
    capacity_ = size;
  }

  ~SparseTensorBase() {}

  int capacity() { return capacity_; }

  int size() { return size_; }

  void setSize(int size) { size_ = size; }

  Ptr<Backend> getBackend() { return backend_; }

  float* data() { return data_; }

  int* indices() { return indices_; }

  // copy to cpu vector
  void get(std::vector<float>& g, std::vector<int>& i) {
    int s = std::min((int)g.size(), size());
    if(backend_->getDevice().type == DeviceType::cpu) {
      std::copy(data(), data() + s, g.data());
      std::copy(indices(), indices() + s, i.data());
    }
#ifdef CUDA_FOUND
    else {
      gpu::copy(backend_, data(), data() + s, g.data());
      gpu::copy(backend_, indices(), indices() + s, i.data());
    }
#endif
  }

  // copy from cpu vector
  void set(const std::vector<float>& g, const std::vector<int>& i) {
    int s = std::min((int)g.size(), capacity());
    size_ = s;
    if(backend_->getDevice().type == DeviceType::cpu) {
      std::copy(g.data(), g.data() + s, data());
      std::copy(i.data(), i.data() + s, indices());
    }
#ifdef CUDA_FOUND
    else {
      gpu::copy(backend_, g.data(), g.data() + s, data());
      gpu::copy(backend_, i.data(), i.data() + s, indices());
    }
#endif
  }

  void copyFrom(float* ndata, int* nindices, int nsize) {
    size_ = nsize;
    if(backend_->getDevice().type == DeviceType::cpu) {
      ABORT("Gradient Dropping for CPU is not yet supported");
    }
#ifdef CUDA_FOUND
    else {
      gpu::copy(backend_, ndata, ndata + nsize, data());
      gpu::copy(backend_, nindices, nindices + nsize, indices());
    }
#endif
  }

  void copyFrom(std::shared_ptr<SparseTensorBase> t) {
    copyFrom(t->data(), t->indices(), t->size());
  }

  // Convert sparseTensor into a Tensor
  void toDense(Tensor t, int offset = 0) {
    t->set(0);
    scatterAdd(t, offset);
  }

  // Convert a tensor into a sparse tensor format
  void fromDense(Tensor t) {
    if(backend_->getDevice().type == DeviceType::cpu) {
      ABORT("Gradient Dropping for CPU is not yet supported");
    }
#ifdef CUDA_FOUND
    else {
      int sparse_size = gpu::buildSparse(t, data(), indices());
      setSize(sparse_size);
    }
#endif
  }

  // Add t[indices[i]] += data[i]
  void scatterAdd(Tensor t, int offset = 0) {
    if(backend_->getDevice().type == DeviceType::cpu) {
      ABORT("Gradient Dropping for CPU is not yet supported");
    }
#ifdef CUDA_FOUND
    else {
      gpu::scatterAdd(t, data(), indices(), size(), offset);
    }
#endif
  }

  // Add t[indices[i]] = data[i]
  void scatterUpdate(Tensor t, int offset = 0) {
    if(backend_->getDevice().type == DeviceType::cpu) {
      ABORT("Gradient Dropping for CPU is not yet supported");
    }
#ifdef CUDA_FOUND
    else {
      gpu::scatterUpdate(t, data(), indices(), size(), offset);
    }
#endif
  }

  // data[i] = t[indices[i]]
  void gather(Tensor t, int offset = 0) {
    if(backend_->getDevice().type == DeviceType::cpu) {
      ABORT("Gradient Dropping for CPU is not yet supported");
    }
#ifdef CUDA_FOUND
    else {
      gpu::gather(t, data(), indices(), size(), offset);
    }
#endif
  }

  std::shared_ptr<SparseTensorBase> subtensor(int pos, int subsize) {
    int startOffset = 0;
    int endOffset = 0;

    std::vector<int> values(2);
    values[0] = pos;
    values[1] = pos + subsize - 1;

    if(backend_->getDevice().type == DeviceType::cpu) {
      ABORT("Gradient Dropping for CPU is not yet supported");
    }
#ifdef CUDA_FOUND
    else {
      std::vector<int> outputs
          = gpu::lower_bounds(indices(), values, size(), backend_->getDevice());

      startOffset = outputs[0];
      endOffset = outputs[1];
    }
#endif

    int subtensorSize = std::max(0, endOffset - startOffset);

    return std::shared_ptr<SparseTensorBase>(new SparseTensorBase(
        data_ + startOffset, indices_ + startOffset, subtensorSize, backend_));
  }
};

typedef std::shared_ptr<SparseTensorBase> SparseTensor;
}
