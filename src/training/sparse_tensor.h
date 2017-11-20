#pragma once

#include <memory>

namespace marian {
class SparseTensorBase : public std::enable_shared_from_this<SparseTensorBase> {
  float* data_;
  int* indices_;
  int size_;
  int capacity_;
  size_t device_;

  int* d_is_unsorted;
  int* gstart_;
  int* gend_;

public:
  SparseTensorBase(int capacity, size_t device);

  SparseTensorBase(float* data, int* indices, int size, size_t device);

  ~SparseTensorBase() {}

  int capacity();

  int size();

  float* data();

  int* indices();

  void copyFrom(float* data, int* indices, int size, bool data_only); 

  // copy from another sparse tensor
  void copyFrom(std::shared_ptr<SparseTensorBase> t, bool data_only = false); 

  void scatterAdd(Tensor t, int offset = 0); 

  std::shared_ptr<SparseTensorBase> subtensor(int pos, int size, int idx);

  size_t getDevice();

  void setSize(int size);

  void toDense(Tensor t, int offset);

};

typedef std::shared_ptr<SparseTensorBase> SparseTensor;
}
