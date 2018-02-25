#pragma once

#include <memory>

#include "common/definitions.h"
#include "tensors/backend.h"

namespace marian {
class SparseTensorBase : public std::enable_shared_from_this<SparseTensorBase> {
  float* data_;
  int* indices_;
  int size_;
  int capacity_;
  Ptr<Backend> backend_;

  int* d_is_unsorted;
  int* gstart_;
  int* gend_;

public:
  SparseTensorBase(int capacity, Ptr<Backend> backend);
  SparseTensorBase(float* data, int* indices, int size, Ptr<Backend> backend);

  ~SparseTensorBase() {}

  int capacity();

  int size();
  void setSize(int size);

  float* data();

  int* indices();

  void copyFrom(float* data, int* indices, int size, bool data_only);
  /**
   * @brief Copy from another sparse tensor
   *
   * @param t Sparse tensor
   * @param data_only False by default
   */
  void copyFrom(std::shared_ptr<SparseTensorBase> t, bool data_only = false);

  void scatterAdd(Tensor t, int offset = 0);
  std::shared_ptr<SparseTensorBase> subtensor(int pos, int size, int idx);

  Ptr<Backend> getBackend();

  void toDense(Tensor t, int offset);
};

typedef std::shared_ptr<SparseTensorBase> SparseTensor;
}
