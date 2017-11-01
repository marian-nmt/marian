#pragma once

#include "gpu/array.h"
#include "gpu/shape.h"
#include "tensors/tensor.h"

namespace marian {
namespace gpu {

template<typename T>
struct Tensor {
  T* data_;
  gpu::Shape shape_;

  __HD__ Tensor() {}

  __HD__ Tensor(T* ptr, const gpu::Shape& shape)
  : data_(ptr), shape_(shape) {}

  __H__ Tensor(marian::Tensor t)
  : data_(t->data()), shape_(t->shape()) {}

  __HDI__ float& operator[](size_t i) { return data_[i]; }
  __HDI__ const float& operator[](size_t i) const { return data_[i]; }

  __HDI__ float& operator[](const gpu::Array<int, gpu::Shape::size()>& indices) {
    return data_[shape_.index(indices)];
  }

  __HDI__ const float& operator[](const gpu::Array<int, gpu::Shape::size()>& indices) const {
    return data_[shape_.index(indices)];
  }

  __HDI__ T* data() { return data_; }
  __HDI__ const T* data() const { return data_; }

  __HDI__ Shape& shape() { return shape_; }
  __HDI__ const Shape& shape() const { return shape_; }
};

}
}