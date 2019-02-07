#pragma once

#include <cstdint>
#include <string>

#include "common/shape.h"

#include "functional/array.h"

namespace marian {

namespace functional {

#define CONST_SHAPE_DIMS 4

/**
 * @brief Represents the size of each dimension in a tensor.
 */

template <const int N>
struct ConstantShape {
  Array<int, N> shape_;
  Array<int, N> stride_;
  Array<int, N> bstride_;
  size_t elements_{1};

  __HD__ ConstantShape() {
    shape_.fill(1);
    stride_.fill(1);
    bstride_.fill(0);
  }

  __HD__ ConstantShape(const ConstantShape& shape)
      : shape_(shape.shape_),
        stride_(shape.stride_),
        bstride_(shape.bstride_),
        elements_(shape.elements_) {}

  ConstantShape(const marian::Shape& shape) {
    size_t filled = shape.size();

    ABORT_IF(filled > N,
             "Recompile with CONST_SHAPE_DIMS >= " + std::to_string(filled));

    std::copy(shape.begin(), shape.end(), shape_.begin() + N - filled);
    if(N - filled)
      std::fill_n(shape_.begin(), N - filled, 1);
    updateStrides();
    updateElements();
  }

  __HDI__ void updateStrides() {
    stride_[N - 1] = 1;
    bstride_[N - 1] = shape_[N - 1] == 1 ? 0 : stride_[N - 1];

    for(int i = N - 2; i >= 0; --i) {
      stride_[i] = stride_[i + 1] * shape_[i + 1];
      bstride_[i] = shape_[i] == 1 ? 0 : stride_[i];
    }
  }

  __HDI__ void updateElements() {
    elements_ = 1;
    for(int i = 0; i < N; ++i)
      elements_ *= shape_[i];
  }

  __HDI__ void set(int i, int dim) {
    shape_[i] = dim;
    updateStrides();
    updateElements();
  }

  __HDI__ int dim(int i) { return shape_[i]; }

  __HDI__ int dim(int i) const {
    return const_cast<ConstantShape&>(*this).dim(i);
  }

  __HDI__ int back() const { return dim(N - 1); }

  __HDI__ int operator[](int i) { return dim(i); }

  __HDI__ int operator[](int i) const { return dim(i); }

  __HDI__ int stride(int i) const { return stride_[i]; }

  __HDI__ int bstride(int i) const { return bstride_[i]; }

  __HDI__ static constexpr size_t size() { return N; }

  __HDI__ int elements() const { return (int)elements_; }

  __HDI__ int index(const Array<int, N>& d) const {
    int i = 0;
    for(int j = 0; j < N; ++j)
      i += d[j] * stride_[j];
    return i;
  }

  __HDI__ int bindex(const Array<int, N>& d) const {
    int i = 0;
    for(int j = 0; j < N; ++j)
      i += d[j] * bstride_[j];
    return i;
  }

  __HDI__ void dims(int i, Array<int, N>& d) const {
    for(int j = 0; j < N; ++j)
      d[j] = (i / stride_[j]) % shape_[j];
  }

  __HDI__ bool operator==(const ConstantShape& other) const {
    for(int i = 0; i < N; ++i)
      if(shape_[i] != other[i])
        return false;
    return true;
  }

  __HDI__ bool operator!=(const ConstantShape& other) const {
    return !(*this == other);
  }
};

typedef ConstantShape<CONST_SHAPE_DIMS> Shape;
}  // namespace functional
}  // namespace marian
