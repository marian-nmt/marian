#pragma once
#include <cassert>
#include "common/base_matrix.h"

namespace amunmt {
namespace GPU {
namespace mblas {

class Shape
{
public:
  Shape()
  {
    dim_[0] = 0;
    dim_[1] = 0;
    dim_[2] = 0;
    dim_[3] = 0;
    updateStrides();
  }

  Shape(const Shape &other, bool colMajor = true)
  {
    dim_[0] = other.dim(0);
    dim_[1] = other.dim(1);
    dim_[2] = other.dim(2);
    dim_[3] = other.dim(3);

    if (colMajor) {
      updateStrides();
    }
    else {
      updateStridesRowMajor();
    }
  }

  Shape(unsigned a, unsigned b, unsigned c, unsigned d)
  {
    dim_[0] = a;
    dim_[1] = b;
    dim_[2] = c;
    dim_[3] = d;
    updateStrides();
  }

  __device__ __host__
  unsigned dim(unsigned i) const
  {  return dim_[i]; }

  __device__ __host__
  unsigned size() const
  {
    return size_;
  }

  __device__ __host__
  unsigned stride(unsigned i) const
  {
    return stride_[i];
  }

  __device__ __host__
  void updateStrides()
  {
    stride_[0] = dim_[1];
    stride_[1] = 1;
    stride_[2] = dim_[0] * dim_[1];
    stride_[3] = dim_[0] * dim_[1] * dim_[2];

    size_ = stride_[3] * dim_[3];
  }

  __device__ __host__
  void updateStridesRowMajor()
  {
    stride_[0] = 1;
    stride_[1] = dim_[0];
    stride_[2] = dim_[0] * dim_[1];
    stride_[3] = dim_[0] * dim_[1] * dim_[2];

    size_ = stride_[3] * dim_[3];
  }

  // indices2Id
  // 4
  __device__ __host__
  unsigned indices2Id(unsigned a, unsigned b, unsigned c, unsigned d) const
  {
    assert(a < dim(0));
    assert(b < dim(1));
    assert(c < dim(2));
    assert(d < dim(3));

    unsigned ind = 0;
    ind += a * stride(0);
    ind += b * stride(1);
    ind += c * stride(2);
    ind += d * stride(3);

    assert(ind < size());
    return ind;
  }

  // 3
  __device__ __host__
  unsigned indices2Id(unsigned a, unsigned b, unsigned c) const
  {
    assert(a < dim(0));
    assert(b < dim(1));
    assert(c < dim(2));

    unsigned ind = 0;
    ind += a * stride(0);
    ind += b * stride(1);
    ind += c * stride(2);

    assert(ind < size());
    return ind;
  }

  // 2
  __device__ __host__
  unsigned indices2Id(unsigned a, unsigned b) const
  {
    assert(a < dim(0));
    assert(b < dim(1));

    unsigned ind = 0;
    ind += a * stride(0);
    ind += b * stride(1);

    assert(ind < size());
    return ind;
  }

  // 1
  __device__ __host__
  unsigned indices2Id(unsigned a) const
  {
    assert(a < dim(0));

    unsigned ind = 0;
    ind += a * stride(0);

    assert(ind < size());
    return ind;
  }

  __device__ __host__
  void id2Indices(unsigned id, unsigned *out) const
  {
    assert(id < size());

    out[3] = id / stride(3);
    id = id % stride(3);

    out[2] = id / stride(2);
    id = id % stride(2);

    out[0] = id / stride(0);
    id = id % stride(0);

    out[1] = id / stride(1);
  }

protected:
  unsigned dim_[SHAPE_SIZE];
  unsigned stride_[SHAPE_SIZE];
  unsigned size_;
};


} // namespace
}
}
