#pragma once
#include "tensor.h"
#include "vector_wrapper.h"

namespace amunmt {
namespace GPU {
namespace mblas {

template <typename T>
class TensorWrapper
{
public:
  TensorWrapper()
  {
    dim_[0] = 0;
    dim_[1] = 0;
    dim_[2] = 0;
    dim_[3] = 0;
    updateStrides();

    data_ = nullptr;
    dataConst_ = nullptr;
  }

  TensorWrapper(const TTensor<T> &matrix)
  {
    dim_[0] = matrix.dim(0);
    dim_[1] = matrix.dim(1);
    dim_[2] = matrix.dim(2);
    dim_[3] = matrix.dim(3);
    updateStrides();

    data_ = nullptr;
    dataConst_ = matrix.data();
  }

  TensorWrapper(TTensor<T> &matrix)
  {
    dim_[0] = matrix.dim(0);
    dim_[1] = matrix.dim(1);
    dim_[2] = matrix.dim(2);
    dim_[3] = matrix.dim(3);
    updateStrides();

    data_ = matrix.data();
    dataConst_ = data_;
  }

  TensorWrapper(unsigned a, unsigned b, unsigned c, unsigned d)
  { // test constructor
    dim_[0] = a;
    dim_[1] = b;
    dim_[2] = c;
    dim_[3] = d;
    updateStrides();

    data_ = nullptr;
    dataConst_ = nullptr;
  }

  __device__
  TensorWrapper(T *ptr, unsigned a, unsigned b, unsigned c, unsigned d)
  {
    dim_[0] = a;
    dim_[1] = b;
    dim_[2] = c;
    dim_[3] = d;
    updateStrides();

    data_ = ptr;
    dataConst_ = ptr;
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

  __device__
  T* data()
  {
    assert(data_);
    return data_;
  }

  __device__
  const T* data() const
  {
    assert(dataConst_);
    return dataConst_;
  }

  __device__
  const T &operator[](unsigned i) const
  {
    assert(i < size());
    return data()[i];
  }

  __device__
  T &operator[](unsigned i)
  {
    assert(i < size());
    return data()[i];
  }

  // operator()
  // 4
  __device__
  inline const T &operator()(unsigned a, unsigned b, unsigned c, unsigned d) const
  {
    unsigned id = indices2Id(a, b, c, d);
    return data()[id];
  }

  __device__
  inline T &operator()(unsigned a, unsigned b, unsigned c, unsigned d)
  {
    unsigned id = indices2Id(a, b, c, d);
    return data()[id];
  }

  // 3
  __device__
  inline const T &operator()(unsigned a, unsigned b, unsigned c) const
  {
    unsigned id = indices2Id(a, b, c);
    return data()[id];
  }

  __device__
  inline T &operator()(unsigned a, unsigned b, unsigned c)
  {
    unsigned id = indices2Id(a, b, c);
    return data()[id];
  }

  // 2
  __device__
  inline const T &operator()(unsigned a, unsigned b) const
  {
    unsigned id = indices2Id(a, b);
    return data()[id];
  }

  __device__
  inline T &operator()(unsigned a, unsigned b)
  {
    unsigned id = indices2Id(a, b);
    return data()[id];
  }

  // 1
  __device__
  inline const T &operator()(unsigned a) const
  {
    unsigned id = indices2Id(a);
    return data()[id];
  }

  __device__
  inline T &operator()(unsigned a)
  {
    unsigned id = indices2Id(a);
    return data()[id];
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

  // indices2Id
  // 4
  __device__ __host__
  inline unsigned indices2Id(unsigned a, unsigned b, unsigned c, unsigned d) const
  {
    assert(a < dim(0));
    assert(b < dim(1));
    assert(c < dim(2));
    assert(d < dim(3));

    unsigned ind =
            a * stride(0)
          + b * stride(1)
          + c * stride(2)
          + d * stride(3);

    assert(ind < size());
    return ind;
  }

  // 3
  __device__ __host__
  inline unsigned indices2Id(unsigned a, unsigned b, unsigned c) const
  {
    assert(a < dim(0));
    assert(b < dim(1));
    assert(c < dim(2));

    unsigned ind =
            a * stride(0)
          + b * stride(1)
          + c * stride(2);

    assert(ind < size());
    return ind;
  }

  // 2
  __device__ __host__
  inline unsigned indices2Id(unsigned a, unsigned b) const
  {
    assert(a < dim(0));
    assert(b < dim(1));

    unsigned ind =
            a * stride(0)
          + b * stride(1);

    assert(ind < size());
    return ind;
  }

  // 1
  __device__ __host__
  inline unsigned indices2Id(unsigned a) const
  {
    assert(a < dim(0));

    unsigned ind =
          a * stride(0);

    assert(ind < size());
    return ind;
  }

  __device__
  VectorWrapper<T> Row(unsigned row)
  {
    T &ele = (*this)(row);
    VectorWrapper<T> ret(&ele, dim(1));
    return ret;
  }

  std::string Debug() const
  {
    std::stringstream strm;

    strm << "dim=";
    for (unsigned i = 0; i < SHAPE_SIZE; ++i) {
      strm << dim_[i] << " ";
    }
    strm << "=" << size_;

    strm << " stride=";
    for (unsigned i = 0; i < SHAPE_SIZE; ++i) {
      strm << stride(i) << " ";
    }

    return strm.str();
  }

protected:
  unsigned dim_[SHAPE_SIZE];
  unsigned stride_[SHAPE_SIZE];
  unsigned size_;

  T *data_;
  const T *dataConst_;

};

///////////////////////////////////////////////////////////////////////////////

inline void testidToMatrixInd()
{
  TensorWrapper<float> matrix(2, 4, 3, 5);

  std::cerr << "matrix=" << matrix.Debug() << std::endl;

  for (unsigned i = 0; i < matrix.size(); ++i) {
    unsigned dim[SHAPE_SIZE];
    matrix.id2Indices(i, dim);

    std::cerr << i << "=";
    for (unsigned j = 0; j < SHAPE_SIZE; ++j) {
      std::cerr << " " << dim[j];
    }

    std::cerr << " = " << matrix.indices2Id(dim[0], dim[1], dim[2], dim[3]);
    std::cerr << std::endl;
  }
}

///////////////////////////////////////////////////////////////////////////////

}
}
}
