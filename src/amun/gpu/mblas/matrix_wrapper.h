#pragma once
#include <sstream>
#include "matrix.h"
#include "vector_wrapper.h"
#include "shape.h"

namespace amunmt {
namespace GPU {
namespace mblas {

template <typename T>
class MatrixWrapper
{
public:
  MatrixWrapper()
  :shape_()
  {
    dim_[0] = 0;
    dim_[1] = 0;
    dim_[2] = 0;
    dim_[3] = 0;
    updateStrides();

    data_ = nullptr;
    dataConst_ = nullptr;
  }

  MatrixWrapper(const TMatrix<T> &matrix, bool colMajor = true)
  :shape_(matrix.dim(0), matrix.dim(1), matrix.dim(2), matrix.dim(3), colMajor)
  {
    dim_[0] = matrix.dim(0);
    dim_[1] = matrix.dim(1);
    dim_[2] = matrix.dim(2);
    dim_[3] = matrix.dim(3);

    if (colMajor) {
      updateStrides();
    }
    else {
      updateStridesRowMajor();
    }

    data_ = nullptr;
    dataConst_ = matrix.data();
  }

  MatrixWrapper(TMatrix<T> &matrix, bool colMajor = true)
  :shape_(matrix.dim(0), matrix.dim(1), matrix.dim(2), matrix.dim(3), colMajor)
  {
    dim_[0] = matrix.dim(0);
    dim_[1] = matrix.dim(1);
    dim_[2] = matrix.dim(2);
    dim_[3] = matrix.dim(3);

    if (colMajor) {
      updateStrides();
    }
    else {
      updateStridesRowMajor();
    }

    data_ = matrix.data();
    dataConst_ = data_;
  }

  MatrixWrapper(unsigned a, unsigned b, unsigned c, unsigned d)
  :shape_(a, b, c, d)
  { // test constructor
    dim_[0] = a;
    dim_[1] = b;
    dim_[2] = c;
    dim_[3] = d;
    updateStrides();
  }

  __device__
  MatrixWrapper(T *ptr, unsigned a, unsigned b, unsigned c, unsigned d)
  :shape_(a, b, c, d)
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
  const Shape &GetShape() const
  { return shape_; }

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

  }

  __device__ __host__
  void updateStridesRowMajor()
  {
    stride_[0] = 1;
    stride_[1] = dim_[0];
    stride_[2] = dim_[0] * dim_[1];
    stride_[3] = dim_[0] * dim_[1] * dim_[2];

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
    return data()[i];
  }

  __device__
  T &operator[](unsigned i)
  {
    return data()[i];
  }

  // operator()
  // 4
  __device__
  const T &operator()(unsigned a, unsigned b, unsigned c, unsigned d) const
  {
    unsigned id = shape_.indices2Id(a, b, c, d);
    return data()[id];
  }

  __device__
  T &operator()(unsigned a, unsigned b, unsigned c, unsigned d)
  {
    unsigned id = shape_.indices2Id(a, b, c, d);
    return data()[id];
  }

  // 3
  __device__
  const T &operator()(unsigned a, unsigned b, unsigned c) const
  {
    unsigned id = shape_.indices2Id(a, b, c);
    return data()[id];
  }

  __device__
  T &operator()(unsigned a, unsigned b, unsigned c)
  {
    unsigned id = shape_.indices2Id(a, b, c);
    return data()[id];
  }

  // 2
  __device__
  const T &operator()(unsigned a, unsigned b) const
  {
    unsigned id = shape_.indices2Id(a, b);
    return data()[id];
  }

  __device__
  T &operator()(unsigned a, unsigned b)
  {
    unsigned id = shape_.indices2Id(a, b);
    return data()[id];
  }

  // 1
  __device__
  const T &operator()(unsigned a) const
  {
    unsigned id = shape_.indices2Id(a);
    return data()[id];
  }

  __device__
  T &operator()(unsigned a)
  {
    unsigned id = shape_.indices2Id(a);
    return data()[id];
  }

  __device__
  VectorWrapper<T> Row(unsigned row)
  {
    T &ele = (*this)(row);
    VectorWrapper<T> ret(&ele, shape_.dim(1));
    return ret;
  }

  std::string Debug() const
  {
    std::stringstream strm;
    strm << shape_.Debug();

    return strm.str();
  }

protected:
  unsigned dim_[SHAPE_SIZE];
  unsigned stride_[SHAPE_SIZE];
  Shape shape_;

  T *data_;
  const T *dataConst_;

};

///////////////////////////////////////////////////////////////////////////////

inline void testidToMatrixInd()
{
  MatrixWrapper<float> matrix(2, 4, 3, 5);

  std::cerr << "matrix=" << matrix.Debug() << std::endl;

  for (unsigned i = 0; i < matrix.GetShape().size(); ++i) {
    unsigned dim[SHAPE_SIZE];
    matrix.GetShape().id2Indices(i, dim);

    std::cerr << i << "=";
    for (unsigned j = 0; j < SHAPE_SIZE; ++j) {
      std::cerr << " " << dim[j];
    }

    std::cerr << " = " << matrix.GetShape().indices2Id(dim[0], dim[1], dim[2], dim[3]);
    std::cerr << std::endl;
  }
}

///////////////////////////////////////////////////////////////////////////////

}
}
}
