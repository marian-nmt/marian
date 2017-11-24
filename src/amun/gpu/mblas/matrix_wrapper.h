#pragma once
#include "matrix.h"
#include "gpu/mblas/array.h"

namespace amunmt {
namespace GPU {
namespace mblas {

template <typename T>
class MatrixWrapper
{
public:
  MatrixWrapper()
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

  MatrixWrapper(uint a, uint b, uint c, uint d)
  { // test constructor
    dim_[0] = a;
    dim_[1] = b;
    dim_[2] = c;
    dim_[3] = d;
    updateStrides();
  }

  MatrixWrapper(const Array<T> &vec)
  {
    dim_[0] = vec.size();
    dim_[1] = 1;
    dim_[2] = 1;
    dim_[3] = 1;
    updateStrides();

    assert(size() == vec.size());

    data_ = nullptr;
    dataConst_ = vec.data();
  }

  MatrixWrapper(Array<T> &vec)
  {
    dim_[0] = vec.size();
    dim_[1] = 1;
    dim_[2] = 1;
    dim_[3] = 1;
    updateStrides();

    assert(size() == vec.size());

    data_ = vec.data();
    dataConst_ = data_;
  }

  __device__
  MatrixWrapper(T *ptr, uint a, uint b, uint c, uint d)
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
  uint dim(uint i) const
  {  return dim_[i]; }

  __device__ __host__
  uint size() const
  {
    return size_;
  }

  __device__ __host__
  uint stride(uint i) const
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
  const T &operator[](uint i) const
  {
    assert(i < size());
    return data()[i];
  }

  __device__
  T &operator[](uint i)
  {
    assert(i < size());
    return data()[i];
  }

  __device__
  const T &operator()(uint a, uint b, uint c, uint d) const
  {
    uint id = indices2Id(a, b, c, d);
    return data()[id];
  }

  __device__
  T &operator()(uint a, uint b, uint c, uint d)
  {
    uint id = indices2Id(a, b, c, d);
    return data()[id];
  }

  __device__ __host__
  void id2Indices(uint id, uint *out) const
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

  __device__ __host__
  uint indices2Id(uint a, uint b, uint c, uint d) const
  {
    assert(a < dim(0));
    assert(b < dim(1));
    assert(c < dim(2));
    assert(d < dim(3));

    uint ind = 0;
    ind += a * stride(0);
    ind += b * stride(1);
    ind += c * stride(2);
    ind += d * stride(3);

    assert(ind < size());
    return ind;
  }

  std::string Debug() const
  {
    std::stringstream strm;

    strm << "dim=";
    for (uint i = 0; i < SHAPE_SIZE; ++i) {
      strm << dim_[i] << " ";
    }
    strm << "=" << size_;

    strm << " stride=";
    for (uint i = 0; i < SHAPE_SIZE; ++i) {
      strm << stride(i) << " ";
    }

    return strm.str();
  }

protected:
  uint dim_[SHAPE_SIZE];
  uint stride_[SHAPE_SIZE];
  uint size_;

  T *data_;
  const T *dataConst_;

};

///////////////////////////////////////////////////////////////////////////////

inline void testidToMatrixInd()
{
  MatrixWrapper<float> matrix(2, 4, 3, 5);

  std::cerr << "matrix=" << matrix.Debug() << std::endl;

  for (uint i = 0; i < matrix.size(); ++i) {
    uint dim[SHAPE_SIZE];
    matrix.id2Indices(i, dim);

    std::cerr << i << "=";
    for (uint j = 0; j < SHAPE_SIZE; ++j) {
      std::cerr << " " << dim[j];
    }

    std::cerr << " = " << matrix.indices2Id(dim[0], dim[1], dim[2], dim[3]);
    std::cerr << std::endl;
  }
}

}
}
}
