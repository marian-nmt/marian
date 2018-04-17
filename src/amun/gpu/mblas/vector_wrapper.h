#pragma once
#include <sstream>
#include "tensor.h"
#include "gpu/mblas/vector.h"

namespace amunmt {
namespace GPU {
namespace mblas {


template <typename T>
class VectorWrapper
{
public:
  VectorWrapper(const Vector<T> &vec)
  {
    size_ = vec.size();
    data_ = nullptr;
    dataConst_ = vec.data();
  }

  VectorWrapper(Vector<T> &vec)
  {
    size_ = vec.size();
    data_ = vec.data();
    dataConst_ = vec.data();
  }

  __device__
  VectorWrapper(T *ptr, unsigned size)
  {
    size_ = size;
    data_ = ptr;
    dataConst_ = ptr;
  }

  __device__ __host__
  unsigned size() const
  {
    return size_;
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

  __device__
  VectorWrapper<T> Offset(unsigned offset)
  {
    T &ele = (*this)[offset];
    VectorWrapper<T> ret(&ele, size_ - offset);
    return ret;
  }

  std::string Debug() const
  {
    std::stringstream strm;
    strm << "size_=" << size_;

    return strm.str();
  }

protected:
  unsigned size_;

  T *data_;
  const T *dataConst_;

};


}
}
}

