#pragma once
/*
 * Vector.h
 *
 *  Created on: 8 Dec 2016
 *      Author: hieu
 */

#include <cassert>
#include <sstream>
#include <vector>
#include "gpu/types-gpu.h"
#include "gpu/mblas/handles.h"

namespace amunmt {
namespace GPU {
namespace mblas {

///////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
__global__ void gSum(const T *data, unsigned count, T &ret)
{
  ret = 0;
  for (unsigned i = 0; i < count; ++i) {
    ret += data[i];
  }
}

template<typename T>
T Sum(const T *data, unsigned count)
{
  T ret;
  T *d_ret;
  HANDLE_ERROR( cudaMalloc(&d_ret, sizeof(T)) );

  const cudaStream_t stream = CudaStreamHandler::GetStream();

  HANDLE_ERROR( cudaStreamSynchronize(stream));
  gSum<<<1, 1, 0, stream>>>(data, count, *d_ret);
  HANDLE_ERROR( cudaMemcpyAsync(&ret, d_ret, sizeof(T), cudaMemcpyDeviceToHost, stream) );

  HANDLE_ERROR( cudaStreamSynchronize(stream));
  HANDLE_ERROR(cudaFree(d_ret));

  return ret;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
class Vector
{
public:
  Vector()
  :size_(0)
  ,maxSize_(0)
  ,data_(nullptr)
  {
  }

  Vector(unsigned size)
  :maxSize_(0)
  ,data_(nullptr)
  {
    newSize(size);
  }

  Vector(unsigned size, const T &val)
  :maxSize_(0)
  ,data_(nullptr)
  {
    newSize(size);

    if (val) {
      abort();
    }
    else {
      HANDLE_ERROR(cudaMemsetAsync(data_, 0, size_ * sizeof(float), CudaStreamHandler::GetStream()));
    }
  }

  Vector(const std::vector<T> &vec)
  :maxSize_(0)
  ,data_(nullptr)
  {
    copyFrom(vec);
  }

  explicit Vector(const Vector<T> &other)
  :maxSize_(other.size_)
  ,size_(other.size_)
  ,data_(nullptr)
  {
    if (size()) {
      HANDLE_ERROR( cudaMalloc(&data_, size_ * sizeof(T)) );
      //std::cerr << "malloc data2:" << data_ << std::endl;
      HANDLE_ERROR( cudaMemcpyAsync(
          data_,
          other.data_,
          size_ * sizeof(T),
          cudaMemcpyDeviceToDevice,
          CudaStreamHandler::GetStream()) );
    }
  }

  ~Vector()
  {
    //std::cerr << "~Vector=" << maxSize_ << " " << this << std::endl;
    HANDLE_ERROR(cudaFree(data_));
  }

  unsigned size() const
  { return size_; }

  unsigned maxSize() const
  { return maxSize_; }

  T *data()
  { return data_; }

  const T *data() const
  { return data_; }

  void resize(unsigned newSize)
  {
    if (newSize > maxSize_) {
      T *newData;
      HANDLE_ERROR( cudaMalloc(&newData, newSize * sizeof(T)) );

      if (maxSize_) {
        assert(data_);

        HANDLE_ERROR( cudaMemcpyAsync(
            newData,
            data_,
            size_ * sizeof(T),
            cudaMemcpyDeviceToDevice,
            CudaStreamHandler::GetStream()) );

        HANDLE_ERROR(cudaFree(data_));
      }
      else {
        assert(data_ == nullptr);
      }

      data_ = newData;
      maxSize_ = newSize;
    }

    size_ = newSize;
  }

  void newSize(unsigned newSize)
  {
    reserve(newSize);
    size_ = newSize;
  }

  void reserve(unsigned newSize)
  {
    //std::cerr << "reserve1=" << newSize << std::endl;
    if (newSize > maxSize_) {
      //std::cerr << "reserve2=" << newSize << std::endl;
      if (maxSize_) {
        HANDLE_ERROR(cudaFree(data_));
      }

      HANDLE_ERROR( cudaMalloc(&data_, newSize * sizeof(T)) );

      maxSize_ = newSize;
    }
    //std::cerr << "reserve3=" << newSize << std::endl;
  }

  void copyFrom(const std::vector<T> &vec)
  {
    newSize(vec.size());

    if (size()) {
      HANDLE_ERROR( cudaMemcpyAsync(data_, vec.data(), vec.size() * sizeof(T), cudaMemcpyHostToDevice, CudaStreamHandler::GetStream()) );
    }
  }

  void clear()
  {
    size_ = 0;
  }

  void swap(Vector &other)
  {
    std::swap(size_, other.size_);
    std::swap(maxSize_, other.maxSize_);
    std::swap(data_, other.data_);
  }

  virtual std::string Debug(unsigned verbosity = 1) const;

protected:
  unsigned size_, maxSize_;
  T *data_;


};

////////////////////////////////////////////////////////////////////////////////
template<typename T>
inline std::string Vector<T>::Debug(unsigned verbosity) const
{
  std::stringstream strm;
  strm << "size=" << size_; // maxSize_ << " " <<

  if (verbosity) {
    T sum = Sum(data(), size());
    strm << " sum=" << sum << std::flush;

    if (verbosity == 2) {
      const cudaStream_t& stream = CudaStreamHandler::GetStream();
      T h_data[size()];

      HANDLE_ERROR( cudaMemcpyAsync(
          &h_data,
          data_,
          size() * sizeof(T),
          cudaMemcpyDeviceToHost,
          stream) );
      HANDLE_ERROR( cudaStreamSynchronize(stream) );

      for (unsigned i = 0; i < size(); ++i) {
        strm << " " << h_data[i];
      }
    }
  }

  return strm.str();
}

template<>
inline std::string Vector<char>::Debug(unsigned verbosity) const
{
  std::stringstream strm;
  strm << "size=" << size_; // maxSize_ << " " <<

  if (verbosity) {
    unsigned sum = Sum(data(), size());
    strm << "sum=" << sum << std::flush;

    if (verbosity == 2) {
      const cudaStream_t& stream = CudaStreamHandler::GetStream();
      char h_data[size()];

      HANDLE_ERROR( cudaMemcpyAsync(
          &h_data,
          data_,
          size() * sizeof(char),
          cudaMemcpyDeviceToHost,
          stream) );
      HANDLE_ERROR( cudaStreamSynchronize(stream) );

      for (unsigned i = 0; i < size(); ++i) {
        strm << " " << (bool) h_data[i];
      }
    }
  }

  return strm.str();
}

}
}
}
