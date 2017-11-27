#pragma once
/*
 * Vector.h
 *
 *  Created on: 8 Dec 2016
 *      Author: hieu
 */

#include "gpu/types-gpu.h"

namespace amunmt {
namespace GPU {
namespace mblas {

template<typename T>
class Vector
{
public:
  Vector()
  :m_size(0)
  ,m_maxSize(0)
  ,m_arr(nullptr)
  {
  }

  Vector(size_t size)
  :m_maxSize(0)
  {
    newSize(size);
  }

  Vector(size_t size, const T &val)
  :m_maxSize(0)
  {
    newSize(size);

    if (val) {
      abort();
    }
    else {
      HANDLE_ERROR(cudaMemsetAsync(m_arr, 0, m_size * sizeof(float), CudaStreamHandler::GetStream()));
    }
  }

  Vector(const std::vector<T> &vec)
  :m_maxSize(0)
  {
    newSize(vec.size());
    HANDLE_ERROR( cudaMemcpyAsync(m_arr, vec.data(), vec.size() * sizeof(T), cudaMemcpyHostToDevice, CudaStreamHandler::GetStream()) );
  }

  Vector(const Vector<T> &other)
  :m_maxSize(other.m_size)
  {
    HANDLE_ERROR( cudaMalloc(&m_arr, m_size * sizeof(T)) );
    //std::cerr << "malloc data2:" << data_ << std::endl;
    HANDLE_ERROR( cudaMemcpyAsync(
        m_arr,
        other.m_arr,
        m_size * sizeof(T),
        cudaMemcpyDeviceToDevice,
        CudaStreamHandler::GetStream()) );
  }

  ~Vector()
  {
    HANDLE_ERROR(cudaFree(m_arr));
  }

  size_t size() const
  { return m_size; }

  T *data()
  { return m_arr; }

  const T *data() const
  { return m_arr; }

  void resize(size_t newSize)
  {
    if (newSize > m_maxSize) {
      T *newData;
      HANDLE_ERROR( cudaMalloc(&newData, newSize * sizeof(T)) );

      if (m_maxSize) {
        assert(m_arr);

        HANDLE_ERROR( cudaMemcpyAsync(
            newData,
            m_arr,
            m_size * sizeof(T),
            cudaMemcpyDeviceToDevice,
            CudaStreamHandler::GetStream()) );

        HANDLE_ERROR(cudaFree(m_arr));
      }

      m_arr = newData;
      m_maxSize = newSize;
    }

    m_size = newSize;
  }

  void newSize(size_t newSize)
  {
    reserve(newSize);
    m_size = newSize;
  }

  void reserve(size_t newSize)
  {
    if (newSize > m_maxSize) {
      if (m_maxSize) {
        HANDLE_ERROR(cudaFree(m_arr));
      }

      HANDLE_ERROR( cudaMalloc(&m_arr, newSize * sizeof(T)) );

      m_maxSize = newSize;
    }
  }

  void clear()
  {
    m_size = 0;
  }

  void swap(Vector &other)
  {
    std::swap(m_size, other.m_size);
    std::swap(m_maxSize, other.m_maxSize);
    std::swap(m_arr, other.m_arr);
  }

protected:
  size_t m_size, m_maxSize;
  T *m_arr;



};

}
}
}
