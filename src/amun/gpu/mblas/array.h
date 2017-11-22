/*
 * Array.h
 *
 *  Created on: 8 Dec 2016
 *      Author: hieu
 */

#pragma once
#include <cuda.h>
#include <thrust/host_vector.h>

namespace amunmt {
namespace GPU {
namespace mblas {

template<typename T>
class Array
{
public:
  Array()
  :m_size(0)
  ,m_maxSize(0)
  ,m_arr(nullptr)
  {
  }

  Array(size_t size)
  :m_maxSize(0)
  {
    resize(size);
  }

  Array(size_t size, const T &val)
  :m_maxSize(0)
  {
    resize(size);

    if (val) {
      abort();
    }
    else {
      HANDLE_ERROR(cudaMemsetAsync(m_arr, 0, m_size * sizeof(float), CudaStreamHandler::GetStream()));
    }
  }

  Array(const thrust::host_vector<T> &vec)
  :m_maxSize(0)
  {
    resize(vec.size());
    HANDLE_ERROR( cudaMemcpyAsync(m_arr, thrust::raw_pointer_cast(vec.data()), vec.size() * sizeof(T), cudaMemcpyHostToDevice, CudaStreamHandler::GetStream()) );
  }

  Array(const std::vector<T> &vec)
  :m_maxSize(0)
  {
    resize(vec.size());
    HANDLE_ERROR( cudaMemcpyAsync(m_arr, vec.data(), vec.size() * sizeof(T), cudaMemcpyHostToDevice, CudaStreamHandler::GetStream()) );
  }

  ~Array()
  {
	cudaFree(m_arr);
  }

  __host__ __device__
  size_t size() const
  { return m_size; }

  __host__
  T *data()
  { return m_arr; }

  __host__
  const T *data() const
  { return m_arr; }

  __device__
  const T& operator[](size_t ind) const
  { return m_arr[ind]; }

  __device__
  T& operator[](size_t ind)
  { return m_arr[ind]; }

  __host__
  void resize(size_t newSize)
  {
    reserve(newSize);
    m_size = newSize;
  }

  __host__
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

  __device__
  void clear()
  {
    m_size = 0;
  }

protected:
  size_t m_size, m_maxSize;
  T *m_arr;



};

}
}
}
