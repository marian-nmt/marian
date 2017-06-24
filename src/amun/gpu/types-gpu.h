#pragma once

#ifndef NO_CUDA

#include <thrust/device_vector.h>

/////////////////////////////////////////////////////////////////////////////////////

void HandleError(cudaError_t err, const char *file, int line );

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

/////////////////////////////////////////////////////////////////////////////////////

template<class T>
using DeviceVector = thrust::device_vector<T>;

template<class T>
using HostVector = thrust::host_vector<T>;

/////////////////////////////////////////////////////////////////////////////////////

template <typename T>
class DeviceVectorWrapper
{
public:
  DeviceVectorWrapper(const DeviceVector<T> &vec)
  {
    size_ = vec.size();

    data_ = nullptr;
    dataConst_ = thrust::raw_pointer_cast(vec.data());
  }

  DeviceVectorWrapper(DeviceVector<T> &vec)
  {
    size_ = vec.size();

    data_ = thrust::raw_pointer_cast(vec.data());
    dataConst_ = data_;
  }

  __device__
  size_t size() const
  { return size_; }

  __device__
  T* data()
  { return data_; }

  __device__
  const T* data() const
  { return dataConst_; }

  __device__
  const T &operator[](size_t i) const
  {
    assert(i < size());
    return dataConst_[i];
  }

  __device__
  T &operator[](size_t i)
  {
	assert(i < size());
    return data_[i];
  }

protected:
  size_t size_;

  T *data_;
  const T *dataConst_;

};

/////////////////////////////////////////////////////////////////////////////////////

namespace algo = thrust;
namespace iteralgo = thrust;
#else

#include <vector>
#include <algorithm>

template<class T>
using DeviceVector = std::vector<T>;

template<class T>
using HostVector = std::vector<T>;

namespace algo = std;
namespace iteralgo = std;


#endif

/////////////////////////////////////////////////////////////////////////////////////

#define BEGIN_TIMER(num) { HANDLE_ERROR( cudaStreamSynchronize(mblas::CudaStreamHandler::GetStream())); timers[num].resume(); }
#define PAUSE_TIMER(num, str) { HANDLE_ERROR( cudaStreamSynchronize(mblas::CudaStreamHandler::GetStream())); \
                          timers[num].stop(); \
                          std::cerr << str << timers[num].format() << std::endl; \
                           }


