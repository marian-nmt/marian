#pragma once

#ifndef NO_CUDA

#include <thrust/device_vector.h>

template<class T>
using DeviceVector = thrust::device_vector<T>;

template<class T>
using HostVector = thrust::host_vector<T>;

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

