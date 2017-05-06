#pragma once

#include <vector>

#ifndef NO_CUDA

void HandleError(cudaError_t err, const char *file, int line );

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))


#include "gpu/mblas/device_vector.h"

template<class T>
using DeviceVector = amunmt::GPU::mblas::device_vector<T>;

template<class T>
using HostVector = std::vector<T>;


#else

#include <algorithm>

template<class T>
using DeviceVector = std::vector<T>;

template<class T>
using HostVector = std::vector<T>;

namespace algo = std;
namespace iteralgo = std;


#endif

