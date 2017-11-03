#pragma once

#ifndef NO_CUDA

#include <unordered_map>
#include <thrust/device_vector.h>
#include <boost/timer/timer.hpp>

namespace amunmt {
namespace GPU {

/////////////////////////////////////////////////////////////////////////////////////

void HandleError(cudaError_t err, const char *file, int line );

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

/////////////////////////////////////////////////////////////////////////////////////

template<class T>
using DeviceVector = thrust::device_vector<T>;

template<class T>
using HostVector = thrust::host_vector<T>;
//using HostVector = std::vector<T>;

/////////////////////////////////////////////////////////////////////////////////////

namespace algo = thrust;
namespace iteralgo = thrust;

/////////////////////////////////////////////////////////////////////////////////////
extern std::unordered_map<std::string, boost::timer::cpu_timer> timers;

//#define BEGIN_TIMER(str) {}
//#define PAUSE_TIMER(str) {}
#define BEGIN_TIMER(str) { HANDLE_ERROR( cudaStreamSynchronize(mblas::CudaStreamHandler::GetStream())); timers[str].resume(); }
#define PAUSE_TIMER(str) { HANDLE_ERROR( cudaStreamSynchronize(mblas::CudaStreamHandler::GetStream())); \
							timers[str].stop(); }

}
}

#else // NO CUDA

/*
#include <vector>
#include <algorithm>

template<class T>
using DeviceVector = std::vector<T>;

template<class T>
using HostVector = std::vector<T>;

namespace algo = std;
namespace iteralgo = std;
*/

#endif // NO CUDA


