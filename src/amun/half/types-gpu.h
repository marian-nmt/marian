#pragma once

#ifndef NO_CUDA

#include <unordered_map>
#include <boost/timer/timer.hpp>
#include <cuda.h>
#include <driver_types.h>

namespace amunmt {
namespace GPU {

/////////////////////////////////////////////////////////////////////////////////////

void HandleError(cudaError_t err, const char *file, int line );

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

/////////////////////////////////////////////////////////////////////////////////////
extern std::unordered_map<std::string, boost::timer::cpu_timer> timers;

#define BEGIN_TIMER(str) {}
#define PAUSE_TIMER(str) {}
//#define BEGIN_TIMER(str) { HANDLE_ERROR( cudaStreamSynchronize(mblas::CudaStreamHandler::GetStream())); timers[str].resume(); }
//#define PAUSE_TIMER(str) { HANDLE_ERROR( cudaStreamSynchronize(mblas::CudaStreamHandler::GetStream())); \
//							timers[str].stop(); }


}
}

#else // NO CUDA

/*
#include <vector>
#include <algorithm>

namespace algo = std;
namespace iteralgo = std;
*/

#endif // NO CUDA


