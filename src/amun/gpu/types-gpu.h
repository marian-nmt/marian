#pragma once

#ifndef NO_CUDA

#include <unordered_map>
#include <boost/timer/timer.hpp>
#include <cuda.h>
#include <cublas_v2.h>
#include <driver_types.h>

namespace amunmt {

extern std::unordered_map<std::string, boost::timer::cpu_timer> timers;

namespace GPU {

#define LOWEST_FLOAT -1111111111111
#define HIGHEST_FLOAT +999999999999

/////////////////////////////////////////////////////////////////////////////////////
void HandleError(cudaError_t err, const char *file, int line );
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

/////////////////////////////////////////////////////////////////////////////////////
void HandleErrorCublas(cublasStatus_t err, const char *file, int line );
#define HANDLE_ERROR_CUBLAS( err ) (HandleErrorCublas( err, __FILE__, __LINE__ ))

/////////////////////////////////////////////////////////////////////////////////////

#define BEGIN_TIMER(str) {}
#define PAUSE_TIMER(str) {}
//#define BEGIN_TIMER(str) { HANDLE_ERROR( cudaStreamSynchronize(mblas::CudaStreamHandler::GetStream())); timers[str].resume(); }
//#define PAUSE_TIMER(str) { HANDLE_ERROR( cudaStreamSynchronize(mblas::CudaStreamHandler::GetStream())); timers[str].stop(); }


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


