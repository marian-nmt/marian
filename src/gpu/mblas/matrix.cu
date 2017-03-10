#include "matrix.h"

namespace amunmt {
namespace GPU {
namespace mblas {

__global__ void gSum(const float *data, size_t count, float &ret)
{
  ret = 0.0f;
  for (size_t i = 0; i < count; ++i) {
    ret += data[i];
  }
}

float Sum(const float *data, size_t count)
{
  float ret;
  float *d_ret;
  HANDLE_ERROR( cudaMalloc((void**)&d_ret, sizeof(float)) );

  HANDLE_ERROR( cudaStreamSynchronize(CudaStreamHandler::GetStream()));

  gSum<<<1,1>>>(data, count, *d_ret);
  HANDLE_ERROR( cudaMemcpy(&ret, d_ret, sizeof(float), cudaMemcpyDeviceToHost) );
  HANDLE_ERROR(cudaFree(d_ret));

  HANDLE_ERROR( cudaStreamSynchronize(CudaStreamHandler::GetStream()));

  return ret;
}

}
}
}
