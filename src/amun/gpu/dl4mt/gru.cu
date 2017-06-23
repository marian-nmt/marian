#include "gru.h"

namespace amunmt {
namespace GPU {

__global__ void gElementwiseOps(mblas::MatrixWrapper<float> outWrap,
                                const mblas::MatrixWrapper<float> stateWrap,
                                const mblas::MatrixWrapper<float> ruhWrap,
                                const mblas::MatrixWrapper<float> tempWrap,
                                const mblas::MatrixWrapper<float> bWrap,
                                const mblas::MatrixWrapper<float> bx1Wrap,
                                const mblas::MatrixWrapper<float> bx2Wrap)
{
  const uint rows = stateWrap.dim(0) * stateWrap.dim(2) * stateWrap.dim(3);
  const uint cols = stateWrap.dim(1);

  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      float* rowOut = &outWrap[0] + j * cols;
      const float* rowRuh = &ruhWrap[0] + j * cols * 3;
      const float* rowT = &tempWrap[0] + j * cols * 3;

      const float* rowH = rowRuh + 2 * cols;
      const float* rowT2 = rowT + 2 * cols;
      const float* rowState = &stateWrap[0] + j * cols;

      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int i = tid + threadIdx.x;
        if(i < cols) {
          float ev1 = expf(-(rowRuh[i] + bWrap[i] + rowT[i]));
          float r = 1.0f / (1.0f + ev1);

          int k = i + cols;
          float ev2 = expf(-(rowRuh[k] + bWrap[k] + rowT[k]));
          float u = 1.0f / (1.0f + ev2);

          float hv = rowH[i] + bx1Wrap[i];
          float t2v = rowT2[i] + bx2Wrap[i];
          hv = tanhf(hv + r * t2v);
          rowOut[i] = (1.0f - u) * hv + u * rowState[i];
        }
      }
    }
  }
}

}
}

