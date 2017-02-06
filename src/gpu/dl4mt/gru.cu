#include "gru.h"

namespace amunmt {
namespace GPU {

__global__ void gElementwiseOps(float* out,
                                const float* state,
                                const float* ruh,
                                const float* t,
                                const float* b,
                                const float* bx1,
                                const float* bx2,
                                size_t rows, size_t cols) {

  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      float* rowOut = out + j * cols;
      const float* rowRuh = ruh + j * cols * 3;
      const float* rowT = t + j * cols * 3;

      const float* rowH = rowRuh + 2 * cols;
      const float* rowT2 = rowT + 2 * cols;
      const float* rowState = state + j * cols;

      for(int tid = 0; tid < cols; tid += blockDim.x) {
        int i = tid + threadIdx.x;
        if(i < cols) {
          float ev1 = expf(-(rowRuh[i] + b[i] + rowT[i]));
          float r = 1.0f / (1.0f + ev1);

          int k = i + cols;
          float ev2 = expf(-(rowRuh[k] + b[k] + rowT[k]));
          float u = 1.0f / (1.0f + ev2);

          float hv = rowH[i] + bx1[i];
          float t2v = rowT2[i] + bx2[i];
          hv = tanhf(hv + r * t2v);
          rowOut[i] = (1.0f - u) * hv + u * rowState[i];
        }
      }
    }
  }
}

}
}

