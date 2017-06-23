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
  assert(blockIdx.x < rows);
  assert(ruhWrap.dim(1) == cols * 3);

  const float* rowRuh = &ruhWrap[0] + blockIdx.x * cols * 3;
  const float* rowT = &tempWrap[0] + blockIdx.x * cols * 3;

  const float* rowH = rowRuh + 2 * cols;
  const float* rowT2 = rowT + 2 * cols;

  for(int tid = 0; tid < cols; tid += blockDim.x) {
    int i = tid + threadIdx.x;
    if(i < cols) {
      float ev1 = expf(-(rowRuh[i]
                         + bWrap[i]
                         + tempWrap[blockIdx.x * tempWrap.dim(1) + i]
                        )
                      );
      float r = 1.0f / (1.0f + ev1);

      int k = i + cols;
      float ev2 = expf(-(rowRuh[k]
                         + bWrap[k]
                         + tempWrap[blockIdx.x * tempWrap.dim(1) + k]
                        )
                      );
      float u = 1.0f / (1.0f + ev2);

      float hv = rowH[i] + bx1Wrap[i];
      float t2v = rowT2[i] + bx2Wrap[i];
      hv = tanhf(hv + r * t2v);
      outWrap[blockIdx.x * cols + i] = (1.0f - u) * hv + u * stateWrap[blockIdx.x * cols + i];
    }
  }
}

}
}

