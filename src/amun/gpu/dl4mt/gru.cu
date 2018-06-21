#include <sstream>
#include "gru.h"

using namespace std;

namespace amunmt {
namespace GPU {

__global__ void gElementwiseOps(mblas::TensorWrapper<float> outWrap,
                                const mblas::TensorWrapper<float> stateWrap,
                                const mblas::TensorWrapper<float> ruhWrap,
                                const mblas::TensorWrapper<float> tempWrap,
                                const mblas::TensorWrapper<float> bWrap,
                                const mblas::TensorWrapper<float> bx1Wrap,
                                const mblas::TensorWrapper<float> bx2Wrap)
{
  const unsigned rows = stateWrap.dim(0);
  const unsigned cols = stateWrap.dim(1);
  assert(blockIdx.x < rows);
  assert(ruhWrap.dim(1) == cols * 3);

  for(int tid = 0; tid < cols; tid += blockDim.x) {
    int i = tid + threadIdx.x;
    if(i < cols) {
      float ev1 = expf(-(ruhWrap(blockIdx.x, i)
                         + bWrap[i]
                         + tempWrap(blockIdx.x, i)
                        )
                      );
      float r = 1.0f / (1.0f + ev1);

      int k = i + cols;
      float ev2 = expf(-(ruhWrap(blockIdx.x, k)
                         + bWrap[k]
                         + tempWrap(blockIdx.x, k)
                        )
                      );
      float u = 1.0f / (1.0f + ev2);

      float hv = ruhWrap(blockIdx.x, 2*cols + i)
               + bx1Wrap[i];

      float t2v = tempWrap(blockIdx.x, 2*cols + i)
                + bx2Wrap[i];

      hv = tanhf(hv + r * t2v);
      outWrap(blockIdx.x, i) = (1.0f - u) * hv + u * stateWrap(blockIdx.x, i);
    }
  }
}


}
}

