#include "tensors/tensor_gpu.h"

namespace marian {

__global__ void gFill(float* d_in, int size, float val) {
  for(int bid = 0; bid < size; bid += blockDim.x * gridDim.x) {
    int index = bid + threadIdx.x + blockDim.x * blockIdx.x;
    if (index < size) {
      d_in[index] = val;
    }
  }
}

}
