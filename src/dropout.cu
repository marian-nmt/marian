#include <curand.h>
#include <curand_kernel.h>

#include "dropout.h"

namespace marian {

__global__ void gInitCurandStates(curandState* states, unsigned int seed) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  curand_init(seed, tid, 0, &states[tid]);
}

unsigned Bernoulli::seed = time(0);

}