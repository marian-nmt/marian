#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <curand.h>

#include "tensors/tensor.h"

namespace marian {

void Dropout(Tensor tensor, float h,
             curandGenerator_t gen);

}
