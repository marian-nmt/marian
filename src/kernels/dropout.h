#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <curand.h>

#include "tensors/tensor.h"

namespace marian {

curandGenerator_t createCurandGenerator(size_t device, size_t seed=1234);

void Dropout(Tensor tensor, float h,
             curandGenerator_t gen);

}
