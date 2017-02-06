#pragma once
#include <cudnn.h>

#include "tensors/tensor.h"

namespace marian {

void CudnnDropoutPrepare(Tensor in, float p,
                        cudnnDropoutDescriptor_t* dropDesc,
                        void** space, size_t* spaceSize,
                        void** states, size_t seed);

void CudnnDropoutDestroy(cudnnDropoutDescriptor_t dropDesc,
                        void* space, void* states);

void CudnnDropoutForward(cudnnDropoutDescriptor_t dropoutDesc,
                 void* space, size_t spaceSize,
                 Tensor out, Tensor in);

void CudnnDropoutBackward(cudnnDropoutDescriptor_t dropoutDesc,
                         void* space, size_t spaceSize,
                         Tensor out, Tensor in);

}
