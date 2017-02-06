#include "dropout_cudnn.h"

#include "tensors/tensor.h"

namespace marian {

static cudnnHandle_t create_handle_dnn() {
 cudnnHandle_t cudnnHandle;
 cudnnCreate(&cudnnHandle);
 return cudnnHandle;
}

cudnnHandle_t cudnnHandle = create_handle_dnn();

void CudnnDropoutPrepare(Tensor in, float p,
                        cudnnDropoutDescriptor_t* dropDesc,
                        void** space, size_t* spaceSize,
                        void** states, size_t seed) {
 size_t statesSize;
 cudnnDropoutGetStatesSize(cudnnHandle, &statesSize);
 cudnnDropoutGetReserveSpaceSize(in->cudnn(), spaceSize);

 cudaMalloc((void**)states, statesSize);
 cudaMalloc((void**)space, *spaceSize);

 cudnnCreateDropoutDescriptor(dropDesc);
 cudnnSetDropoutDescriptor(*dropDesc,
                           cudnnHandle,
                           p,
                           (void*)*states,
                           statesSize,
                           seed);
}

void CudnnDropoutDestroy(cudnnDropoutDescriptor_t dropDesc,
                        void* space, void* states) {
 cudnnDestroyDropoutDescriptor(dropDesc);
 cudaFree(space);
 cudaFree(states);
}

void CudnnDropoutForward(cudnnDropoutDescriptor_t dropoutDesc,
                 void* space, size_t spaceSize,
                 Tensor out, Tensor in) {
 cudnnDropoutForward(cudnnHandle,
                     dropoutDesc,
                     in->cudnn(),
                     in->data(),
                     out->cudnn(),
                     out->data(),
                     space,
                     spaceSize);
}

/* void CudnnDropoutBackward(cudnnDropoutDescriptor_t dropoutDesc, */
                         /* void* space, size_t spaceSize, */
                         /* Tensor out, Tensor in) { */
 /* auto inGpu = static_cast<TensorGPU*>(in.get()); */
 /* auto outGpu = static_cast<TensorGPU*>(out.get()); */
 /* cudnnDropoutBackward(cudnnHandle, */
                     /* dropoutDesc, */
                     /* inGpu->cudnn(), */
                     /* inGpu->data(), */
                     /* outGpu->cudnn(), */
                     /* outGpu->data(), */
                     /* space, */
                     /* spaceSize); */
/* } */

}
