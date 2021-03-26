#include <cuda.h>
#include <limits>

#include "data/types.h"
#include "tensors/tensor.h"
#include "translator/helpers.h"

#include "tensors/gpu/cuda_helpers.h"

namespace marian {

namespace gpu {

template <typename T>
__global__ void gSetColumns(T* out,
                            int rows,
                            int cols,
                            const IndexType* wordIndices,
                            int numIndices,
                            T value) {
  for(int bid = 0; bid < rows; bid += gridDim.x) {
    int j = bid + blockIdx.x;
    if(j < rows) {
      T* rowOut = out + j * cols;
      for(int tid = 0; tid < numIndices; tid += blockDim.x) {
        int i = tid + threadIdx.x;
        if(i < numIndices)
          rowOut[wordIndices[i]] = value;
      }
    }
  }
}

void SetColumns(Tensor in, Tensor wordIndices, float value) {
  matchOrAbort<IndexType>(wordIndices->type());

  int rows = in->shape().elements() / in->shape().back();
  int cols = in->shape().back();

  int numIndices = wordIndices->size();

  int threads = std::min(MAX_THREADS, numIndices);
  int blocks = std::min(MAX_BLOCKS, rows);

  if(in->type() == Type::float32) {
    gSetColumns<<<blocks, threads>>>(in->data<float>(), rows, cols, wordIndices->data<WordIndex>(), numIndices, value);
#if COMPILE_FP16
  } else if(in->type() == Type::float16) {
    gSetColumns<<<blocks, threads>>>(in->data<half>(),  rows, cols, wordIndices->data<WordIndex>(), numIndices, (half)value);
#endif
  } else {
    ABORT("suppressWord not implemented for type {}", in->type());
  }
}

void suppressWords(Expr probs, Expr wordIndices) {
  SetColumns(probs->val(), wordIndices->val(), NumericLimits<float>(probs->value_type()).lowest);
}

}  // namespace gpu
}  // namespace marian
