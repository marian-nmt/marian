#include <cuda.h>

#include "kernels/cuda_helpers.h"
#include "tensors/memory_piece.h"

namespace marian {

void MemoryPiece::insert(uint8_t* ptr, size_t num) {
  CUDA_CHECK(cudaMemcpy(data_, ptr, num * sizeof(uint8_t), cudaMemcpyDefault));
}
}
